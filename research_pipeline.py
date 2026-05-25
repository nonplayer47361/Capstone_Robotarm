from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from active_learning_core import IMAGE_EXTS, all_images, count_labels, predict_to_dataset, train_model, write_dataset_yaml

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "research_configs"
RUNS_ROOT = HERE / "research_runs"


@dataclass
class ResearchConfig:
    target_name: str
    source_images: str
    manual_seed_count: int
    class_id: int | None
    class_names: list[str]
    base_model: str = "yolo11n.pt"
    freeze: int = 10
    confidence: float = 0.35
    fallback_confidence: float | None = None
    manual_seed_step: int = 25
    max_manual_seed: int | None = None
    min_stage1_auto_labels: int = 1
    test_images: str | None = None
    holdout_source: str | None = None
    require_review_complete: bool = True
    auto_accept_confidence: float | None = None
    auto_accept_min_area_ratio: float = 0.0003
    auto_accept_max_area_ratio: float = 0.35
    auto_accept_min_aspect_ratio: float = 0.35
    auto_accept_max_aspect_ratio: float = 3.0
    auto_accept_require_single_detection: bool = True
    auto_accept_audit_ratio: float = 0.05
    auto_accept_consistency_iou: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "ResearchConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = cls(**data)
        cfg.validate(path)
        return cfg

    def validate(self, path: Path) -> None:
        if not self.class_names:
            raise ValueError(f"class_names is empty: {path}")
        if self.class_id is not None and not 0 <= self.class_id < len(self.class_names):
            raise ValueError(f"class_id={self.class_id} is outside class_names in {path}")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be between 0 and 1 in {path}")
        if self.fallback_confidence is not None and not 0.0 < self.fallback_confidence < 1.0:
            raise ValueError(f"fallback_confidence must be between 0 and 1 in {path}")
        if self.freeze < 0:
            raise ValueError(f"freeze must be >= 0 in {path}")
        if self.manual_seed_count <= 0:
            raise ValueError(f"manual_seed_count must be positive in {path}")
        if self.auto_accept_confidence is not None and not 0.0 < self.auto_accept_confidence < 1.0:
            raise ValueError(f"auto_accept_confidence={self.auto_accept_confidence!r} must be between 0 and 1 in {path}")
        if not 0.0 <= self.auto_accept_audit_ratio <= 1.0:
            raise ValueError(f"auto_accept_audit_ratio={self.auto_accept_audit_ratio!r} must be between 0 and 1 in {path}")
        if self.auto_accept_min_area_ratio < 0 or self.auto_accept_max_area_ratio <= self.auto_accept_min_area_ratio:
            raise ValueError(
                f"auto_accept area ratio bounds are invalid in {path}: "
                f"min={self.auto_accept_min_area_ratio!r} max={self.auto_accept_max_area_ratio!r}"
            )
        if self.auto_accept_min_aspect_ratio <= 0 or self.auto_accept_max_aspect_ratio <= self.auto_accept_min_aspect_ratio:
            raise ValueError(
                f"auto_accept aspect ratio bounds are invalid in {path}: "
                f"min={self.auto_accept_min_aspect_ratio!r} max={self.auto_accept_max_aspect_ratio!r}"
            )
        if self.auto_accept_consistency_iou < 0 or self.auto_accept_consistency_iou > 1:
            raise ValueError(f"auto_accept_consistency_iou={self.auto_accept_consistency_iou!r} must be between 0 and 1 in {path}")


def paths(target: str) -> dict[str, Path]:
    root = RUNS_ROOT / target
    return {
        "root": root,
        "sources": root / "00_sources",
        "manual_seed": root / "01_manual_seed_dataset",
        "stage1_auto": root / "02_stage1_auto_dataset",
        "stage1_reviewed": root / "03_stage1_reviewed_dataset",
        "stage2_relabel": root / "04_stage2_relabel_dataset",
        "stage2_reviewed": root / "05_stage2_reviewed_dataset",
        "stage3_relabel": root / "06_stage3_relabel_dataset",
        "stage3_reviewed": root / "07_stage3_reviewed_dataset",
        "missed_images": root / "08_missed_images",
        "final_dataset": root / "09_final_dataset",
        "detections": root / "10_detection_eval",
        "reports": root / "11_reports",
        "holdout_dataset": root / "12_holdout_test_dataset",
        "holdout_eval_dataset": root / "13_holdout_eval_dataset",
        "holdout_eval": root / "14_holdout_eval",
        "runs": root / "runs",
        "logs": root / "logs",
        "archive": root / "archive",
    }


def ensure_tree(target: str) -> None:
    for p in paths(target).values():
        p.mkdir(parents=True, exist_ok=True)


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def reset_dataset_dir(dataset_dir: Path, class_names: list[str]) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    write_dataset_yaml(dataset_dir, class_names)


def latest_existing(candidates: list[Path]) -> Path | None:
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def find_seed_model(cfg: ResearchConfig) -> Path | None:
    """Return the best available seed model for resuming from stage 2.

    Prefers the reviewed-seed model (01b) which was trained on human-corrected
    stage1 labels; falls back to the raw seed model (01) if not found.
    """
    p = paths(cfg.target_name)
    candidates = [
        p["runs"] / "01b_reviewed_seed_model" / "weights" / "best.pt",
        p["runs"] / "01_seed_model" / "weights" / "best.pt",
    ]
    candidates.extend(p["archive"].glob("old_outputs_*/*active_runs/stage1_manual50/weights/best.pt"))
    return latest_existing(candidates)


def find_reviewed_model(cfg: ResearchConfig) -> Path | None:
    p = paths(cfg.target_name)
    candidates = [
        p["runs"] / "02_reviewed_model" / "weights" / "best.pt",
    ]
    candidates.extend(p["archive"].glob("old_outputs_*/*active_runs/stage2_manual_plus_auto/weights/best.pt"))
    return latest_existing(candidates)


def find_final_model(cfg: ResearchConfig) -> Path | None:
    p = paths(cfg.target_name)
    candidates = [
        p["runs"] / "04_final_model" / "weights" / "best.pt",
        p["runs"] / "03_stage3_model" / "weights" / "best.pt",
        p["runs"] / "02_reviewed_model" / "weights" / "best.pt",
        p["runs"] / "01b_reviewed_seed_model" / "weights" / "best.pt",
        p["runs"] / "01_seed_model" / "weights" / "best.pt",
    ]
    return latest_existing(candidates)


def prepare_sources(cfg: ResearchConfig) -> Path:
    p = paths(cfg.target_name)
    src = (HERE / cfg.source_images).resolve()
    if not src.exists() or not src.is_dir():
        raise RuntimeError(f"Source image folder not found: {src}")
    dst = p["sources"] / Path(cfg.source_images).name
    dst.mkdir(parents=True, exist_ok=True)
    for img in all_images(src):
        out = dst / img.name
        if not out.exists():
            shutil.copy2(img, out)
    return src


def dataset_label_stems(dataset_dir: Path) -> set[str]:
    stems: set[str] = set()
    for split in ("train", "val"):
        labels_dir = dataset_dir / split / "labels"
        if labels_dir.exists():
            stems.update(path.stem for path in labels_dir.glob("*.txt"))
    return stems


def expected_review_stems(dataset_dir: Path, all_labels: bool, backup_dir: Path | None) -> list[str]:
    expected = dataset_label_stems(dataset_dir)
    if not all_labels and backup_dir is not None:
        expected -= dataset_label_stems(backup_dir)
    return sorted(expected)


def auto_accept_rules(cfg: ResearchConfig) -> dict[str, object] | None:
    if cfg.auto_accept_confidence is None:
        return None
    return {
        "confidence": cfg.auto_accept_confidence,
        "min_area_ratio": cfg.auto_accept_min_area_ratio,
        "max_area_ratio": cfg.auto_accept_max_area_ratio,
        "min_aspect_ratio": cfg.auto_accept_min_aspect_ratio,
        "max_aspect_ratio": cfg.auto_accept_max_aspect_ratio,
        "require_single_detection": cfg.auto_accept_require_single_detection,
        "audit_ratio": cfg.auto_accept_audit_ratio,
        "consistency_iou": cfg.auto_accept_consistency_iou,
    }


def select_evenly(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or not paths:
        return []
    if len(paths) <= count:
        return list(paths)
    if count == 1:
        return [paths[len(paths) // 2]]
    indices = sorted({round(i * (len(paths) - 1) / (count - 1)) for i in range(count)})
    selected = [paths[i] for i in indices]
    # Rounding can rarely collapse indices. Fill from the middle-out leftovers.
    if len(selected) < count:
        used = {p.resolve() for p in selected}
        leftovers = [p for p in paths if p.resolve() not in used]
        selected.extend(leftovers[: count - len(selected)])
    return selected[:count]


def select_unlabeled_seed_images(source: Path, dataset_dir: Path, count: int) -> list[Path]:
    done = dataset_label_stems(dataset_dir)
    candidates = [path for path in all_images(source) if path.stem not in done]
    return select_evenly(candidates, count)


def assert_review_complete(dataset_dir: Path, expected: list[str]) -> None:
    if not expected:
        return
    status = read_review_status(dataset_dir)
    missing = [stem for stem in expected if not bool(status.get(stem))]
    if missing:
        sample = ", ".join(missing[:10])
        raise RuntimeError(
            f"Review incomplete for {dataset_dir}: "
            f"{len(expected) - len(missing)}/{len(expected)} checked. "
            f"Unchecked sample: {sample}"
        )


def read_review_status(dataset_dir: Path) -> dict[str, bool]:
    status_path = dataset_dir / "review_status.json"
    if not status_path.exists():
        return {}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Review status file is invalid: {status_path}") from exc
    return {str(k): bool(v) for k, v in status.items()}


def labeler_target(cfg: ResearchConfig) -> str:
    if cfg.target_name in {"blueberry", "strawberry"}:
        return cfg.target_name
    if cfg.class_id == 0:
        return "blueberry"
    if cfg.class_id == 1:
        return "strawberry"
    raise RuntimeError(
        f"Cannot determine labeler target for config '{cfg.target_name}' (class_id={cfg.class_id}). "
        "Set class_id to 0 (blueberry) or 1 (strawberry)."
    )


def manual_label_command(cfg: ResearchConfig, image_dir: Path, dataset_dir: Path) -> list[str]:
    if cfg.target_name in {"blueberry", "strawberry"}:
        return [
            sys.executable,
            str(HERE / "start_labeling.py"),
            "--images",
            str(image_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--target",
            labeler_target(cfg),
        ]

    class_id = cfg.class_id if cfg.class_id is not None else 0
    if not 0 <= class_id < len(cfg.class_names):
        raise RuntimeError(
            f"Cannot open generic labeler for config '{cfg.target_name}': "
            f"class_id={class_id}, class_names={cfg.class_names}"
        )
    return [
        sys.executable,
        str(HERE / "label_clay_ball.py"),
        "--images",
        str(image_dir),
        "--dataset-dir",
        str(dataset_dir),
        "--class-id",
        str(class_id),
        "--class-name",
        cfg.class_names[class_id],
        "--no-auto-assist",
    ]


def manual_label_paths(cfg: ResearchConfig, image_paths: list[Path], manual: Path, name: str) -> None:
    if not image_paths:
        return
    p = paths(cfg.target_name)
    selection_dir = p["sources"] / f"{cfg.target_name}_{name}_selection"
    if selection_dir.exists():
        shutil.rmtree(selection_dir)
    selection_dir.mkdir(parents=True, exist_ok=True)
    for src in image_paths:
        shutil.copy2(src, selection_dir / src.name)

    print(f"[manual] opening labeler for {len(image_paths)} evenly selected seed images")
    before = count_labels(manual)
    subprocess.run(manual_label_command(cfg, selection_dir, manual), check=True)
    write_dataset_yaml(manual, cfg.class_names)
    after = count_labels(manual)
    if after <= before:
        raise RuntimeError(f"Manual labeling did not add labels. labels={after}, selection={selection_dir}")


def bootstrap_manual_seed(cfg: ResearchConfig) -> None:
    p = paths(cfg.target_name)
    source = prepare_sources(cfg)
    manual = p["manual_seed"]
    existing = count_labels(manual)
    if existing >= cfg.manual_seed_count:
        write_dataset_yaml(manual, cfg.class_names)
        print(f"[manual] existing seed labels={existing}")
        return

    manual.mkdir(parents=True, exist_ok=True)
    selected = select_unlabeled_seed_images(source, manual, cfg.manual_seed_count)
    manual_label_paths(cfg, selected, manual, "manual_seed")
    if count_labels(manual) < cfg.manual_seed_count:
        raise RuntimeError(
            f"Manual seed is incomplete: {count_labels(manual)}/{cfg.manual_seed_count}. "
            "Finish the seed labels before running the full pipeline."
        )


def expand_manual_seed(cfg: ResearchConfig, source: Path, manual: Path) -> bool:
    current = count_labels(manual)
    max_seed = cfg.max_manual_seed or len(all_images(source))
    if current >= max_seed:
        print(f"[manual] cannot expand seed: current={current}, max={max_seed}")
        return False
    target = min(max_seed, current + max(1, cfg.manual_seed_step))
    selected = select_unlabeled_seed_images(source, manual, target - current)
    manual_label_paths(cfg, selected, manual, f"manual_expand_{current}_to_{target}")
    after = count_labels(manual)
    expanded = after > current
    print(f"[manual] seed labels: {current} -> {after}")
    return expanded


def review(
    dataset_dir: Path,
    all_labels: bool = True,
    backup_dir: Path | None = None,
    require_complete: bool = True,
) -> None:
    expected = expected_review_stems(dataset_dir, all_labels, backup_dir)
    if not expected:
        print(f"[review] no labels to review: {dataset_dir}")
        return
    status = read_review_status(dataset_dir)
    if all(bool(status.get(stem)) for stem in expected):
        print(f"[review] all {len(expected)} labels already reviewed/auto-accepted: {dataset_dir}")
        return
    cmd = [
        sys.executable,
        str(HERE / "review_label_gallery.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--sequential",
        "--unchecked-only",
    ]
    if all_labels:
        cmd.append("--all")
    elif backup_dir:
        cmd += ["--backup-dir", str(backup_dir)]
    subprocess.run(cmd, check=True)
    if require_complete:
        assert_review_complete(dataset_dir, expected)


def manual_label_missed(cfg: ResearchConfig, missed_dir: Path, dataset_dir: Path) -> None:
    if not missed_dir.exists() or not any(missed_dir.iterdir()):
        print("[missed] no missed images")
        return
    missed_count = len(all_images(missed_dir))
    before = count_labels(dataset_dir)
    subprocess.run(manual_label_command(cfg, missed_dir, dataset_dir), check=True)
    after = count_labels(dataset_dir)
    if after < before + missed_count:
        raise RuntimeError(
            f"Missed-image manual labeling incomplete: added {after - before}/{missed_count} labels. "
            "Finish the missed-image labels before final training."
        )


def copy_holdout_from_source_if_ready(cfg: ResearchConfig, dataset_dir: Path, expected: int) -> bool:
    if not cfg.holdout_source:
        return False
    source_holdout = paths(cfg.holdout_source)["holdout_dataset"]
    if not source_holdout.exists() or count_labels(source_holdout) < expected:
        return False
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    copytree_replace(source_holdout, dataset_dir)
    write_dataset_yaml(dataset_dir, cfg.class_names)
    print(f"[holdout] copied {count_labels(dataset_dir)} labels from holdout_source={cfg.holdout_source}")
    return True


def label_holdout(cfg: ResearchConfig) -> None:
    if not cfg.test_images:
        raise RuntimeError("No test_images folder is configured for this target.")
    ensure_tree(cfg.target_name)
    p = paths(cfg.target_name)
    test_dir = (HERE / cfg.test_images).resolve()
    if not test_dir.exists() or not test_dir.is_dir():
        raise RuntimeError(f"Test image folder not found: {test_dir}")
    expected = len(all_images(test_dir))
    if expected == 0:
        raise RuntimeError(f"No test images found: {test_dir}")
    dataset_dir = p["holdout_dataset"]
    if copy_holdout_from_source_if_ready(cfg, dataset_dir, expected):
        return
    if cfg.holdout_source:
        print(f"[holdout] holdout_source={cfg.holdout_source} not ready, falling back to manual labeling")
    before = count_labels(dataset_dir)
    print(f"[holdout] target={cfg.target_name} test_images={expected} existing_labels={before}")
    subprocess.run(manual_label_command(cfg, test_dir, dataset_dir), check=True)
    write_dataset_yaml(dataset_dir, cfg.class_names)
    after = count_labels(dataset_dir)
    if after < expected:
        raise RuntimeError(f"Holdout labeling incomplete: {after}/{expected} labels. Finish all holdout labels before evaluation.")
    print(f"[holdout] labels ready: {after}/{expected} at {dataset_dir}")


def image_for_label(dataset_dir: Path, split: str, stem: str) -> Path | None:
    image_dir = dataset_dir / split / "images"
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def build_eval_dataset_from_labels(label_dataset: Path, eval_dataset: Path, class_names: list[str]) -> int:
    if eval_dataset.exists():
        shutil.rmtree(eval_dataset)
    write_dataset_yaml(eval_dataset, class_names)
    count = 0
    for split in ("train", "val"):
        label_dir = label_dataset / split / "labels"
        if not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_for_label(label_dataset, split, label_path.stem)
            if not image_path:
                continue
            dst_img = eval_dataset / "val" / "images" / image_path.name
            dst_lbl = eval_dataset / "val" / "labels" / label_path.name
            shutil.copy2(image_path, dst_img)
            shutil.copy2(label_path, dst_lbl)
            count += 1
    return count


def evaluate_holdout(cfg: ResearchConfig, model_path: str | None = None) -> dict[str, object]:
    from ultralytics import YOLO

    ensure_tree(cfg.target_name)
    p = paths(cfg.target_name)
    model = Path(model_path).resolve() if model_path else find_final_model(cfg)
    if model is None or not model.exists():
        raise RuntimeError("No model was found for holdout evaluation. Train a model first or pass --model.")
    if not cfg.test_images:
        raise RuntimeError("No test_images folder is configured for this target.")
    expected = len(all_images((HERE / cfg.test_images).resolve()))
    holdout_labels = count_labels(p["holdout_dataset"])
    if holdout_labels < expected:
        hint = "Run BAT action 4 first."
        if cfg.holdout_source:
            hint = f"Run BAT action 4 after the base holdout '{cfg.holdout_source}' is ready."
        raise RuntimeError(
            f"Holdout labels are not ready: {holdout_labels}/{expected}. {hint}"
        )

    eval_count = build_eval_dataset_from_labels(p["holdout_dataset"], p["holdout_eval_dataset"], cfg.class_names)
    if eval_count == 0:
        raise RuntimeError(f"No holdout labels could be copied for evaluation: {p['holdout_dataset']}")

    yaml_path = write_dataset_yaml(p["holdout_eval_dataset"], cfg.class_names)
    eval_project = p["holdout_eval"]
    eval_project.mkdir(parents=True, exist_ok=True)
    metrics = YOLO(str(model)).val(
        data=str(yaml_path.resolve()),
        split="val",
        imgsz=640,
        batch=8,
        project=str(eval_project.resolve()),
        name="val",
        exist_ok=True,
        plots=True,
        save_json=False,
    )
    box = getattr(metrics, "box", None)
    metric_values: dict[str, float | None] = {}
    for name in ("mp", "mr", "map50", "map75", "map"):
        value = getattr(box, name, None) if box is not None else None
        metric_values[name] = float(value) if value is not None else None
    summary = {
        "target": cfg.target_name,
        "model": str(model),
        "holdout_images": expected,
        "holdout_labels": holdout_labels,
        "eval_dataset": str(p["holdout_eval_dataset"]),
        "eval_dir": str(eval_project / "val"),
        "metrics": metric_values,
    }
    out_json = p["reports"] / "holdout_eval_summary.json"
    out_md = p["reports"] / "holdout_eval_report.md"
    p["reports"].mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {cfg.target_name} Holdout Evaluation",
        "",
        f"- model: `{model}`",
        f"- holdout images: `{expected}`",
        f"- holdout labels: `{holdout_labels}`",
        f"- mAP50: `{metric_values.get('map50')}`",
        f"- mAP50-95: `{metric_values.get('map')}`",
        f"- precision(mp): `{metric_values.get('mp')}`",
        f"- recall(mr): `{metric_values.get('mr')}`",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[holdout_eval] report: {out_md}")
    return summary


def evaluate_source_pool(cfg: ResearchConfig, model_path: str | None = None) -> dict[str, object]:
    ensure_tree(cfg.target_name)
    p = paths(cfg.target_name)
    model = Path(model_path).resolve() if model_path else find_final_model(cfg)
    if model is None or not model.exists():
        raise RuntimeError("No model was found for source-pool evaluation. Train a model first or pass --model.")
    source = (HERE / cfg.source_images).resolve()
    if not source.exists():
        raise RuntimeError(f"Source image folder not found: {source}")
    summary = detect_eval(str(model), source, p["detections"] / "source_pool", cfg.confidence)
    report_data = {
        "target": cfg.target_name,
        "model": str(model),
        "source_images": len(all_images(source)),
        "source_dir": str(source),
        "note": "This is a full source-pool detection smoke test, not an unbiased mAP score.",
        "eval": summary,
    }
    p["reports"].mkdir(parents=True, exist_ok=True)
    out_json = p["reports"] / "source_pool_detection_summary.json"
    out_md = p["reports"] / "source_pool_detection_report.md"
    out_json.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {cfg.target_name} Source Pool Detection",
        "",
        f"- model: `{model}`",
        f"- source images: `{report_data['source_images']}`",
        f"- success: `{summary.get('success')}`",
        f"- failed: `{summary.get('failed')}`",
        "",
        "This checks whether the model detects objects across the full image pool. It is not an unbiased accuracy score because the source pool participates in active learning.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[source_eval] report: {out_md}")
    return report_data


def detect_eval(model_path: str, images_dir: Path, out_dir: Path, conf: float) -> dict[str, object]:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    success: list[str] = []
    failed: list[str] = []
    for img in all_images(images_dir):
        result = model.predict(str(img), conf=conf, verbose=False, save=True, project=str(out_dir), name="predictions", exist_ok=True)[0]
        if result.boxes:
            success.append(img.name)
        else:
            failed.append(img.name)
    (out_dir / "success.txt").write_text("\n".join(success), encoding="utf-8")
    (out_dir / "failed.txt").write_text("\n".join(failed), encoding="utf-8")
    return {"success": len(success), "failed": len(failed), "success_list": str(out_dir / "success.txt"), "failed_list": str(out_dir / "failed.txt")}


def report(cfg: ResearchConfig, summary: dict[str, object]) -> Path:
    p = paths(cfg.target_name)
    p["reports"].mkdir(parents=True, exist_ok=True)
    json_path = p["reports"] / "summary.json"
    md_path = p["reports"] / "research_report.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {cfg.target_name} YOLO Research Report", "", "## Summary", ""]
    for k, v in summary.items():
        lines.append(f"- `{k}`: {v}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def run(cfg: ResearchConfig, no_review: bool, epochs1: int, epochs2: int, epochs_final: int, start_stage: int = 1, stage1_only: bool = False) -> None:
    ensure_tree(cfg.target_name)
    source = prepare_sources(cfg)
    bootstrap_manual_seed(cfg)
    p = paths(cfg.target_name)
    conf = cfg.confidence
    review_min_rules = auto_accept_rules(cfg)

    if start_stage <= 1:
        attempt = 1
        while True:
            seed_count = count_labels(p["manual_seed"])
            print(f"[stage1] attempt={attempt} manual_seed_labels={seed_count}")
            copytree_replace(p["manual_seed"], p["stage1_auto"])
            model1 = train_model(p["stage1_auto"], cfg.base_model, p["runs"], "01_seed_model", epochs1, class_names=cfg.class_names, freeze=cfg.freeze)
            stage1_stats = predict_to_dataset(
                source,
                p["stage1_auto"],
                model1,
                conf,
                0.2,
                cfg.class_id,
                True,
                True,
                class_names=cfg.class_names,
                missed_list_path=p["stage1_auto"] / "stage1_missed.txt",
            )
            if stage1_stats.labeled < cfg.min_stage1_auto_labels and cfg.fallback_confidence and cfg.fallback_confidence < conf:
                print(f"[stage1] auto labels={stage1_stats.labeled}; retrying with fallback_conf={cfg.fallback_confidence}")
                stage1_stats = predict_to_dataset(
                    source,
                    p["stage1_auto"],
                    model1,
                    cfg.fallback_confidence,
                    0.2,
                    cfg.class_id,
                    True,
                    True,
                    class_names=cfg.class_names,
                    missed_list_path=p["stage1_auto"] / "stage1_missed.txt",
                )
            if stage1_stats.labeled >= cfg.min_stage1_auto_labels:
                break
            print(
                f"[stage1] auto labels={stage1_stats.labeled}, "
                f"required={cfg.min_stage1_auto_labels}. More manual seed labels are needed."
            )
            if not expand_manual_seed(cfg, source, p["manual_seed"]):
                raise RuntimeError(
                    "Stage1 auto-labeling did not produce enough labels and manual seed expansion is not possible. "
                    "Inspect the seed labels/model or adjust fallback_confidence/min_stage1_auto_labels."
                )
            attempt += 1
        if not no_review:
            review(
                p["stage1_auto"],
                all_labels=False,
                backup_dir=p["manual_seed"],
                require_complete=cfg.require_review_complete,
            )
        copytree_replace(p["stage1_auto"], p["stage1_reviewed"])
        model1_rev = train_model(
            p["stage1_reviewed"], cfg.base_model, p["runs"], "01b_reviewed_seed_model", epochs1, class_names=cfg.class_names, freeze=cfg.freeze
        )
        if stage1_only:
            print("\n" + "=" * 60)
            print(f"[stage1-only] Stage 1 완료 (학습 + 자동라벨 + 리뷰 + 재학습)")
            print(f"  타겟     : {cfg.target_name}")
            print(f"  리뷰모델 : {p['runs'] / '01b_reviewed_seed_model' / 'weights' / 'best.pt'}")
            print(f"  다음 단계: python research_pipeline.py --config research_configs/{cfg.target_name}.json --run --start-stage 2")
            print("=" * 60 + "\n")
            return
    else:
        model1_rev_path = find_seed_model(cfg)
        if model1_rev_path is None:
            raise RuntimeError("Cannot skip stage 1 because no seed model was found.")
        model1_rev = str(model1_rev_path)
        print(f"[resume] skipping stage 1, using seed model: {model1_rev}")

    if start_stage <= 2:
        reset_dataset_dir(p["stage2_relabel"], cfg.class_names)
        predict_to_dataset(
            source,
            p["stage2_relabel"],
            model1_rev,
            conf,
            0.2,
            cfg.class_id,
            False,
            True,
            class_names=cfg.class_names,
            auto_accept_rules=review_min_rules,
        )
        if not no_review:
            review(p["stage2_relabel"], all_labels=True, require_complete=cfg.require_review_complete)
        copytree_replace(p["stage2_relabel"], p["stage2_reviewed"])
        model2 = train_model(p["stage2_reviewed"], cfg.base_model, p["runs"], "02_reviewed_model", epochs2, class_names=cfg.class_names, freeze=cfg.freeze)
    else:
        model2_path = find_reviewed_model(cfg)
        if model2_path is None:
            raise RuntimeError("Cannot skip stage 2 because no reviewed model was found.")
        model2 = str(model2_path)
        print(f"[resume] skipping stage 2, using reviewed model: {model2}")

    if start_stage <= 3:
        reset_dataset_dir(p["stage3_relabel"], cfg.class_names)
        predict_to_dataset(
            source,
            p["stage3_relabel"],
            model2,
            conf,
            0.2,
            cfg.class_id,
            False,
            True,
            class_names=cfg.class_names,
            auto_accept_rules=review_min_rules,
        )
        if not no_review:
            review(p["stage3_relabel"], all_labels=True, require_complete=cfg.require_review_complete)
        copytree_replace(p["stage3_relabel"], p["stage3_reviewed"])

    model3 = train_model(p["stage3_reviewed"], cfg.base_model, p["runs"], "03_stage3_model", epochs2, class_names=cfg.class_names, freeze=cfg.freeze)
    reset_dataset_dir(p["final_dataset"], cfg.class_names)
    final_stats = predict_to_dataset(
        source,
        p["final_dataset"],
        model3,
        conf,
        0.2,
        cfg.class_id,
        False,
        True,
        class_names=cfg.class_names,
        missed_dir=p["missed_images"],
        missed_list_path=p["final_dataset"] / "final_missed.txt",
    )
    manual_label_missed(cfg, p["missed_images"], p["final_dataset"])
    final_model = train_model(p["final_dataset"], cfg.base_model, p["runs"], "04_final_model", epochs_final, class_names=cfg.class_names, freeze=cfg.freeze)
    source_eval_summary = evaluate_source_pool(cfg, final_model)
    holdout_summary: dict[str, object] | None = None
    if cfg.test_images:
        expected_holdout = len(all_images((HERE / cfg.test_images).resolve()))
        if count_labels(p["holdout_dataset"]) < expected_holdout:
            copy_holdout_from_source_if_ready(cfg, p["holdout_dataset"], expected_holdout)
        holdout_labels = count_labels(p["holdout_dataset"])
        if holdout_labels >= expected_holdout:
            holdout_summary = evaluate_holdout(cfg, final_model)
        else:
            print(
                f"[holdout_eval] skipped: holdout labels are incomplete "
                f"({holdout_labels}/{expected_holdout}). Run BAT action 4 first, then action 6."
            )

    summary = {
        "target": cfg.target_name,
        "manual_seed_labels": count_labels(p["manual_seed"]),
        "stage1_auto_labels": count_labels(p["stage1_auto"]),
        "stage2_relabel_labels": count_labels(p["stage2_relabel"]),
        "stage3_relabel_labels": count_labels(p["stage3_relabel"]),
        "final_dataset_labels": count_labels(p["final_dataset"]),
        "final_missed": final_stats.low_conf,
        "final_model": final_model,
        "source_pool_eval": source_eval_summary,
        "holdout_eval": holdout_summary,
    }
    print(f"[done] report: {report(cfg, summary)}")


def status(cfg: ResearchConfig) -> None:
    ensure_tree(cfg.target_name)
    source = (HERE / cfg.source_images).resolve()
    source_count = len(all_images(source)) if source.exists() else 0
    print(f"target={cfg.target_name} source_images={source_count:4d} path={source}")
    if cfg.test_images:
        test = (HERE / cfg.test_images).resolve()
        test_count = len(all_images(test)) if test.exists() else 0
        print(f"test_images={test_count:4d} path={test}")
    for key, path in paths(cfg.target_name).items():
        if key.endswith("dataset") or key in {
            "manual_seed",
            "stage1_auto",
            "stage1_reviewed",
            "stage2_relabel",
            "stage2_reviewed",
            "stage3_relabel",
            "stage3_reviewed",
            "final_dataset",
            "holdout_dataset",
            "holdout_eval_dataset",
        }:
            print(f"{key:18s} labels={count_labels(path):4d} path={path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic YOLO research pipeline")
    p.add_argument("--config", default=str(CONFIG_DIR / "strawberry.json"))
    p.add_argument("--init", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--label-holdout", action="store_true")
    p.add_argument("--eval-holdout", action="store_true")
    p.add_argument("--eval-source", action="store_true")
    p.add_argument("--model", default="", help="Optional model path for --eval-holdout. Defaults to latest final/reviewed model.")
    p.add_argument("--no-review", action="store_true")
    p.add_argument("--stage1-only", action="store_true", help="Stage 1 (train+autolabel+review+retrain) 만 실행 후 정지. 저녁에 실행, 이후 --start-stage 2 로 야간 재개.")
    p.add_argument("--epochs1", type=int, default=40)
    p.add_argument("--epochs2", type=int, default=60)
    p.add_argument("--epochs-final", type=int, default=80)
    p.add_argument("--start-stage", type=int, choices=[1, 2, 3], default=1, help="Resume from stage 2 or 3 when earlier outputs already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ResearchConfig.load(Path(args.config))
    if args.init:
        ensure_tree(cfg.target_name)
        prepare_sources(cfg)
        bootstrap_manual_seed(cfg)
    if args.run:
        run(cfg, args.no_review, args.epochs1, args.epochs2, args.epochs_final, args.start_stage, args.stage1_only)
    if args.label_holdout:
        label_holdout(cfg)
    if args.eval_holdout:
        evaluate_holdout(cfg, args.model or None)
    if args.eval_source:
        evaluate_source_pool(cfg, args.model or None)
    if args.status or not any([args.init, args.run, args.label_holdout, args.eval_holdout, args.eval_source]):
        status(cfg)


if __name__ == "__main__":
    main()
