from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELING_TOOLS = HERE.parent
if str(LABELING_TOOLS) not in sys.path:
    sys.path.insert(0, str(LABELING_TOOLS))

from active_learning_core import all_images, count_labels, read_image, train_model, write_dataset_yaml, yolo_line_from_xyxy  # noqa: E402
from dataset_utils import ensure_yolo_dirs, label_exists_in_dataset, save_labeled  # noqa: E402


@dataclass
class FullAutoConfig:
    target_name: str
    source_images: str
    seed_dataset: str
    class_id: int
    class_names: list[str]
    base_model: str = "yolo11n.pt"
    freeze: int = 10
    iterations: int = 4
    epochs: int = 40
    final_epochs: int = 60
    train_confidence: float = 0.25
    accept_confidence: float = 0.88
    min_area_ratio: float = 0.0003
    max_area_ratio: float = 0.35
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 3.0
    require_single_detection: bool = True
    val_ratio: float = 0.2
    min_new_labels: int = 5

    @classmethod
    def load(cls, path: Path) -> "FullAutoConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = cls(**data)
        cfg.validate(path)
        return cfg

    def validate(self, path: Path) -> None:
        if not self.class_names:
            raise ValueError(f"class_names is empty: {path}")
        if not 0 <= self.class_id < len(self.class_names):
            raise ValueError(f"class_id={self.class_id} is outside class_names: {path}")
        if not 0.0 < self.train_confidence < 1.0:
            raise ValueError(f"train_confidence must be between 0 and 1: {path}")
        if not 0.0 < self.accept_confidence < 1.0:
            raise ValueError(f"accept_confidence must be between 0 and 1: {path}")
        if self.max_area_ratio <= self.min_area_ratio:
            raise ValueError(f"area ratio bounds are invalid: {path}")
        if self.max_aspect_ratio <= self.min_aspect_ratio:
            raise ValueError(f"aspect ratio bounds are invalid: {path}")


def resolve_tool_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (LABELING_TOOLS / path).resolve()


def output_paths(target: str) -> dict[str, Path]:
    root = HERE / "runs" / target
    return {
        "root": root,
        "datasets": root / "datasets",
        "models": root / "models",
        "reports": root / "reports",
    }


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_dataset(src: Path, dst: Path, class_names: list[str]) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    write_dataset_yaml(dst, class_names)


def label_stems(dataset_dir: Path) -> set[str]:
    stems: set[str] = set()
    for split in ("train", "val"):
        labels = dataset_dir / split / "labels"
        if labels.exists():
            stems.update(p.stem for p in labels.glob("*.txt"))
    return stems


def passes_accept_rules(cfg: FullAutoConfig, boxes: list, img_w: int, img_h: int) -> tuple[bool, str, dict[str, float]]:
    if not boxes:
        return False, "no_box", {}
    if cfg.require_single_detection and len(boxes) != 1:
        return False, "multiple_boxes", {"box_count": float(len(boxes))}
    box = boxes[0]
    confidence = float(box.conf[0])
    if confidence < cfg.accept_confidence:
        return False, "low_confidence", {"confidence": confidence}
    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
    bw = max(0.0, x2 - x1) / max(1, img_w)
    bh = max(0.0, y2 - y1) / max(1, img_h)
    area = bw * bh
    aspect = bw / bh if bh > 0 else 999.0
    metrics = {
        "confidence": confidence,
        "area_ratio": area,
        "aspect_ratio": aspect,
        "box_count": float(len(boxes)),
    }
    if area < cfg.min_area_ratio:
        return False, "area_too_small", metrics
    if area > cfg.max_area_ratio:
        return False, "area_too_large", metrics
    if aspect < cfg.min_aspect_ratio:
        return False, "aspect_too_narrow", metrics
    if aspect > cfg.max_aspect_ratio:
        return False, "aspect_too_wide", metrics
    return True, "accepted", metrics


def auto_label_accepted_only(
    *,
    cfg: FullAutoConfig,
    source_dir: Path,
    base_dataset: Path,
    out_dataset: Path,
    model_path: str,
    accepted_txt: Path,
    rejected_txt: Path,
) -> dict[str, object]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"Missing ultralytics: {exc}")

    copy_dataset(base_dataset, out_dataset, cfg.class_names)
    model = YOLO(model_path)
    known = label_stems(out_dataset)
    images = [p for p in all_images(source_dir) if p.stem not in known]
    accepted: list[str] = []
    rejected: list[str] = []
    reason_counts: dict[str, int] = {}
    t0 = time.monotonic()

    print(f"[auto] candidates={len(images)} existing_labels={len(known)}")
    for idx, img_path in enumerate(images, start=1):
        img = read_image(img_path)
        if img is None:
            reason = "read_failed"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            rejected.append(f"{img_path.name}\t{reason}")
            continue
        result = model.predict(img, conf=cfg.train_confidence, verbose=False)[0]
        boxes = sorted(result.boxes, key=lambda b: float(b.conf[0]), reverse=True)
        ok, reason, metrics = passes_accept_rules(cfg, boxes, img.shape[1], img.shape[0])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if ok:
            box = boxes[0]
            line = yolo_line_from_xyxy(cfg.class_id, tuple(float(v) for v in box.xyxy[0]), img.shape[1], img.shape[0])
            save_labeled(img, [line], img_path.stem, out_dataset, cfg.val_ratio)
            accepted.append(
                f"{img_path.name}\tconf={metrics.get('confidence', 0):.4f}"
                f"\tarea={metrics.get('area_ratio', 0):.5f}"
                f"\taspect={metrics.get('aspect_ratio', 0):.3f}"
            )
        else:
            rejected.append(f"{img_path.name}\t{reason}\t{json.dumps(metrics, ensure_ascii=False)}")
        if idx % 50 == 0 or idx == len(images):
            print(f"  [{idx:4d}/{len(images):4d}] accepted={len(accepted)} rejected={len(rejected)}")

    accepted_txt.write_text("\n".join(accepted), encoding="utf-8")
    rejected_txt.write_text("\n".join(rejected), encoding="utf-8")
    write_dataset_yaml(out_dataset, cfg.class_names)
    return {
        "candidates": len(images),
        "new_accepted": len(accepted),
        "rejected": len(rejected),
        "total_labels": count_labels(out_dataset),
        "reason_counts": reason_counts,
        "seconds": round(time.monotonic() - t0, 2),
        "accepted_list": str(accepted_txt.resolve()),
        "rejected_list": str(rejected_txt.resolve()),
        "dataset": str(out_dataset.resolve()),
    }


def write_reports(cfg: FullAutoConfig, summary: dict[str, object]) -> None:
    reports = output_paths(cfg.target_name)["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "full_auto_summary.json"
    md_path = reports / "full_auto_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Full Auto Summary - {cfg.target_name}",
        "",
        f"- Source images: {summary.get('source_images')}",
        f"- Seed labels: {summary.get('seed_labels')}",
        f"- Final labels: {summary.get('final_labels')}",
        f"- Final model: `{summary.get('final_model', '')}`",
        "",
        "## Iterations",
        "",
        "| Iteration | Candidate Images | New Accepted | Rejected | Total Labels |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary.get("iterations", []):
        lines.append(
            f"| {item['iteration']} | {item['candidates']} | {item['new_accepted']} | "
            f"{item['rejected']} | {item['total_labels']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {json_path}")
    print(f"[report] {md_path}")


def run(cfg: FullAutoConfig) -> None:
    source_dir = resolve_tool_path(cfg.source_images)
    seed_dataset = resolve_tool_path(cfg.seed_dataset)
    if not source_dir.exists():
        raise SystemExit(f"Source image folder not found: {source_dir}")
    if not seed_dataset.exists() or count_labels(seed_dataset) == 0:
        raise SystemExit(
            "Seed dataset is missing or empty. Finish manual seed labeling first:\n"
            f"  {seed_dataset}"
        )

    out = output_paths(cfg.target_name)
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)

    seed_out = out["datasets"] / "iter_00_seed"
    copy_dataset(seed_dataset, seed_out, cfg.class_names)
    current_dataset = seed_out
    source_count = len(all_images(source_dir))
    iterations: list[dict[str, object]] = []

    for i in range(1, cfg.iterations + 1):
        model_name = f"iter_{i - 1:02d}_model"
        print("\n" + "=" * 60)
        print(f"[iter {i}] training on labels={count_labels(current_dataset)}")
        model_path = train_model(
            current_dataset,
            cfg.base_model,
            out["models"],
            model_name,
            cfg.epochs,
            class_names=cfg.class_names,
            freeze=cfg.freeze,
        )

        next_dataset = out["datasets"] / f"iter_{i:02d}_auto"
        accepted_txt = out["reports"] / f"iter_{i:02d}_accepted.txt"
        rejected_txt = out["reports"] / f"iter_{i:02d}_rejected.txt"
        stats = auto_label_accepted_only(
            cfg=cfg,
            source_dir=source_dir,
            base_dataset=current_dataset,
            out_dataset=next_dataset,
            model_path=model_path,
            accepted_txt=accepted_txt,
            rejected_txt=rejected_txt,
        )
        stats["iteration"] = i
        stats["model_used"] = model_path
        iterations.append(stats)
        current_dataset = next_dataset
        if int(stats["new_accepted"]) < cfg.min_new_labels:
            print(
                f"[stop] new accepted labels={stats['new_accepted']} "
                f"< min_new_labels={cfg.min_new_labels}"
            )
            break

    print("\n" + "=" * 60)
    print(f"[final] training final model on labels={count_labels(current_dataset)}")
    final_model = train_model(
        current_dataset,
        cfg.base_model,
        out["models"],
        "final_model",
        cfg.final_epochs,
        class_names=cfg.class_names,
        freeze=cfg.freeze,
    )

    summary = {
        "target": cfg.target_name,
        "source_images": source_count,
        "seed_dataset": str(seed_dataset.resolve()),
        "seed_labels": count_labels(seed_out),
        "final_dataset": str(current_dataset.resolve()),
        "final_labels": count_labels(current_dataset),
        "final_model": final_model,
        "config": cfg.__dict__,
        "iterations": iterations,
    }
    write_reports(cfg, summary)


def status(cfg: FullAutoConfig) -> None:
    out = output_paths(cfg.target_name)
    source_dir = resolve_tool_path(cfg.source_images)
    seed_dataset = resolve_tool_path(cfg.seed_dataset)
    print(f"target        : {cfg.target_name}")
    print(f"source images : {len(all_images(source_dir)) if source_dir.exists() else 0} path={source_dir}")
    print(f"seed labels   : {count_labels(seed_dataset) if seed_dataset.exists() else 0} path={seed_dataset}")
    for dataset in sorted((out["datasets"]).glob("iter_*")) if out["datasets"].exists() else []:
        print(f"{dataset.name:14s}: labels={count_labels(dataset):4d} path={dataset}")
    final_model = out["models"] / "final_model" / "weights" / "best.pt"
    print(f"final model   : {'exists' if final_model.exists() else 'missing'} path={final_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-auto accepted-only YOLO expansion experiment")
    parser.add_argument("--config", required=True, help="Config JSON path")
    parser.add_argument("--status", action="store_true", help="Show current full-auto run status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FullAutoConfig.load(Path(args.config).resolve())
    if args.status:
        status(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
