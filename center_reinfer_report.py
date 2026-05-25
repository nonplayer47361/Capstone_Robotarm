"""
Re-run YOLO detection on manually labeled holdout datasets and report
center-point error metrics.

This report is intended for robot-arm use cases where the object center is
more important than tight-box IoU. The default comparison evaluates:

  - reviewed pipeline final models
  - fullauto pipeline final models

Custom model paths can also be supplied for later robot-camera models.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from dataset_utils import IMAGE_EXTS, load_label_from_dataset  # noqa: E402


TARGET_CLASS_ID = {
    "blueberry": 0,
    "strawberry": 1,
}


@dataclass(frozen=True)
class EvalJob:
    target: str
    model_name: str
    model_path: Path
    holdout_dataset: Path
    class_id: int


def resolve_tool_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return TOOL_DIR / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def holdout_images(dataset_dir: Path) -> list[Path]:
    images: list[Path] = []
    for split in ("train", "val"):
        image_dir = dataset_dir / split / "images"
        if image_dir.exists():
            images.extend(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return sorted(images, key=lambda p: p.stem)


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def default_holdout(target: str) -> Path:
    return TOOL_DIR / "research_runs" / target / "12_holdout_test_dataset"


def default_model_path(target: str, model_name: str) -> Path:
    if model_name == "reviewed":
        return TOOL_DIR / "research_runs" / target / "runs" / "04_final_model" / "weights" / "best.pt"
    if model_name == "fullauto":
        return TOOL_DIR / "fullauto" / "runs" / target / "models" / "final_model" / "weights" / "best.pt"
    raise ValueError(f"Unknown default model name: {model_name}")


def load_growth_summary(target: str) -> dict[str, Any]:
    reviewed = read_json(TOOL_DIR / "research_runs" / target / "11_reports" / "summary.json")
    fullauto = read_json(TOOL_DIR / "fullauto" / "runs" / target / "reports" / "full_auto_summary.json")
    return {
        "target": target,
        "reviewed_final_labels": reviewed.get("final_dataset_labels"),
        "fullauto_final_labels": fullauto.get("final_labels"),
        "source_images": fullauto.get("source_images")
        or reviewed.get("source_pool_eval", {}).get("source_images"),
        "reviewed_summary_path": str((TOOL_DIR / "research_runs" / target / "11_reports" / "summary.json").resolve()),
        "fullauto_summary_path": str(
            (TOOL_DIR / "fullauto" / "runs" / target / "reports" / "full_auto_summary.json").resolve()
        ),
    }


def build_jobs(args: argparse.Namespace) -> list[EvalJob]:
    targets = ["blueberry", "strawberry"] if args.target == "both" else [args.target]
    model_names: list[str]
    if args.model_set == "both":
        model_names = ["reviewed", "fullauto"]
    elif args.model_set in ("reviewed", "fullauto"):
        model_names = [args.model_set]
    elif args.model_set == "custom":
        model_names = []
    else:
        raise ValueError(args.model_set)

    jobs: list[EvalJob] = []
    for target in targets:
        holdout = resolve_tool_path(args.holdout_dataset) if args.holdout_dataset else default_holdout(target)
        for model_name in model_names:
            jobs.append(
                EvalJob(
                    target=target,
                    model_name=model_name,
                    model_path=default_model_path(target, model_name),
                    holdout_dataset=holdout,
                    class_id=TARGET_CLASS_ID[target],
                )
            )

    if args.custom_model:
        if args.target == "both":
            raise SystemExit("--custom-model requires --target blueberry or --target strawberry")
        custom_name = args.custom_name or Path(args.custom_model).stem
        jobs.append(
            EvalJob(
                target=args.target,
                model_name=custom_name,
                model_path=resolve_tool_path(args.custom_model),
                holdout_dataset=resolve_tool_path(args.holdout_dataset)
                if args.holdout_dataset
                else default_holdout(args.target),
                class_id=TARGET_CLASS_ID[args.target],
            )
        )

    if not jobs:
        raise SystemExit("No evaluation jobs were selected.")
    return jobs


def evaluate_job(
    job: EvalJob,
    conf: float,
    imgsz: int,
    device: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not job.model_path.exists():
        raise FileNotFoundError(f"Model not found: {job.model_path}")
    if not job.holdout_dataset.exists():
        raise FileNotFoundError(f"Holdout dataset not found: {job.holdout_dataset}")

    model = YOLO(str(job.model_path))
    rows: list[dict[str, Any]] = []
    image_paths = holdout_images(job.holdout_dataset)
    missing_gt = 0
    misses = 0
    wrong_class_only = 0

    for img_path in image_paths:
        img = read_image(img_path)
        img_h, img_w = img.shape[:2]
        gt_boxes = [
            box
            for box in load_label_from_dataset(img_path.stem, job.holdout_dataset, img_w, img_h)
            if box[0] == job.class_id
        ]
        if not gt_boxes:
            missing_gt += 1
            continue

        gt = gt_boxes[0]
        gt_xy = tuple(float(v) for v in gt[1:])
        gt_cx = (gt_xy[0] + gt_xy[2]) / 2
        gt_cy = (gt_xy[1] + gt_xy[3]) / 2
        gt_w = gt_xy[2] - gt_xy[0]
        gt_h = gt_xy[3] - gt_xy[1]
        gt_diag = math.hypot(gt_w, gt_h)
        img_diag = math.hypot(img_w, img_h)

        predict_kwargs: dict[str, Any] = {"conf": conf, "imgsz": imgsz, "verbose": False}
        if device:
            predict_kwargs["device"] = device
        result = model.predict(img, **predict_kwargs)[0]

        same_class: list[tuple[float, tuple[float, float, float, float]]] = []
        any_boxes = 0
        if result.boxes is not None:
            for pred in result.boxes:
                any_boxes += 1
                pred_cls = int(pred.cls[0])
                if pred_cls != job.class_id:
                    continue
                xyxy = tuple(float(x) for x in pred.xyxy[0].tolist())
                same_class.append((float(pred.conf[0]), xyxy))

        if not same_class:
            misses += 1
            if any_boxes:
                wrong_class_only += 1
            rows.append(
                {
                    "target": job.target,
                    "model": job.model_name,
                    "stem": img_path.stem,
                    "predicted": False,
                    "wrong_class_only": bool(any_boxes),
                    "conf": None,
                    "iou": None,
                    "center_error_px": None,
                    "center_error_x_px": None,
                    "center_error_y_px": None,
                    "center_error_abs_x_px": None,
                    "center_error_abs_y_px": None,
                    "center_error_norm_gt_diag": None,
                    "center_error_norm_image_diag": None,
                    "pred_center_in_gt_box": False,
                    "gt_w": gt_w,
                    "gt_h": gt_h,
                    "image_w": img_w,
                    "image_h": img_h,
                    "image_path": str(img_path.resolve()),
                }
            )
            continue

        score, pred_xy = max(same_class, key=lambda item: item[0])
        px1, py1, px2, py2 = pred_xy
        pred_cx = (px1 + px2) / 2
        pred_cy = (py1 + py2) / 2
        dx = pred_cx - gt_cx
        dy = pred_cy - gt_cy
        center_error = math.hypot(dx, dy)

        rows.append(
            {
                "target": job.target,
                "model": job.model_name,
                "stem": img_path.stem,
                "predicted": True,
                "wrong_class_only": False,
                "conf": score,
                "iou": iou_xyxy(gt_xy, pred_xy),
                "center_error_px": center_error,
                "center_error_x_px": dx,
                "center_error_y_px": dy,
                "center_error_abs_x_px": abs(dx),
                "center_error_abs_y_px": abs(dy),
                "center_error_norm_gt_diag": center_error / gt_diag if gt_diag else None,
                "center_error_norm_image_diag": center_error / img_diag if img_diag else None,
                "pred_center_in_gt_box": gt_xy[0] <= pred_cx <= gt_xy[2] and gt_xy[1] <= pred_cy <= gt_xy[3],
                "gt_w": gt_w,
                "gt_h": gt_h,
                "image_w": img_w,
                "image_h": img_h,
                "gt_cx": gt_cx,
                "gt_cy": gt_cy,
                "pred_cx": pred_cx,
                "pred_cy": pred_cy,
                "image_path": str(img_path.resolve()),
            }
        )

    predicted_rows = [row for row in rows if row["predicted"]]
    center_errors = [row["center_error_px"] for row in predicted_rows]
    norm_gt = [row["center_error_norm_gt_diag"] for row in predicted_rows]
    norm_img = [row["center_error_norm_image_diag"] for row in predicted_rows]
    ious = [row["iou"] for row in predicted_rows]
    confs = [row["conf"] for row in predicted_rows]
    abs_x = [row["center_error_abs_x_px"] for row in predicted_rows]
    abs_y = [row["center_error_abs_y_px"] for row in predicted_rows]

    summary = {
        "target": job.target,
        "model": job.model_name,
        "model_path": str(job.model_path.resolve()),
        "holdout_dataset": str(job.holdout_dataset.resolve()),
        "conf_threshold": conf,
        "imgsz": imgsz,
        "images": len(image_paths),
        "gt_images": len(rows),
        "missing_gt": missing_gt,
        "predicted": len(predicted_rows),
        "misses": misses,
        "wrong_class_only": wrong_class_only,
        "detect_rate": len(predicted_rows) / len(rows) if rows else None,
        "mean_conf": mean(confs),
        "mean_iou": mean(ious),
        "median_iou": median(ious),
        "iou50_count": sum(1 for v in ious if v >= 0.5),
        "iou75_count": sum(1 for v in ious if v >= 0.75),
        "mean_center_error_px": mean(center_errors),
        "median_center_error_px": median(center_errors),
        "p90_center_error_px": percentile(center_errors, 0.90),
        "max_center_error_px": max(center_errors) if center_errors else None,
        "mean_abs_x_error_px": mean(abs_x),
        "mean_abs_y_error_px": mean(abs_y),
        "mean_center_error_norm_gt_diag": mean(norm_gt),
        "median_center_error_norm_gt_diag": median(norm_gt),
        "p90_center_error_norm_gt_diag": percentile(norm_gt, 0.90),
        "mean_center_error_norm_image_diag": mean(norm_img),
        "center_inside_gt_box_count": sum(1 for row in predicted_rows if row["pred_center_in_gt_box"]),
        "within_10pct_gt_diag_count": sum(1 for v in norm_gt if v <= 0.10),
        "within_20pct_gt_diag_count": sum(1 for v in norm_gt if v <= 0.20),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    growth: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Center Re-inference Report")
    lines.append("")
    lines.append(
        "This report re-runs YOLO detection on manually labeled holdout images "
        "and compares predicted object centers with manual label centers."
    )
    lines.append("")
    lines.append("## Dataset Growth")
    lines.append("")
    lines.append("| target | reviewed labels | fullauto labels | source images | fullauto coverage |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in growth:
        source = item.get("source_images") or 0
        fullauto_labels = item.get("fullauto_final_labels") or 0
        coverage = (fullauto_labels / source * 100) if source else 0
        lines.append(
            f"| {item['target']} | {item.get('reviewed_final_labels', '')} | "
            f"{item.get('fullauto_final_labels', '')} | {source} | {coverage:.1f}% |"
        )

    lines.append("")
    lines.append("## Holdout Center Metrics")
    lines.append("")
    lines.append(
        "| target | model | detect | mean center px | median px | p90 px | "
        "mean x px | mean y px | mean norm/GT diag | center in GT box | mean IoU | IoU>=0.5 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        detect = f"{summary['predicted']}/{summary['gt_images']}"
        inside = f"{summary['center_inside_gt_box_count']}/{summary['predicted']}" if summary["predicted"] else "0/0"
        iou50 = f"{summary['iou50_count']}/{summary['predicted']}" if summary["predicted"] else "0/0"
        lines.append(
            f"| {summary['target']} | {summary['model']} | {detect} | "
            f"{fmt(summary['mean_center_error_px'])} | {fmt(summary['median_center_error_px'])} | "
            f"{fmt(summary['p90_center_error_px'])} | {fmt(summary['mean_abs_x_error_px'])} | "
            f"{fmt(summary['mean_abs_y_error_px'])} | "
            f"{fmt(summary['mean_center_error_norm_gt_diag'], 3)} | {inside} | "
            f"{fmt(summary['mean_iou'], 3)} | {iou50} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- IoU can look low when predicted boxes are wider than tight manual boxes.")
    lines.append("- For robot pickup, center error and coordinate conversion error are often more relevant.")
    lines.append("- Pixel error should be converted to paper/world units in the deployment camera pose.")
    lines.append("")
    lines.append("## Model Paths")
    lines.append("")
    for summary in summaries:
        lines.append(f"- `{summary['target']}/{summary['model']}`: `{summary['model_path']}`")

    worst = sorted(
        [row for row in rows if row["predicted"]],
        key=lambda row: row["center_error_px"],
        reverse=True,
    )[:10]
    if worst:
        lines.append("")
        lines.append("## Largest Center Errors")
        lines.append("")
        lines.append("| target | model | image | center px | x px | y px | conf | IoU |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for row in worst:
            lines.append(
                f"| {row['target']} | {row['model']} | {row['stem']} | "
                f"{fmt(row['center_error_px'])} | {fmt(row['center_error_x_px'])} | "
                f"{fmt(row['center_error_y_px'])} | {fmt(row['conf'], 3)} | {fmt(row['iou'], 3)} |"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO center-point re-inference report")
    parser.add_argument(
        "--target",
        choices=["blueberry", "strawberry", "both"],
        default="both",
        help="Target to evaluate. Default: both",
    )
    parser.add_argument(
        "--model-set",
        choices=["reviewed", "fullauto", "both", "custom"],
        default="both",
        help="Default model set to evaluate. Default: both",
    )
    parser.add_argument("--custom-model", help="Optional custom model path for a single target.")
    parser.add_argument("--custom-name", help="Name used in the report for --custom-model.")
    parser.add_argument(
        "--holdout-dataset",
        help="Optional holdout dataset override. Use only with a single target.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", help="Optional Ultralytics device value, for example 0 or cpu.")
    parser.add_argument(
        "--out-dir",
        default="fullauto/comparison_reports",
        help="Output directory relative to labeling_tools, unless absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.holdout_dataset and args.target == "both":
        raise SystemExit("--holdout-dataset can only be used with one target.")

    jobs = build_jobs(args)
    out_dir = resolve_tool_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for job in jobs:
        print(f"[eval] target={job.target} model={job.model_name}")
        print(f"       model  : {job.model_path}")
        print(f"       holdout: {job.holdout_dataset}")
        summary, job_rows = evaluate_job(job, args.conf, args.imgsz, args.device)
        summaries.append(summary)
        rows.extend(job_rows)

    targets = sorted({job.target for job in jobs})
    growth = [load_growth_summary(target) for target in targets]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"center_reinfer_report_{stamp}.json"
    csv_path = out_dir / f"center_reinfer_rows_{stamp}.csv"
    md_path = out_dir / f"center_reinfer_report_{stamp}.md"

    json_path.write_text(
        json.dumps({"summaries": summaries, "rows": rows, "growth": growth}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(csv_path, rows)
    write_markdown(md_path, summaries, rows, growth)

    print("\nSaved reports:")
    print(f"  {md_path}")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print("\nSummary:")
    for summary in summaries:
        print(
            f"  {summary['target']}/{summary['model']}: "
            f"detect={summary['predicted']}/{summary['gt_images']}, "
            f"mean_center={fmt(summary['mean_center_error_px'])} px, "
            f"median={fmt(summary['median_center_error_px'])} px, "
            f"mean_iou={fmt(summary['mean_iou'], 3)}"
        )


if __name__ == "__main__":
    main()
