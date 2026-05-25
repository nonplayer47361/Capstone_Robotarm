from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from active_learning_core import IMAGE_EXTS, all_images


def read_image(path: Path) -> np.ndarray | None:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def resized_gray(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def detect_with_model(model_path: str, image_path: Path, conf: float) -> int | None:
    from ultralytics import YOLO

    if not hasattr(detect_with_model, "_model"):
        detect_with_model._model = YOLO(model_path)  # type: ignore[attr-defined]
    model = detect_with_model._model  # type: ignore[attr-defined]
    result = model.predict(str(image_path), conf=conf, verbose=False)[0]
    return len(result.boxes) if result.boxes is not None else 0


def scan(args: argparse.Namespace) -> tuple[list[dict[str, str]], Counter]:
    images_dir = Path(args.images).resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"[ERROR] Image folder not found: {images_dir}")

    rows: list[dict[str, str]] = []
    summary: Counter = Counter()
    seen_hashes: list[tuple[int, str]] = []

    for image_path in all_images(images_dir):
        flags: list[str] = []
        duplicate_of = ""
        img = read_image(image_path)
        if img is None:
            rows.append({
                "file": image_path.name,
                "width": "0",
                "height": "0",
                "brightness": "",
                "contrast": "",
                "blur_laplacian_var": "",
                "detections": "",
                "duplicate_of": "",
                "flags": "unreadable",
            })
            summary["unreadable"] += 1
            continue

        h, w = img.shape[:2]
        gray = resized_gray(img)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        image_hash = dhash(gray)

        if blur < args.blur_threshold:
            flags.append("blurry")
        if brightness < args.dark_threshold:
            flags.append("dark")
        if brightness > args.bright_threshold:
            flags.append("bright")
        if contrast < args.low_contrast_threshold:
            flags.append("low_contrast")

        for old_hash, old_name in seen_hashes:
            if hamming(image_hash, old_hash) <= args.duplicate_distance:
                duplicate_of = old_name
                flags.append("duplicate_candidate")
                break
        if not duplicate_of:
            seen_hashes.append((image_hash, image_path.name))

        detections = ""
        if args.model:
            count = detect_with_model(args.model, image_path, args.conf)
            detections = str(count)
            if count == 0:
                flags.append("no_detection")

        for flag in flags:
            summary[flag] += 1
        if not flags:
            summary["ok"] += 1

        rows.append({
            "file": image_path.name,
            "width": str(w),
            "height": str(h),
            "brightness": f"{brightness:.2f}",
            "contrast": f"{contrast:.2f}",
            "blur_laplacian_var": f"{blur:.2f}",
            "detections": detections,
            "duplicate_of": duplicate_of,
            "flags": ";".join(flags) if flags else "ok",
        })

    return rows, summary


def write_outputs(args: argparse.Namespace, rows: list[dict[str, str]], summary: Counter) -> None:
    images_dir = Path(args.images).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "photo_quality_report.csv"
    fields = ["file", "width", "height", "brightness", "contrast", "blur_laplacian_var", "detections", "duplicate_of", "flags"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "photo_quality_report.md"
    lines = [
        "# Photo Quality Report",
        "",
        f"- source: `{images_dir}`",
        f"- total images: `{len(rows)}`",
        f"- blur threshold: `{args.blur_threshold}`",
        f"- model check: `{args.model or 'not used'}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in sorted(summary.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Notes",
        "",
        "- `blurry`, `dark`, `bright`, `low_contrast`, and `duplicate_candidate` are automatic review hints, not final decisions.",
        "- `no_detection` is available only when `--model` is provided.",
        "- Put confirmed no-object images in `confirmed_negative` before adding empty labels.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    confirmed_negative = out_dir / "confirmed_negative"
    confirmed_negative.mkdir(exist_ok=True)

    if not args.no_copy:
        review_dir = out_dir / "review"
        review_dir.mkdir(exist_ok=True)
        for row in rows:
            flags = [flag for flag in row["flags"].split(";") if flag and flag != "ok"]
            for flag in flags:
                dst_dir = review_dir / flag
                dst_dir.mkdir(exist_ok=True)
                src = images_dir / row["file"]
                if src.exists():
                    shutil.copy2(src, dst_dir / src.name)

    print(f"[quality] report: {csv_path}")
    print(f"[quality] summary: {md_path}")
    print(f"[quality] confirmed-negative folder: {confirmed_negative}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen source images for obvious quality issues before YOLO labeling.")
    parser.add_argument("--images", required=True, help="Source image folder")
    parser.add_argument("--output-dir", required=True, help="Output folder for reports and review copies")
    parser.add_argument("--blur-threshold", type=float, default=80.0)
    parser.add_argument("--dark-threshold", type=float, default=45.0)
    parser.add_argument("--bright-threshold", type=float, default=225.0)
    parser.add_argument("--low-contrast-threshold", type=float, default=18.0)
    parser.add_argument("--duplicate-distance", type=int, default=3)
    parser.add_argument("--model", default="", help="Optional YOLO model path for no-detection screening")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for --model screening")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy flagged images into review folders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = scan(args)
    write_outputs(args, rows, summary)


if __name__ == "__main__":
    main()
