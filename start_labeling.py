"""
start_labeling.py -- Team-friendly launcher for berry YOLO labeling.

This wrapper asks which target the teammate is labeling, then starts the
appropriate existing labeler while keeping one shared class map:

  0: blueberry
  1: strawberry

Examples:
  python start_labeling.py
  python start_labeling.py --images raw/team_a --target strawberry
  python start_labeling.py --images raw/team_b --target blueberry --dataset-dir dataset
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset_utils import create_yaml, ensure_yolo_dirs, print_stats  # noqa: E402
from label_clay_ball import Labeler  # noqa: E402
from tk_labeler import run_tk_annotator  # noqa: E402

CLASSES = ["blueberry", "strawberry"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEIC_EXTS = {".heic", ".heif"}
TARGETS = {
    "1": "blueberry",
    "2": "strawberry",
    "blueberry": "blueberry",
    "strawberry": "strawberry",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose blueberry/strawberry labeling mode and start the right tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--images",
        default="",
        help="Folder containing source images. If omitted, the launcher asks.",
    )
    parser.add_argument(
        "--target",
        choices=["blueberry", "strawberry"],
        default="",
        help="What this teammate will label (blueberry or strawberry).",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="YOLO dataset output folder. Default: ./dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio. Default: 0.2",
    )
    parser.add_argument("--hue-lo", type=int, default=110, help="Blueberry HSV hue low.")
    parser.add_argument("--hue-hi", type=int, default=165, help="Blueberry HSV hue high.")
    parser.add_argument("--sat-lo", type=int, default=35, help="Blueberry HSV saturation low.")
    parser.add_argument("--val-lo", type=int, default=30, help="Blueberry HSV value low.")
    parser.add_argument("--pad-ratio", type=float, default=0.08, help="Auto bbox padding ratio.")
    parser.add_argument("--min-area", type=float, default=0.001, help="Minimum detected area ratio.")
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Only label image index range, 0-based and inclusive.",
    )
    return parser.parse_args()


def ask_target() -> str:
    while True:
        print()
        print("Select labeling target")
        print("  1. blueberry  (purple round clay model, auto assist)")
        print("  2. strawberry (cone-shaped clay model, manual bbox)")
        choice = input("Target [1/2]: ").strip().lower()
        if choice in TARGETS:
            return TARGETS[choice]
        print("Please enter 1 or 2.")


def ask_images() -> Path:
    while True:
        raw = input("Image folder path: ").strip().strip('"')
        if not raw:
            print("Please enter a folder path.")
            continue
        path = Path(raw).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path
        print(f"Image folder not found: {path}")


def supported_image_paths(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def heic_paths(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in HEIC_EXTS)


def unique_output_path(out_dir: Path, filename: str) -> Path:
    """Return a non-conflicting output path inside out_dir."""
    dst = out_dir / filename
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    n = 1
    while True:
        candidate = out_dir / f"{stem}_{n:02d}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def convert_heic_folder(images_dir: Path) -> Path:
    heics = heic_paths(images_dir)
    supported = supported_image_paths(images_dir)
    if not heics:
        return images_dir

    out_dir = images_dir.with_name(images_dir.name + "_jpg")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        from pillow_heif import register_heif_opener
    except ImportError as exc:
        raise SystemExit(
            "HEIC photos were found, but HEIC support packages are not installed.\n"
            "Run RUN_RESEARCH_PIPELINE.bat or RUN_LABELING.bat again so dependencies can be checked.\n"
            f"Missing package detail: {exc}"
        )

    register_heif_opener()
    converted = 0
    copied = 0
    skipped = 0
    print(f"[INFO] HEIC images found: {len(heics)}")
    print(f"[INFO] Preparing JPG work folder: {out_dir}")
    copied_names: set[str] = set()
    for src in supported:
        dst = out_dir / src.name
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            copied_names.add(dst.name)
            skipped += 1
            continue
        try:
            shutil.copy2(src, dst)
            copied_names.add(dst.name)
            copied += 1
        except Exception as exc:
            print(f"[WARN] Failed to copy {src.name}: {exc}")
    for src in heics:
        dst = out_dir / f"{src.stem}.jpg"
        if dst.name in copied_names:
            dst = unique_output_path(out_dir, f"{src.stem}.jpg")
        elif dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue
        elif dst.exists():
            dst = unique_output_path(out_dir, f"{src.stem}.jpg")
        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                img.save(dst, "JPEG", quality=95)
            converted += 1
        except Exception as exc:
            print(f"[WARN] Failed to convert {src.name}: {exc}")
    print(f"[INFO] Work folder ready. copied={copied}, converted={converted}, skipped={skipped}")
    return out_dir


def prepare_images_dir(images_dir: Path) -> Path:
    if heic_paths(images_dir):
        return convert_heic_folder(images_dir)
    return images_dir


def prepare_dataset(dataset_dir: Path) -> None:
    ensure_yolo_dirs(dataset_dir)
    create_yaml(dataset_dir, nc=len(CLASSES), names=CLASSES)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1.")
    if args.range and args.range[1] < args.range[0]:
        raise SystemExit("--range END must be greater than or equal to START.")
    if args.range and args.range[0] < 0:
        raise SystemExit("--range START must be 0 or higher.")


def range_values(args: argparse.Namespace) -> tuple[int, int]:
    return (args.range[0], args.range[1]) if args.range else (0, -1)


def run_blueberry(args: argparse.Namespace, images_dir: Path, dataset_dir: Path) -> None:
    hsv = dict(
        hue_lo=args.hue_lo,
        hue_hi=args.hue_hi,
        sat_lo=args.sat_lo,
        val_lo=args.val_lo,
        pad=args.pad_ratio,
        min_area=args.min_area,
    )
    labeler = Labeler(
        images_dir=images_dir,
        dataset_dir=dataset_dir,
        class_id=0,
        class_name="blueberry",
        hsv=hsv,
        val_ratio=args.val_ratio,
        range_start=range_values(args)[0],
        range_end=range_values(args)[1],
    )
    labeler.run()


def run_strawberry(args: argparse.Namespace, images_dir: Path, dataset_dir: Path) -> None:
    run_tk_annotator(
        images_dir=images_dir,
        dataset_dir=dataset_dir,
        classes=CLASSES,
        val_ratio=args.val_ratio,
        locked_class=1,
        range_start=range_values(args)[0],
        range_end=range_values(args)[1],
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    target = args.target or ask_target()
    images_dir = Path(args.images).expanduser().resolve() if args.images else ask_images()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Image folder not found: {images_dir}")
    images_dir = prepare_images_dir(images_dir)
    if not supported_image_paths(images_dir):
        raise SystemExit(f"No supported images found in: {images_dir}")

    prepare_dataset(dataset_dir)

    print()
    print("Labeling session")
    print(f"  target  : {target}")
    print(f"  images  : {images_dir}")
    print(f"  dataset : {dataset_dir}")
    print(f"  classes : 0=blueberry, 1=strawberry")
    print()

    if target == "blueberry":
        run_blueberry(args, images_dir, dataset_dir)
    else:
        run_strawberry(args, images_dir, dataset_dir)

    create_yaml(dataset_dir, nc=len(CLASSES), names=CLASSES)
    print_stats(dataset_dir)


if __name__ == "__main__":
    main()
