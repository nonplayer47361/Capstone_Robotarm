from __future__ import annotations

import argparse
from pathlib import Path

from active_learning_core import IMAGE_EXTS, all_images, read_image, write_dataset_yaml
from dataset_utils import existing_label_path, save_labeled


def load_from_list(list_path: Path, source_root: Path | None) -> list[Path]:
    paths: list[Path] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        item = raw.strip().strip('"')
        if not item or item.startswith("#"):
            continue
        path = Path(item)
        if not path.is_absolute():
            if source_root is None:
                raise SystemExit(f"[ERROR] Relative list entry needs --source-root: {item}")
            path = source_root / path
        paths.append(path.resolve())
    return paths


def candidate_images(args: argparse.Namespace) -> list[Path]:
    if args.list:
        return load_from_list(Path(args.list).resolve(), Path(args.source_root).resolve() if args.source_root else None)
    if not args.images:
        raise SystemExit("[ERROR] Pass --images or --list")
    images_dir = Path(args.images).resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"[ERROR] Image folder not found: {images_dir}")
    return all_images(images_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add confirmed no-object images to a YOLO dataset as empty label files.")
    parser.add_argument("--images", default="", help="Folder containing confirmed no-object images")
    parser.add_argument("--list", default="", help="Text file containing confirmed no-object image paths or names")
    parser.add_argument("--source-root", default="", help="Base folder for relative entries in --list")
    parser.add_argument("--dataset-dir", required=True, help="YOLO dataset folder to update")
    parser.add_argument("--class-names", default="pill_cap", help="Comma-separated class names for dataset.yaml")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels with empty labels")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    if not class_names:
        raise SystemExit("[ERROR] class_names is empty")

    added = 0
    skipped = 0
    missing = 0
    for image_path in candidate_images(args):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if not image_path.exists():
            print(f"[MISSING] {image_path}")
            missing += 1
            continue
        if existing_label_path(image_path.stem, dataset_dir, args.val_ratio) and not args.overwrite:
            print(f"[SKIP] existing label: {image_path.name}")
            skipped += 1
            continue
        img = read_image(image_path)
        if img is None:
            print(f"[SKIP] unreadable image: {image_path}")
            skipped += 1
            continue
        if not args.dry_run:
            save_labeled(img, [], image_path.stem, dataset_dir, args.val_ratio)
        print(f"[EMPTY] {image_path.name}")
        added += 1

    if not args.dry_run:
        write_dataset_yaml(dataset_dir, class_names)
    print(f"\nDone. added_empty={added}, skipped={skipped}, missing={missing}, dataset={dataset_dir}")


if __name__ == "__main__":
    main()
