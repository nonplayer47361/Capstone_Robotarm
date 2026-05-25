"""
prepare_openset.py — 오픈소스 원형 객체 데이터셋 백업 준비

Roboflow Universe 또는 Google Open Images V7(OIV7) 공개 데이터셋을 다운로드하거나,
수동으로 내려받은 ZIP을 읽어 기존 파이프라인 형태로 변환.
여러 데이터셋을 한 번에 지정하면 자동으로 병합.

모든 원본 클래스 → "cap" (ID=0) 단일 클래스로 통합.
NEGATIVE_CLASSES 에 해당하는 클래스(뚜껑 없음 등)는 제거.

── 추천 프리셋 ────────────────────────────────────────────────
  [Roboflow]
  bottle_cap      병뚜껑 ~639장   workspace: nesne-zs4j1
  coin            동전  ~2,815장  workspace: tutorial-tpn0b
  pills           알약   ~822장   workspace: pills

  [Google Open Images V7]
  oiv7_bottle_cap 병뚜껑 최대2,500장 · Google 직접 어노테이션
  oiv7_coin       동전   최대2,500장 · Google 직접 어노테이션

── 소스 비교 실험 ───────────────────────────────────────────────
  --compare 플래그 사용 시 RF + OIV7 각각 별도 폴더에 준비.
  이후 train_openset.py 로 두 모델을 따로 학습해 탐지 성능 비교.

── Roboflow API 키 발급 ────────────────────────────────────────
  https://app.roboflow.com → Settings → API Keys → 무료 발급

── OIV7 의존성 ─────────────────────────────────────────────────
  pip install fiftyone
  (최초 실행 시 OIV7 인덱스 다운로드 — 수 분 소요)

── 사용 예 ────────────────────────────────────────────────────
  # Roboflow 단일
  python prepare_openset.py --api-key <KEY> --preset bottle_cap

  # OIV7 단일
  python prepare_openset.py --preset oiv7_bottle_cap

  # RF vs OIV7 비교 준비 (별도 폴더)
  python prepare_openset.py --api-key <KEY> --preset bottle_cap --compare
  python prepare_openset.py --preset oiv7_bottle_cap --compare

  # 수동 ZIP
  python prepare_openset.py --zip C:/Downloads/bottle-cap.zip

  # 학습까지 이어서
  python prepare_openset.py --api-key <KEY> --preset bottle_cap --train

  # 프리셋 목록 확인
  python prepare_openset.py --list-presets
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from active_learning_core import train_model, write_dataset_yaml  # noqa: E402
from dataset_utils import ensure_yolo_dirs                         # noqa: E402

# ── 알려진 데이터셋 프리셋 ────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    # ── Roboflow ──────────────────────────────────────────────────────
    "bottle_cap": {
        "source":           "roboflow",
        "workspace":        "nesne-zs4j1",
        "project":          "bottle-cap-iuzcs-tkx9q",
        "version":          1,
        "fmt":              "yolov8",
        "negative_classes": {"No Cap"},
        "description":      "RF  병뚜껑 ~639장   · 5클래스 → cap 통합 (No Cap 제거)",
    },
    "coin": {
        "source":           "roboflow",
        "workspace":        "tutorial-tpn0b",
        "project":          "coin-detection-qajzi",
        "version":          1,
        "fmt":              "yolov5pytorch",
        "negative_classes": set(),
        "description":      "RF  동전  ~2,815장 · 원형 객체 다양한 각도",
    },
    "pills": {
        "source":           "roboflow",
        "workspace":        "pills",
        "project":          "pills-data-v1",
        "version":          1,
        "fmt":              "yolov8",
        "negative_classes": set(),
        "description":      "RF  알약  ~822장   · 둥글고 작은 원형 객체",
    },
    # ── Google Open Images V7 ────────────────────────────────────────
    "oiv7_bottle_cap": {
        "source":           "oiv7",
        "classes":          ["Bottle cap"],
        "max_train":        2000,
        "max_val":          500,
        "negative_classes": set(),
        "description":      "OIV7 병뚜껑 최대2,500장 · Google 직접 어노테이션",
    },
    "oiv7_coin": {
        "source":           "oiv7",
        "classes":          ["Coin"],
        "max_train":        2000,
        "max_val":          500,
        "negative_classes": set(),
        "description":      "OIV7 동전  최대2,500장 · Google 직접 어노테이션",
    },
}

# RF/OIV7 비교 실험 쌍 정의 (--compare 모드에서 사용)
COMPARE_PAIRS: dict[str, tuple[str, str]] = {
    "bottle_cap": ("bottle_cap",      "oiv7_bottle_cap"),
    "coin":       ("coin",            "oiv7_coin"),
}

IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TARGET = "cap"

# Roboflow 폴더명 → 출력 split 이름 매핑
_SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "val"}


# ══════════════════════════════════════════════════════════════════
# 라벨 변환
# ══════════════════════════════════════════════════════════════════

def _load_class_names(rf_dir: Path) -> list[str]:
    """data.yaml / dataset.yaml 에서 클래스 이름 목록 읽기."""
    try:
        import yaml
    except ImportError:
        return []

    for fname in ("data.yaml", "dataset.yaml"):
        yp = rf_dir / fname
        if not yp.exists():
            continue
        try:
            data = yaml.safe_load(yp.read_text(encoding="utf-8"))
            names = data.get("names", {})
            if isinstance(names, dict):
                return [names[i] for i in sorted(names.keys())]
            if isinstance(names, list):
                return list(names)
        except Exception:
            pass
    return []


def _remap_label(
    src: Path,
    dst: Path,
    class_names: list[str],
    negative_classes: set[str],
) -> int:
    """YOLO 라벨 파일을 읽어 클래스 리맵 후 저장. 저장된 어노테이션 수 반환."""
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return 0

    out_lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue

        cls_name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        if cls_name in negative_classes:
            continue                          # 네거티브 클래스 → 제거
        out_lines.append("0 " + " ".join(parts[1:]))  # 모두 ID=0 으로 통합

    dst.write_text("\n".join(out_lines), encoding="utf-8")
    return len(out_lines)


# ══════════════════════════════════════════════════════════════════
# 데이터셋 처리
# ══════════════════════════════════════════════════════════════════

def _process_rf_dir(
    rf_dir: Path,
    out_dir: Path,
    negative_classes: set[str],
    name: str,
) -> dict:
    """Roboflow/OIV7 다운로드 폴더 → out_dir 에 병합."""
    class_names = _load_class_names(rf_dir)
    if not class_names:
        print(f"  [WARN] {name}: data.yaml 클래스 불명 → 원본 ID를 그대로 cap으로 처리")
        class_names = [DEFAULT_TARGET]

    print(f"  원본 클래스: {class_names}")
    neg_found = {c for c in negative_classes if c in class_names}
    if neg_found:
        print(f"  제외 클래스: {neg_found}")

    total_img = total_ann = 0
    for rf_split, out_split in _SPLIT_MAP.items():
        src_img = rf_dir / rf_split / "images"
        src_lbl = rf_dir / rf_split / "labels"
        if not src_img.exists():
            continue

        dst_img = out_dir / out_split / "images"
        dst_lbl = out_dir / out_split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n_img = n_ann = 0
        for img_p in sorted(src_img.iterdir()):
            if img_p.suffix.lower() not in IMAGE_EXTS:
                continue
            dst = dst_img / img_p.name
            if not dst.exists():
                shutil.copy2(img_p, dst)
            lbl_src = src_lbl / f"{img_p.stem}.txt"
            lbl_dst = dst_lbl / f"{img_p.stem}.txt"
            n = _remap_label(lbl_src, lbl_dst, class_names, negative_classes)
            n_img += 1
            n_ann += n

        print(f"  [{out_split}] {n_img} images  {n_ann} annotations")
        total_img += n_img
        total_ann += n_ann

    return {"images": total_img, "annotations": total_ann}


def _extract_zip(zip_path: Path, tmp_dir: Path) -> Path:
    """ZIP 압축 해제 후 train/ 또는 valid/ 가 있는 루트 경로 반환."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    # train/ 또는 valid/ 폴더가 있는 디렉터리 탐색 (최대 2단계)
    for candidate in [tmp_dir, *tmp_dir.iterdir()]:
        if isinstance(candidate, Path) and candidate.is_dir():
            if (candidate / "train").exists() or (candidate / "valid").exists():
                return candidate
    return tmp_dir


def _download_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    fmt: str,
    tmp_dir: Path,
) -> Path:
    """Roboflow API로 데이터셋 다운로드. 실제 폴더 경로 반환."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "[ERROR] roboflow 패키지가 없습니다.\n"
            "  pip install roboflow"
        )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    rf      = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download(
        fmt, location=str(tmp_dir)
    )

    loc = getattr(dataset, "location", None)
    if loc and Path(loc).exists():
        return Path(loc)

    for candidate in [tmp_dir, *sorted(tmp_dir.iterdir())]:
        if isinstance(candidate, Path) and candidate.is_dir():
            if (candidate / "train").exists() or (candidate / "valid").exists():
                return candidate
    return tmp_dir


def _download_oiv7(
    oiv7_classes: list[str],
    tmp_dir: Path,
    max_train: int = 2000,
    max_val: int = 500,
) -> Path:
    """
    fiftyone으로 OIV7 특정 클래스를 다운로드하고
    _process_rf_dir 가 읽을 수 있는 폴더 구조(train/valid + data.yaml)로 저장.

    bounding_box 포맷 변환:
      fiftyone: [x_tl, y_tl, w, h] normalized (top-left 기준)
      YOLO:     [cx, cy, w, h]     normalized (center 기준)
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        raise SystemExit(
            "[ERROR] fiftyone 패키지가 없습니다.\n"
            "  pip install fiftyone\n"
            "  (최초 실행 시 OIV7 인덱스 다운로드로 수 분 소요)"
        )

    cls_to_idx = {c: i for i, c in enumerate(oiv7_classes)}

    for out_split, fo_split, max_s in [
        ("train", "train",      max_train),
        ("valid", "validation", max_val),
    ]:
        dst_img = tmp_dir / out_split / "images"
        dst_lbl = tmp_dir / out_split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        # 기존 캐시 데이터셋 정리 (재실행 시 충돌 방지)
        ds_name = f"_openset_oiv7_{fo_split}_{'_'.join(oiv7_classes)}"
        if fo.dataset_exists(ds_name):
            fo.delete_dataset(ds_name)

        print(f"  [OIV7] {fo_split} split 다운로드 중 (최대 {max_s}장) ...")
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=fo_split,
            label_types=["detections"],
            classes=oiv7_classes,
            max_samples=max_s,
            dataset_name=ds_name,
        )

        n_img = n_ann = 0
        for sample in ds:
            src = Path(sample.filepath)
            dst = dst_img / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

            lines: list[str] = []
            detections = (
                sample.ground_truth.detections
                if sample.ground_truth and sample.ground_truth.detections
                else []
            )
            for det in detections:
                if det.label not in cls_to_idx:
                    continue
                cid = cls_to_idx[det.label]
                x, y, w, h = det.bounding_box        # [x_tl, y_tl, w, h] normalized
                cx = max(0.0, min(1.0, x + w / 2))   # → center_x
                cy = max(0.0, min(1.0, y + h / 2))   # → center_y
                w  = max(0.001, min(1.0, w))
                h  = max(0.001, min(1.0, h))
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                n_ann += 1

            (dst_lbl / f"{src.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_img += 1

        print(f"  [{out_split}] {n_img} images  {n_ann} annotations")
        fo.delete_dataset(ds_name)   # fiftyone 내부 DB 정리

    # _process_rf_dir 가 _load_class_names 로 읽을 data.yaml 작성
    yaml_lines = ["names:"]
    yaml_lines += [f"  - {c}" for c in oiv7_classes]
    yaml_lines.append(f"nc: {len(oiv7_classes)}")
    (tmp_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return tmp_dir


# ══════════════════════════════════════════════════════════════════
# 통계
# ══════════════════════════════════════════════════════════════════

def _dataset_stats(out_dir: Path) -> dict:
    result = {}
    for split in ("train", "val"):
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        n_img = sum(1 for p in img_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS) if img_dir.exists() else 0
        n_lbl = sum(1 for p in lbl_dir.glob("*.txt")
                    if p.stat().st_size > 0) if lbl_dir.exists() else 0
        result[split] = {"images": n_img, "labeled": n_lbl}
    return result


def _print_stats(out_dir: Path, yaml_path: Path) -> int:
    """통계 출력 후 labeled 총 수 반환."""
    stats     = _dataset_stats(out_dir)
    total_img = sum(v["images"]  for v in stats.values())
    total_lbl = sum(v["labeled"] for v in stats.values())

    print(f"\n{'='*60}")
    print(f"[완료] 출력: {out_dir}")
    print(f"  train : {stats['train']['images']:4d} images  /  {stats['train']['labeled']:4d} labeled")
    print(f"  val   : {stats['val']['images']:4d} images  /  {stats['val']['labeled']:4d} labeled")
    print(f"  합계  : {total_img:4d} images  /  {total_lbl:4d} labeled")
    print(f"  YAML  : {yaml_path}")
    return total_lbl


# ══════════════════════════════════════════════════════════════════
# 단일 출력 디렉터리에 데이터셋 준비
# ══════════════════════════════════════════════════════════════════

def prepare_dataset(
    tasks: list[tuple[str, Path | None, dict]],
    out_dir: Path,
    api_key: str,
    target_class: str,
) -> Path:
    """tasks 목록을 처리해 out_dir 에 YOLO 데이터셋 구성. YAML 경로 반환."""
    ensure_yolo_dirs(out_dir)
    tmp_root = out_dir / "_tmp"

    for name, zip_path, info in tasks:
        print(f"\n{'='*60}")
        print(f"[{name}]  {info.get('description', '')}")

        neg = info.get("negative_classes", set())
        source = info.get("source", "roboflow")

        if zip_path is not None:
            print(f"  압축 해제 중: {zip_path}")
            rf_dir = _extract_zip(zip_path, tmp_root / name)

        elif source == "oiv7":
            rf_dir = _download_oiv7(
                oiv7_classes = info["classes"],
                tmp_dir      = tmp_root / name,
                max_train    = info.get("max_train", 2000),
                max_val      = info.get("max_val",   500),
            )

        else:  # roboflow
            if not api_key:
                print("  [ERROR] Roboflow API 키가 없습니다.")
                print("          --api-key 옵션 또는 PREPARE_OPENSET.bat 실행 시 입력하세요.")
                print("          발급: https://app.roboflow.com → Settings → API Keys")
                continue
            print(f"  다운로드 중: {info['workspace']}/{info['project']} v{info['version']}")
            try:
                rf_dir = _download_roboflow(
                    api_key   = api_key,
                    workspace = info["workspace"],
                    project   = info["project"],
                    version   = info["version"],
                    fmt       = info.get("fmt", "yolov8"),
                    tmp_dir   = tmp_root / name,
                )
            except Exception as e:
                print(f"  [ERROR] 다운로드 실패: {e}")
                continue

        _process_rf_dir(rf_dir, out_dir, neg, name)

    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    return write_dataset_yaml(out_dir, [target_class])


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="오픈소스 원형 객체 데이터셋 준비 (pill_cap 백업용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--api-key",   default="",
                   help="Roboflow API 키 (app.roboflow.com > Settings > API)")
    p.add_argument("--preset",    action="append", default=[],
                   choices=list(PRESETS.keys()),
                   help="다운로드 프리셋 (여러 번 사용 시 병합)")
    p.add_argument("--workspace", default="",   help="커스텀 Roboflow workspace 슬러그")
    p.add_argument("--project",   default="",   help="커스텀 Roboflow project 슬러그")
    p.add_argument("--version",   type=int, default=1)
    p.add_argument("--fmt",       default="yolov8",
                   help="Roboflow 다운로드 포맷 (기본: yolov8)")
    p.add_argument("--zip",       action="append", default=[], metavar="PATH",
                   help="수동 다운로드 ZIP 경로 (여러 번 사용 시 병합)")
    p.add_argument("--negative-classes", nargs="*", default=None,
                   help="제외할 클래스 이름 목록 (기본: 프리셋 자동)")
    p.add_argument("--target-class", default=DEFAULT_TARGET,
                   help=f"출력 클래스 이름 (기본: {DEFAULT_TARGET})")
    p.add_argument("--out-dir",   default=str(_HERE / "openset_dataset"),
                   help="출력 데이터셋 경로 (기본: labeling_tools/openset_dataset)")
    p.add_argument("--compare",   action="store_true",
                   help=(
                       "RF vs OIV7 비교 실험용 별도 폴더에 준비. "
                       "출력: openset_compare/<preset>_rf/ 와 openset_compare/<preset>_oiv7/"
                   ))
    p.add_argument("--train",     action="store_true",
                   help="데이터 준비 완료 후 바로 학습 실행")
    p.add_argument("--base-model", default="yolo11n.pt",
                   help="--train 시 사용할 베이스 모델 (기본: yolo11n.pt)")
    p.add_argument("--epochs",    type=int, default=80)
    p.add_argument("--list-presets", action="store_true",
                   help="프리셋 목록 출력 후 종료")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_presets:
        print("\n── 사용 가능한 프리셋 ──────────────────────────────────────")
        rf_presets  = {k: v for k, v in PRESETS.items() if v.get("source") != "oiv7"}
        oiv7_presets = {k: v for k, v in PRESETS.items() if v.get("source") == "oiv7"}
        print("  [Roboflow]")
        for key, info in rf_presets.items():
            print(f"    {key:<20} {info['description']}")
        print("  [Google Open Images V7]")
        for key, info in oiv7_presets.items():
            print(f"    {key:<20} {info['description']}")
        print()
        return

    # ── 작업 목록 구성 ────────────────────────────────────────────────
    tasks: list[tuple[str, Path | None, dict]] = []

    for key in args.preset:
        tasks.append((key, None, PRESETS[key]))

    if args.workspace and args.project:
        neg = set(args.negative_classes) if args.negative_classes is not None else set()
        tasks.append((f"{args.workspace}/{args.project}", None, {
            "source":           "roboflow",
            "workspace":        args.workspace,
            "project":          args.project,
            "version":          args.version,
            "fmt":              args.fmt,
            "negative_classes": neg,
            "description":      "커스텀",
        }))

    for zip_str in args.zip:
        zp = Path(zip_str).resolve()
        if not zp.exists():
            print(f"[ERROR] ZIP 파일 없음: {zp}")
            continue
        neg = set(args.negative_classes) if args.negative_classes is not None else set()
        tasks.append((zp.stem, zp, {"source": "zip", "negative_classes": neg, "description": "ZIP"}))

    if not tasks:
        print("[ERROR] 처리할 데이터셋이 없습니다.")
        print("  --preset bottle_cap  또는  --zip <파일>  을 지정하세요.")
        print("  --list-presets 로 프리셋 목록 확인")
        sys.exit(1)

    # ── 비교 실험 모드 ────────────────────────────────────────────────
    if args.compare:
        _run_compare_mode(tasks, args)
        return

    # ── 일반 모드 ─────────────────────────────────────────────────────
    out_dir   = Path(args.out_dir)
    yaml_path = prepare_dataset(tasks, out_dir, args.api_key, args.target_class)
    total_lbl = _print_stats(out_dir, yaml_path)

    if total_lbl == 0:
        print("\n[WARN] 라벨이 0개입니다. 클래스 이름이나 ZIP 구조를 확인하세요.")
        return

    if args.train:
        _run_training(out_dir, args.base_model, args.epochs, args.target_class, "cap_openset")
    else:
        print(f"\n학습하려면:")
        print(f"  python train_openset.py")
        print(f"  또는: PREPARE_OPENSET.bat 에서 학습 옵션 선택")


def _run_compare_mode(
    tasks: list[tuple[str, Path | None, dict]],
    args: argparse.Namespace,
) -> None:
    """RF vs OIV7 각각 별도 폴더에 준비하고 비교 정보 출력."""
    compare_root = _HERE / "openset_compare"

    rf_tasks   = [(n, z, i) for n, z, i in tasks if i.get("source") != "oiv7"]
    oiv7_tasks = [(n, z, i) for n, z, i in tasks if i.get("source") == "oiv7"]

    trained: list[tuple[str, str, str]] = []  # (label, out_dir, best_pt)

    for source_label, src_tasks in [("rf", rf_tasks), ("oiv7", oiv7_tasks)]:
        if not src_tasks:
            continue
        tag     = "_".join(n for n, _, _ in src_tasks)
        out_dir = compare_root / f"{tag}_{source_label}"
        print(f"\n{'#'*60}")
        print(f"# 비교 준비: {source_label.upper()}  →  {out_dir.name}")
        print(f"{'#'*60}")

        yaml_path = prepare_dataset(src_tasks, out_dir, args.api_key, args.target_class)
        total_lbl = _print_stats(out_dir, yaml_path)

        if total_lbl == 0:
            print(f"  [WARN] 라벨 0개 — {source_label} 건너뜀")
            continue

        if args.train:
            best = _run_training(
                out_dir,
                args.base_model,
                args.epochs,
                args.target_class,
                f"{tag}_{source_label}_compare",
            )
            trained.append((source_label.upper(), str(out_dir), best))

    if trained:
        print(f"\n{'='*60}")
        print("[비교 실험 학습 완료]")
        for label, ddir, best in trained:
            print(f"  {label:<6}  dataset={Path(ddir).name:<30}  model={best}")
        print()
        print("  다음 단계 — a4_plane_research.py 에서 각 모델로 탐지 성능 비교")


def _run_training(
    out_dir: Path,
    base_model: str,
    epochs: int,
    target_class: str,
    run_name: str,
) -> str:
    project = _HERE / "openset_runs"
    print(f"\n학습 시작 (base={base_model}, epochs={epochs}, name={run_name}) ...")
    best = train_model(
        dataset_dir = out_dir,
        base_model  = base_model,
        project     = project,
        name        = run_name,
        epochs      = epochs,
        class_names = [target_class],
    )
    print(f"\n[학습 완료] best.pt → {best}")
    return best


if __name__ == "__main__":
    main()
