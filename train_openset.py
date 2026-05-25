"""
train_openset.py — 오픈소스 데이터셋으로 YOLO 학습

prepare_openset.py 로 준비된 openset_dataset/ 을 사용해
단일 클래스(cap) YOLO 모델을 학습.

──────────────────────────────────────────────────────────────
  사용 예
──────────────────────────────────────────────────────────────
  # 기본 (yolo11n.pt 에서 처음부터)
  python train_openset.py

  # 에포크·배치 지정
  python train_openset.py --epochs 120 --batch 16

  # pill_cap 파인튜닝 (기존 학습 결과 활용)
  python train_openset.py --base-model research_runs/pill_cap/runs/04_final_model/weights/best.pt

  # GPU 지정
  python train_openset.py --device 0

  # RF vs OIV7 비교 실험 — 각각 학습
  python train_openset.py --dataset-dir openset_compare/bottle_cap_rf   --name bottle_cap_rf_compare
  python train_openset.py --dataset-dir openset_compare/bottle_cap_oiv7 --name bottle_cap_oiv7_compare
──────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from active_learning_core import latest_best, train_model  # noqa: E402

DEFAULT_DATASET = _HERE / "openset_dataset"
DEFAULT_OUTPUT  = _HERE / "openset_runs"
DEFAULT_MODEL   = "yolo11n.pt"
DEFAULT_EPOCHS  = 80
DEFAULT_BATCH   = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="오픈소스 원형 객체 데이터셋 학습 (pill_cap 백업용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET),
        help=f"학습 데이터셋 경로 (기본: {DEFAULT_DATASET})",
    )
    p.add_argument(
        "--base-model",
        default=DEFAULT_MODEL,
        help=(
            f"베이스 모델 (기본: {DEFAULT_MODEL})\n"
            "  pill_cap 파인튜닝 예: "
            "research_runs/pill_cap/runs/04_final_model/weights/best.pt"
        ),
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help=f"학습 에포크 수 (기본: {DEFAULT_EPOCHS})")
    p.add_argument("--batch",  type=int, default=DEFAULT_BATCH,
                   help=f"배치 크기 (기본: {DEFAULT_BATCH})")
    p.add_argument("--imgsz", type=int, default=640,
                   help="학습 이미지 크기 (기본: 640)")
    p.add_argument("--device", default="",
                   help='학습 장치 (기본: 자동). 예: "0" "cpu"')
    p.add_argument("--name",   default="cap_openset",
                   help="학습 실행 이름 (기본: cap_openset)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUTPUT),
                   help=f"결과 저장 폴더 (기본: {DEFAULT_OUTPUT})")
    p.add_argument("--freeze", type=int, default=10,
                   help="백본 동결 레이어 수 (기본: 10, 0이면 전체 학습)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir     = Path(args.out_dir)

    # ── 데이터셋 존재 확인 ──────────────────────────────────────────
    if not dataset_dir.exists():
        print(f"[ERROR] 데이터셋 폴더가 없습니다: {dataset_dir}")
        print()
        print("  먼저 prepare_openset.py 를 실행해 데이터셋을 준비하세요.")
        print("  예: python prepare_openset.py --api-key <KEY> --preset bottle_cap")
        print("  또는: PREPARE_OPENSET.bat 실행")
        sys.exit(1)

    train_img = dataset_dir / "train" / "images"
    if not train_img.exists() or not any(train_img.iterdir()):
        print(f"[ERROR] 학습 이미지가 없습니다: {train_img}")
        print("  prepare_openset.py 를 먼저 실행하세요.")
        sys.exit(1)

    # ── base_model 경로 정규화 ──────────────────────────────────────
    base_model = args.base_model
    bm_path = Path(base_model)
    if not bm_path.is_absolute():
        bm_path = _HERE / bm_path
    if bm_path.exists():
        base_model = str(bm_path)
        print(f"[BASE] 로컬 모델 사용: {bm_path}")
    else:
        print(f"[BASE] 모델: {base_model}")

    # ── 학습 정보 출력 ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  오픈소스 데이터셋 학습")
    print("=" * 60)
    print(f"  데이터셋 : {dataset_dir}")
    print(f"  베이스   : {base_model}")
    print(f"  에포크   : {args.epochs}")
    print(f"  배치     : {args.batch}")
    print(f"  이미지크기: {args.imgsz}")
    print(f"  장치     : {args.device or '자동'}")
    print(f"  출력     : {out_dir / args.name}")
    print()

    # ── 학습 실행 ───────────────────────────────────────────────────
    best = train_model(
        dataset_dir = dataset_dir,
        base_model  = base_model,
        project     = out_dir,
        name        = args.name,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = args.device,
        class_names = ["cap"],
        freeze      = args.freeze,
    )

    print()
    print("=" * 60)
    print(f"[학습 완료]")
    print(f"  best.pt  → {best}")
    print()
    print("  다음 단계 — A4 좌표 실험에서 이 모델 사용:")
    print(f"    python a4_detect/a4_plane_research.py --model {best}")
    print()


if __name__ == "__main__":
    main()
