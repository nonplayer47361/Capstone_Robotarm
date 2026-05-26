#!/usr/bin/env python3
"""
tools/setup_models.py  —  오픈소스 YOLO 모델 다운로드 & 클래스 검증

지원 대상
---------
  coin        : Open Images V7  (Ultralytics 자동 다운로드 — API 키 불필요)
                  --expected-class Coin
  bottle_cap  : HuggingFace Murugan098/BottleCapDetection (API 키 불필요)
                  --expected-class good_cap
  stone       : YOLO-World yoloworld_s.pt (이미 models/에 포함)
                  --model models/yoloworld_s.pt  (WORLD_CLASS_MAP 이 "rock" 자동 적용)
  *Roboflow 백업*: --rf bottle_cap / stone  (--api-key 필요)

사용법
------
  # 동전 모델 다운로드 + 클래스 확인
  python tools/setup_models.py --coin

  # 병뚜껑 모델 다운로드 (HuggingFace, API 키 불필요)
  python tools/setup_models.py --bottle-cap

  # 돌멩이: YOLO-World 이미 models/yoloworld_s.pt 에 있음 → 별도 다운로드 불필요
  #   사용 인자: --model models/yoloworld_s.pt --object-type stone
  #   (WORLD_CLASS_MAP 이 "rock" 으로 자동 변환)

  # Roboflow 백업 (병뚜껑 / 돌멩이)
  python tools/setup_models.py --rf bottle_cap --api-key YOUR_KEY
  python tools/setup_models.py --rf stone       --api-key YOUR_KEY

  # 이미 받은 .pt 파일 클래스 목록 출력
  python tools/setup_models.py --inspect models/coin_oi7n.pt

모델 저장 위치
--------------
  a4_detect/models/coin_oi7n.pt        <- Open Images V7 nano  (expected-class: Coin)
  a4_detect/models/bottle_cap_hf.pt    <- HuggingFace          (expected-class: good_cap)
  a4_detect/models/yoloworld_s.pt      <- YOLO-World (stone 포함 임의 물체 탐지)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # a4_detect/
MODELS_DIR = HERE / "models"


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise SystemExit("[ERROR] pip install ultralytics")


def _inspect_model(pt_path: Path) -> None:
    """모델의 클래스 목록 + 클래스 수를 출력한다."""
    YOLO = _require_ultralytics()
    print(f"\n[inspect] {pt_path}")
    model = YOLO(str(pt_path))
    names = model.names                      # {id: class_name}
    print(f"  클래스 수 : {len(names)}")
    for idx, name in sorted(names.items()):
        print(f"  {idx:>4d}  {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 동전 — Open Images V7 (Ultralytics yolov8n-oiv7.pt)
# ─────────────────────────────────────────────────────────────────────────────

# OI7 601-class 모델에서 Coin 이 속한 실제 클래스 이름.
# Ultralytics 배포 yolov8n-oiv7.pt 기준 "Coin" (대문자 C).
OI7_COIN_CLASS = "Coin"

def setup_coin_oi7() -> Path:
    """
    yolov8n-oiv7.pt 를 다운로드하여 models/coin_oi7n.pt 로 저장하고,
    Coin 클래스 ID 를 확인한다.

    Returns
    -------
    저장된 .pt 경로
    """
    YOLO = _require_ultralytics()
    MODELS_DIR.mkdir(exist_ok=True)

    dest = MODELS_DIR / "coin_oi7n.pt"
    if dest.exists():
        print(f"[coin] 이미 존재: {dest}  (재다운로드 생략)")
    else:
        print("[coin] yolov8n-oiv7.pt 다운로드 중 (Ultralytics 자동 다운로드) ...")
        model = YOLO("yolov8n-oiv7.pt")
        # Ultralytics 가 ~/.cache 에 저장한 파일을 models/ 로 복사
        cached = Path(model.ckpt_path)
        shutil.copy2(cached, dest)
        print(f"[coin] 저장 완료: {dest}")

    # 클래스 확인
    model = YOLO(str(dest))
    names = model.names
    coin_ids = [idx for idx, n in names.items() if n.lower() == OI7_COIN_CLASS.lower()]
    if coin_ids:
        print(f"[coin] '{OI7_COIN_CLASS}' 클래스 ID: {coin_ids[0]}  (총 {len(names)}개 클래스)")
        print(f"[coin] 사용 인자: --model models/coin_oi7n.pt --expected-class {OI7_COIN_CLASS}")
    else:
        # 혹시 대소문자가 다른 경우 fuzzy 검색
        candidates = [n for n in names.values() if "coin" in n.lower()]
        print(f"[coin] '{OI7_COIN_CLASS}' 클래스를 찾지 못했습니다.")
        print(f"[coin] 유사 후보: {candidates}")
        print("[coin] --inspect 옵션으로 전체 클래스 목록을 확인하세요.")

    return dest


# ─────────────────────────────────────────────────────────────────────────────
# 병뚜껑 / 돌멩이 — Roboflow Universe
# ─────────────────────────────────────────────────────────────────────────────

# Roboflow Universe 권장 프로젝트 (2024-05 기준 검색 결과)
# 실제 사용 전 https://universe.roboflow.com 에서 최신 버전을 확인하세요.
RF_PRESETS = {
    "bottle_cap": {
        "workspace" : "roboflow-100",
        "project"   : "bottle-caps",
        "version"   : 1,
        "class_name": "bottle_cap",      # 실제 클래스 이름 (inspect 후 확인 필요)
        "dest_name" : "bottle_cap_rf.pt",
        "note"      : (
            "탐지 클래스명은 모델마다 다릅니다. "
            "다운로드 후 --inspect 로 확인하고 --expected-class 를 맞춰주세요."
        ),
    },
    "stone": {
        "workspace" : "roboflow-100",
        "project"   : "rock-detection-msswr",
        "version"   : 1,
        "class_name": "rock",
        "dest_name" : "stone_rf.pt",
        "note"      : (
            "탐지 클래스명은 'rock' 또는 'stone' 일 수 있습니다. "
            "다운로드 후 --inspect 로 확인하고 --expected-class 를 맞춰주세요."
        ),
    },
}


def setup_rf_model(object_name: str, api_key: str) -> Path:
    """
    Roboflow Universe 에서 모델을 다운로드하여 models/<dest_name>.pt 로 저장.

    Parameters
    ----------
    object_name : 'bottle_cap' | 'stone'
    api_key     : Roboflow API 키 (https://app.roboflow.com 에서 발급)
    """
    if object_name not in RF_PRESETS:
        raise SystemExit(
            f"[ERROR] 지원 대상: {list(RF_PRESETS.keys())}  입력값: '{object_name}'"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "[ERROR] Roboflow 패키지 필요: pip install roboflow\n"
            "       설치 후 다시 실행하세요."
        )

    preset = RF_PRESETS[object_name]
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / preset["dest_name"]

    if dest.exists():
        print(f"[{object_name}] 이미 존재: {dest}  (재다운로드 생략)")
    else:
        print(f"[{object_name}] Roboflow 다운로드: "
              f"{preset['workspace']}/{preset['project']} v{preset['version']}")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(preset["workspace"]).project(preset["project"])
        version = project.version(preset["version"])
        model_dir = version.download("yolov8", location=str(MODELS_DIR / object_name))

        # weights/best.pt 를 dest 로 복사
        best = next(Path(model_dir).rglob("best.pt"), None)
        if best is None:
            raise SystemExit(f"[ERROR] best.pt 를 찾지 못했습니다: {model_dir}")
        shutil.copy2(best, dest)
        print(f"[{object_name}] 저장 완료: {dest}")

    # 클래스 확인
    YOLO = _require_ultralytics()
    model = YOLO(str(dest))
    names = model.names
    cls_guess = preset["class_name"]
    matched = [(i, n) for i, n in names.items() if cls_guess in n.lower()]

    print(f"[{object_name}] 클래스 수: {len(names)}")
    if matched:
        for idx, name in matched:
            print(f"[{object_name}] 후보 클래스 -> ID {idx}: '{name}'")
        actual_cls = matched[0][1]
        print(f"[{object_name}] 사용 인자: --model models/{preset['dest_name']} "
              f"--expected-class {actual_cls}")
    else:
        print(f"[{object_name}] '{cls_guess}' 매칭 없음. 전체 클래스:")
        for idx, name in names.items():
            print(f"  {idx:>3d}  {name}")

    if preset.get("note"):
        print(f"[{object_name}] NOTE: {preset['note']}")

    return dest


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def setup_bottle_cap_hf() -> Path:
    """HuggingFace Murugan098/BottleCapDetection 모델 다운로드."""
    from huggingface_hub import hf_hub_download
    YOLO = _require_ultralytics()
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / "bottle_cap_hf.pt"

    if dest.exists():
        print(f"[bottle_cap] 이미 존재: {dest}  (재다운로드 생략)")
    else:
        print("[bottle_cap] HuggingFace 다운로드: Murugan098/BottleCapDetection")
        cached = hf_hub_download(
            repo_id="Murugan098/BottleCapDetection",
            filename="best.pt",
            local_dir=str(MODELS_DIR / "_tmp_bc"),
        )
        shutil.copy2(cached, dest)
        tmp = MODELS_DIR / "_tmp_bc"
        if tmp.exists():
            shutil.rmtree(tmp)
        print(f"[bottle_cap] 저장 완료: {dest}  ({dest.stat().st_size/1024/1024:.1f} MB)")

    model = YOLO(str(dest))
    names = model.names
    print(f"[bottle_cap] 클래스: {names}")
    print(f"[bottle_cap] 사용 인자: --model models/bottle_cap_hf.pt --expected-class good_cap")
    print("[bottle_cap] NOTE: 산업용 품질검사 모델. 탁자/종이 위 병뚜껑도 인식하나 "
          "conf 가 낮을 수 있음 → --conf 0.20 권장")
    return dest


def main():
    p = argparse.ArgumentParser(
        description="오픈소스 YOLO 모델 다운로드 & 클래스 검증",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--coin",       action="store_true",
                   help="Open Images V7 동전 모델 다운로드 (API 키 불필요) -> models/coin_oi7n.pt")
    g.add_argument("--bottle-cap", action="store_true",
                   help="HuggingFace 병뚜껑 모델 다운로드 (API 키 불필요) -> models/bottle_cap_hf.pt")
    g.add_argument("--rf", metavar="OBJECT",
                   help="Roboflow 백업 다운로드: bottle_cap | stone  (--api-key 필요)")
    g.add_argument("--inspect", metavar="PT_PATH",
                   help="기존 .pt 파일의 클래스 목록 출력")

    p.add_argument("--api-key", default="",
                   help="Roboflow API 키 (--rf 사용 시 필수)")

    args = p.parse_args()

    if args.coin:
        setup_coin_oi7()

    elif args.bottle_cap:
        setup_bottle_cap_hf()

    elif args.rf:
        if not args.api_key:
            p.error("--rf 사용 시 --api-key 가 필요합니다.")
        setup_rf_model(args.rf, args.api_key)

    elif args.inspect:
        pt = Path(args.inspect)
        if not pt.exists():
            raise SystemExit(f"[ERROR] 파일 없음: {pt}")
        _inspect_model(pt)


if __name__ == "__main__":
    main()
