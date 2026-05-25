"""
merge_datasets.py  --  YOLO 데이터셋 병합 도구

여러 팀원의 라벨링 결과를 하나의 YOLO 표준 4폴더 구조로 병합합니다.
파일명 MD5 해시 기반 결정론적 분할이므로 같은 이미지는 항상 같은 split에
배정됩니다 → 팀원 병합 시 train/val 오염 없음

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
지원 입력 구조 (어느 형식이든 자동 인식):
  ① YOLO 4폴더: {dir}/train/images/  +  {dir}/train/labels/
  ② YOLO 2폴더: {dir}/images/        +  {dir}/labels/
  ③ 평면 혼재:  {dir}/*.jpg          +  {dir}/*.txt

출력은 항상 YOLO 표준 4폴더 구조로 저장됩니다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용법:
  # 두 팀원 결과 병합
  python merge_datasets.py \\
      --inputs alice/dataset bob/dataset \\
      --output merged_dataset

  # 내 데이터셋에 팀원 결과 추가 (--output = 내 기존 데이터셋)
  python merge_datasets.py \\
      --inputs teammate_dataset \\
      --output my_dataset

  # 세 명 이상 병합
  python merge_datasets.py \\
      --inputs alice/dataset bob/dataset charlie/dataset \\
      --output merged

  # 결과 미리 보기 (복사 없음)
  python merge_datasets.py --inputs A B --output C --dry-run

충돌(같은 이미지, 라벨 내용이 다른 경우) 처리 옵션:
  --conflict keep-first   : 첫 번째 입력 우선 (기본값)
  --conflict keep-last    : 마지막 입력 우선
  --conflict most-boxes   : 박스 수가 많은 쪽 우선
  --conflict interactive  : OpenCV 화면에서 직접 선택
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── dataset_utils 임포트 ──────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset_utils import (
    create_yaml,
    ensure_yolo_dirs,
    get_split,
    print_stats,
    save_labeled,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 충돌 표시용 색상 (BGR) — 소스마다 다른 색
_CONFLICT_COLORS = [
    (0,   220,  60),   # 녹색
    (0,   100, 255),   # 주황
    (220,   0,   0),   # 파랑
    (180,   0, 220),   # 보라
    (0,   200, 200),   # 시안
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 이미지 / 라벨 I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def read_image(path: Path) -> np.ndarray | None:
    """한글 경로 대응 이미지 로드."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def read_label_lines(lbl_path: Path | None) -> list[str]:
    """라벨 txt → 줄 목록 (빈 줄 제거)."""
    if not lbl_path or not lbl_path.exists():
        return []
    return [ln for ln in lbl_path.read_text(encoding="utf-8").strip().splitlines()
            if ln.strip()]


def normalize_labels(lines: list[str]) -> list[str]:
    """비교용 정규화: 각 줄의 숫자를 반올림해서 문자열로."""
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            out.append(line.strip())
            continue
        try:
            cls  = int(parts[0])
            vals = [round(float(v), 4) for v in parts[1:]]
            out.append(f"{cls} " + " ".join(f"{v:.4f}" for v in vals))
        except ValueError:
            out.append(line.strip())
    return sorted(out)


def count_boxes(lbl_path: Path | None) -> int:
    return len(read_label_lines(lbl_path))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 입력 폴더 스캔 (3가지 구조 자동 인식)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scan_dataset(base_dir: Path) -> dict[str, tuple[Path, Path | None]]:
    """
    입력 폴더에서 {stem: (img_path, lbl_path|None)} 반환.

    지원 구조:
      ① YOLO 4폴더: train/images/ + train/labels/ (val 포함)
      ② YOLO 2폴더: images/ + labels/
      ③ 평면 혼재:  *.jpg + *.txt 같은 폴더
    """
    result: dict[str, tuple[Path, Path | None]] = {}

    # ① 4폴더 구조
    for split in ("train", "val"):
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"
        if img_dir.exists():
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTS:
                    lbl = lbl_dir / f"{img_path.stem}.txt"
                    result[img_path.stem] = (img_path, lbl if lbl.exists() else None)
    if result:
        return result

    # ② 2폴더 구조
    img_dir = base_dir / "images"
    lbl_dir = base_dir / "labels"
    if img_dir.exists():
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTS:
                lbl = lbl_dir / f"{img_path.stem}.txt"
                result[img_path.stem] = (img_path, lbl if lbl.exists() else None)
        return result

    # ③ 평면 구조
    for img_path in sorted(base_dir.iterdir()):
        if img_path.suffix.lower() in IMAGE_EXTS:
            lbl = base_dir / f"{img_path.stem}.txt"
            result[img_path.stem] = (img_path, lbl if lbl.exists() else None)
    return result


def detect_structure(base_dir: Path) -> str:
    """폴더 구조 타입 반환: '4folder' | '2folder' | 'flat' | 'empty'"""
    if any((base_dir / s / "images").exists() for s in ("train", "val")):
        return "4folder"
    if (base_dir / "images").exists():
        return "2folder"
    if base_dir.exists() and any(p.suffix.lower() in IMAGE_EXTS
                                  for p in base_dir.iterdir()):
        return "flat"
    return "empty"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# dataset.yaml에서 클래스 이름 추출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_class_names(source_dirs: list[Path]) -> list[str] | None:
    """입력 폴더의 dataset.yaml에서 클래스 이름 추출. 없으면 None."""
    for src in source_dirs:
        yaml_path = src / "dataset.yaml"
        if not yaml_path.exists():
            continue
        text  = yaml_path.read_text(encoding="utf-8")
        names: list[str] = []
        in_names = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                in_names = True
                continue
            if in_names:
                # "  0: clay_ball" 형식
                if ":" in stripped and stripped[0].isdigit():
                    names.append(stripped.split(":", 1)[1].strip())
                # "  - clay_ball" 형식
                elif stripped.startswith("-"):
                    names.append(stripped[1:].strip())
                # 다른 최상위 키가 나오면 종료
                elif stripped and not stripped[0].isdigit() and not stripped.startswith("-"):
                    break
        if names:
            return names
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 대화형 충돌 해결 (OpenCV)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_boxes(display: np.ndarray, lbl_path: Path | None,
                color: tuple, label: str) -> None:
    """이미지에 라벨 파일의 bbox를 그린다."""
    h, w = display.shape[:2]
    for line in read_label_lines(lbl_path):
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cx, cy, bw, bh = [float(v) for v in parts[1:]]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1 + 2, max(y1 - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        except ValueError:
            continue


def interactive_resolve(
    stem:       str,
    candidates: list[tuple[Path, Path | None, str]],   # (img, lbl, source_name)
) -> int:
    """
    OpenCV 창으로 모든 후보의 bbox를 동시에 표시하고
    사용자가 1~N 키로 선택한 인덱스를 반환.
    선택 없이 q/Esc → 0(첫 번째)
    """
    img = read_image(candidates[0][0])
    if img is None:
        return 0

    h, w = img.shape[:2]

    while True:
        display = img.copy()

        # 각 소스의 bbox를 다른 색상으로 표시
        for i, (_, lbl_path, src_name) in enumerate(candidates):
            color = _CONFLICT_COLORS[i % len(_CONFLICT_COLORS)]
            tag   = f"[{i+1}] {src_name} ({count_boxes(lbl_path)}box)"
            _draw_boxes(display, lbl_path, color, tag)

        # HUD
        cv2.rectangle(display, (0, 0), (w, 68), (0, 0, 0), -1)
        cv2.putText(display,
                    f"충돌: {stem}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        legend = "  ".join(
            f"[{i+1}]{c[2]}"
            for i, c in enumerate(candidates)
        )
        cv2.putText(display,
                    f"{legend}    |  1~{len(candidates)} = 선택  s/Enter = 첫번째  q = 건너뜀(첫번째)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1)

        # 색상 범례 (우측 상단)
        for i, (_, _, src_name) in enumerate(candidates):
            color = _CONFLICT_COLORS[i % len(_CONFLICT_COLORS)]
            cv2.putText(display,
                        f"[{i+1}] {src_name}",
                        (w - 200, 80 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("충돌 해결 (Conflict Resolver)", display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27, ord('s'), 13):    # q / Esc / s / Enter
            return 0
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(candidates):
                return idx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 병합 핵심 로직
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def merge(
    input_dirs:  list[Path],
    output_dir:  Path,
    val_ratio:   float        = 0.2,
    conflict:    str          = "keep-first",
    dry_run:     bool         = False,
    class_names: list[str] | None = None,
) -> None:
    """
    여러 입력 폴더를 output_dir 로 병합.

    출력 폴더가 이미 데이터를 가지고 있으면 가장 높은 우선순위(첫 번째)로
    자동 포함됩니다 → "팀원 결과를 내 데이터셋에 추가" 사용법 지원.
    """
    # ── 소스 목록 구성 ───────────────────────────────────────────────
    # output이 이미 존재하면 첫 번째 소스로 포함 (highest priority)
    all_sources: list[Path] = []
    if output_dir.exists() and detect_structure(output_dir) != "empty":
        all_sources.append(output_dir)
    all_sources.extend(p for p in input_dirs if p.resolve() != output_dir.resolve())

    if not all_sources:
        print("[오류] 병합할 소스가 없습니다.")
        return

    # ── 스캔 ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  입력 소스 스캔")
    print(f"{'─'*60}")
    combined: dict[str, list[tuple[Path, Path | None, str]]] = defaultdict(list)
    for src in all_sources:
        entries    = scan_dataset(src)
        struct     = detect_structure(src)
        src_name   = src.name
        priority   = "(기준 데이터셋)" if src.resolve() == output_dir.resolve() else ""
        print(f"  {src_name:<30s} {len(entries):4d}장  [{struct}] {priority}")
        for stem, (img, lbl) in entries.items():
            combined[stem].append((img, lbl, src_name))
    print(f"  {'─'*56}")
    print(f"  합산 고유 이미지: {len(combined):4d}장\n")

    # ── 클래스 정보 ──────────────────────────────────────────────────
    if class_names is None:
        class_names = find_class_names(all_sources)
    if class_names:
        print(f"  클래스: {class_names}")
    else:
        print("  [주의] dataset.yaml 없음 → yaml 미생성. --classes 로 지정 가능.")

    # ── 병합 루프 ────────────────────────────────────────────────────
    if not dry_run:
        ensure_yolo_dirs(output_dir)

    n_total      = len(combined)
    n_ok         = 0
    n_skip_img   = 0
    n_dup        = 0       # 동일 라벨 중복 (실제 충돌 아님)
    n_conflict   = 0       # 실제 충돌 발생 수
    conflict_log: list[dict] = []

    if conflict == "interactive":
        cv2.namedWindow("충돌 해결 (Conflict Resolver)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("충돌 해결 (Conflict Resolver)", 960, 720)

    for stem, candidates in sorted(combined.items()):
        chosen_idx = 0  # 기본: 첫 번째

        if len(candidates) > 1:
            # 라벨 내용 비교 (정규화 후)
            norm_labels = [normalize_labels(read_label_lines(c[1])) for c in candidates]
            all_same    = all(nl == norm_labels[0] for nl in norm_labels[1:])

            if all_same:
                n_dup += 1      # 내용 동일 → 그냥 첫 번째 사용
            else:
                n_conflict += 1

                # 충돌 해결 전략 적용
                if conflict == "keep-first":
                    chosen_idx = 0
                elif conflict == "keep-last":
                    chosen_idx = len(candidates) - 1
                elif conflict == "most-boxes":
                    chosen_idx = max(range(len(candidates)),
                                     key=lambda i: count_boxes(candidates[i][1]))
                elif conflict == "interactive":
                    chosen_idx = interactive_resolve(stem, candidates)
                else:
                    chosen_idx = 0

                srcs     = [c[2] for c in candidates]
                box_cnts = [count_boxes(c[1]) for c in candidates]
                conflict_log.append({
                    "stem":    stem,
                    "sources": srcs,
                    "boxes":   box_cnts,
                    "chosen":  candidates[chosen_idx][2],
                    "strategy": conflict,
                })

        img_path, lbl_path, src_name = candidates[chosen_idx]

        if dry_run:
            split = get_split(stem, val_ratio)
            src_tag = f"(소스: {src_name})"
            norm_dry = [normalize_labels(read_label_lines(c[1])) for c in candidates]
            conf_tag = " ★충돌" if len(candidates) > 1 and not all(nl == norm_dry[0] for nl in norm_dry[1:]) else ""
            print(f"  [DRY] {stem:<40s} → {split}/{src_tag}{conf_tag}")
            n_ok += 1
            continue

        img = read_image(img_path)
        if img is None:
            print(f"  [WARN] 이미지 로드 실패 (건너뜀): {img_path}")
            n_skip_img += 1
            continue

        lines = read_label_lines(lbl_path)
        save_labeled(img, lines, stem, output_dir, val_ratio)
        n_ok += 1

    if conflict == "interactive":
        cv2.destroyAllWindows()

    # ── yaml 생성 ─────────────────────────────────────────────────────
    if not dry_run and class_names:
        yaml_p = create_yaml(output_dir, nc=len(class_names), names=class_names)
        print(f"\n  dataset.yaml 생성: {yaml_p}")

    # ── 충돌 로그 저장 ────────────────────────────────────────────────
    if conflict_log and not dry_run:
        log_path = output_dir / "_merge_conflicts.json"
        log_path.write_text(
            json.dumps(conflict_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  충돌 로그:      {log_path}")

    # ── 최종 통계 ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  병합 결과 {'[DRY RUN - 실제 저장 안 함]' if dry_run else ''}")
    print(f"{'─'*60}")
    print(f"  처리 이미지    : {n_ok:4d}장")
    print(f"  동일 라벨 중복 : {n_dup:4d}장  (자동 제거)")
    print(f"  실제 충돌      : {n_conflict:4d}장  (전략: {conflict})")
    print(f"  이미지 로드 실패: {n_skip_img:4d}장")
    if n_conflict:
        print(f"\n  충돌 상세:")
        for log in conflict_log[:10]:
            srcs = " vs ".join(
                f"{s}({b}box)" for s, b in zip(log["sources"], log["boxes"])
            )
            print(f"    {log['stem'][:35]:35s}  {srcs}  → 채택: {log['chosen']}")
        if len(conflict_log) > 10:
            print(f"    ... 외 {len(conflict_log)-10}건 (_merge_conflicts.json 참고)")

    if not dry_run:
        print_stats(output_dir)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# argparse + main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO 데이터셋 병합 도구 (4폴더 구조 자동 저장)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--inputs", nargs="+", required=True, metavar="DIR",
        help="병합할 입력 폴더 경로 (여러 개 나열 가능)",
    )
    p.add_argument(
        "--output", required=True, metavar="DIR",
        help="병합 결과를 저장할 폴더 경로. 이미 데이터가 있으면 첫 번째 소스로 포함됨.",
    )
    p.add_argument(
        "--conflict",
        choices=["keep-first", "keep-last", "most-boxes", "interactive"],
        default="keep-first",
        help=(
            "충돌(같은 이미지, 다른 라벨) 처리 방식 (기본: keep-first)\n"
            "  keep-first  : --inputs 순서에서 앞쪽 우선\n"
            "  keep-last   : --inputs 순서에서 뒤쪽 우선\n"
            "  most-boxes  : 박스 수가 많은 쪽 우선\n"
            "  interactive : OpenCV 화면에서 직접 선택"
        ),
    )
    p.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="검증셋 비율 (기본 0.2 = 20%%). 기존 데이터셋과 동일하게 맞출 것.",
    )
    p.add_argument(
        "--classes", nargs="+", default=None, metavar="NAME",
        help=(
            "클래스 이름 목록 (미지정 시 입력 폴더의 dataset.yaml 에서 자동 추출)\n"
            "예) --classes clay_ball blueberry"
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="실제 파일 복사 없이 무엇을 병합할지만 출력",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dirs = [Path(d).resolve() for d in args.inputs]
    output_dir = Path(args.output).resolve()

    # 입력 경로 존재 확인
    for d in input_dirs:
        if not d.exists():
            raise SystemExit(f"[오류] 입력 폴더 없음: {d}")

    print(f"{'━'*60}")
    print(f"  YOLO 데이터셋 병합 도구")
    print(f"{'━'*60}")
    print(f"  입력 폴더  : {', '.join(d.name for d in input_dirs)}")
    print(f"  출력 폴더  : {output_dir}")
    print(f"  충돌 전략  : {args.conflict}")
    print(f"  검증셋 비율: {args.val_ratio:.0%}")
    if args.dry_run:
        print(f"  ★ DRY RUN 모드 (실제 저장 안 함)")

    merge(
        input_dirs  = input_dirs,
        output_dir  = output_dir,
        val_ratio   = args.val_ratio,
        conflict    = args.conflict,
        dry_run     = args.dry_run,
        class_names = args.classes,
    )


if __name__ == "__main__":
    main()
