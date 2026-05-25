#!/usr/bin/env python3
"""
[LEGACY] detect_a4.py  —  Real-time YOLO detection with A4 paper coordinate output.

⚠️  이 파일은 레거시(구버전) 진입점입니다.
    새로운 연구/실험에는 a4_plane_research.py 를 사용하세요.

    레거시 유지 이유:
      - 수동 4-클릭 캘리브레이션 방식 (color_dot 없는 환경)
      - a4_calibration.json 기반 단순 좌표 변환

    대체 명령어:
      python a4_plane_research.py --live --method aruco --model MODEL.pt
      python a4_plane_research.py --validate --method aruco --model MODEL.pt

────────────────────────────────────────────────────────────
Workflow (3 steps):
  Step 1.  python detect_a4.py --sheet
           → a4_sheet.png 생성 (A4 100% 스케일로 프린트)

  Step 2.  python detect_a4.py --calibrate [--camera N]
           → 카메라 화면에서 빨간 마커 4개를 순서대로 클릭
           → a4_calibration.json 저장

  Step 3.  python detect_a4.py --detect --model MODEL.pt [--camera N]
           → 실시간 탐지 + A4 좌표 출력

오차 검증 (사전 설정 좌표 지정):
  python detect_a4.py --detect --model MODEL.pt --check 105,148.5 52.5,74

A4 좌표계: 원점(0,0) = 종이 왼쪽 상단,  단위 mm
  X: 오른쪽 방향  (0 ~ 210 mm)
  Y: 아래쪽 방향  (0 ~ 297 mm)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _configure_utf8_stdio() -> None:
    """Windows 기본 CP949 콘솔에서도 한글 help 출력이 죽지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


_configure_utf8_stdio()

import warnings
warnings.warn(
    "\n[LEGACY] detect_a4.py 는 구버전 진입점입니다.\n"
    "  신규 실험에는 a4_plane_research.py 를 사용하세요.\n"
    "  예: python a4_plane_research.py --live --method aruco --model MODEL.pt",
    DeprecationWarning,
    stacklevel=1,
)

HERE = Path(__file__).resolve().parent

# ── A4 상수 ───────────────────────────────────────────────────────────────────
A4_W_MM = 210.0
A4_H_MM = 297.0

# 캘리브레이션 마커 위치 (A4 좌표, mm) — 출력물에 프린트된 정확한 위치
# 클릭 순서: TL → TR → BL → BR
CALIB_MM: list[tuple[float, float]] = [
    ( 25.0,  25.0),   # 0: Top-Left
    (185.0,  25.0),   # 1: Top-Right
    ( 25.0, 272.0),   # 2: Bottom-Left
    (185.0, 272.0),   # 3: Bottom-Right
]
CALIB_LABELS = [
    "TL (25, 25)",
    "TR (185, 25)",
    "BL (25, 272)",
    "BR (185, 272)",
]

CALIB_FILE = HERE / "a4_calibration.json"
SHEET_FILE = HERE / "a4_sheet.png"
SHEET_DPI  = 200          # 출력 DPI (200 이상 권장)

# A4 미니맵 표시 스케일 (1mm = MINI_SCALE px)
MINI_SCALE = 2


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — A4 시트 생성
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sheet() -> None:
    """캘리브레이션/격자 A4 시트를 PNG로 저장합니다."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise SystemExit("Pillow 필요: pip install pillow")

    def mm(v: float) -> int:
        return round(v * SHEET_DPI / 25.4)

    W, H = mm(A4_W_MM), mm(A4_H_MM)
    img  = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # 폰트 로드 (없으면 기본 폰트)
    font_sm = font_md = None
    for fp in ["arial.ttf", "C:/Windows/Fonts/arial.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        try:
            from PIL import ImageFont as IF
            font_sm = IF.truetype(fp, mm(3.2))
            font_md = IF.truetype(fp, mm(4.5))
            break
        except OSError:
            continue
    if font_sm is None:
        from PIL import ImageFont as IF
        font_sm = font_md = IF.load_default()

    # 격자선 10mm (연한 회색)
    for x in range(0, int(A4_W_MM) + 1, 10):
        draw.line([(mm(x), 0), (mm(x), H)], fill="#E0E0E0", width=1)
    for y in range(0, int(A4_H_MM) + 1, 10):
        draw.line([(0, mm(y)), (W, mm(y))], fill="#E0E0E0", width=1)

    # 격자선 50mm (중간 회색) + 좌표 라벨
    for x in range(0, int(A4_W_MM) + 1, 50):
        draw.line([(mm(x), 0), (mm(x), H)], fill="#AAAAAA", width=1)
        if x > 0:
            draw.text((mm(x) + 2, 4), str(x), fill="#888888", font=font_sm)
    for y in range(0, int(A4_H_MM) + 1, 50):
        draw.line([(0, mm(y)), (W, mm(y))], fill="#AAAAAA", width=1)
        if y > 0:
            draw.text((3, mm(y) + 2), str(y), fill="#888888", font=font_sm)

    # 테두리
    draw.rectangle([(0, 0), (W - 1, H - 1)], outline="black", width=2)

    # 캘리브레이션 마커 (빨간 원 + 십자)
    r  = mm(4)
    cl = mm(7)
    for i, (xm, ym) in enumerate(CALIB_MM):
        cx, cy = mm(xm), mm(ym)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="#CC0000", width=2)
        draw.line([(cx - cl, cy), (cx + cl, cy)], fill="#CC0000", width=2)
        draw.line([(cx, cy - cl), (cx, cy + cl)], fill="#CC0000", width=2)
        # 라벨 위치 (코너별)
        lx = cx + r + 2  if xm < 105 else cx - r - mm(32)
        ly = cy + r + 2  if ym < 148 else cy - r - mm(5)
        draw.text((lx, ly), CALIB_LABELS[i], fill="#CC0000", font=font_sm)

    # 중앙 마커 (파란 십자)
    cx, cy = mm(105), mm(148.5)
    s = mm(5)
    draw.line([(cx - s, cy), (cx + s, cy)], fill="#0055CC", width=2)
    draw.line([(cx, cy - s), (cx, cy + s)], fill="#0055CC", width=2)
    draw.text((cx + s + 2, cy - mm(4)), "CENTER  (105, 148.5)", fill="#0055CC", font=font_sm)

    # 검증 테스트 포인트 (초록 십자) — 물체를 여기 올려놓고 오차 확인
    TEST_PTS = [( 52.5,  74.0), (157.5,  74.0),
                ( 52.5, 223.0), (157.5, 223.0)]
    ts = mm(3.5)
    for xm, ym in TEST_PTS:
        cx2, cy2 = mm(xm), mm(ym)
        draw.line([(cx2 - ts, cy2), (cx2 + ts, cy2)], fill="#007700", width=2)
        draw.line([(cx2, cy2 - ts), (cx2, cy2 + ts)], fill="#007700", width=2)
        draw.text((cx2 + ts + 1, cy2 - mm(4)), f"({xm:.0f},{ym:.0f})", fill="#007700", font=font_sm)

    # 제목
    draw.text((mm(30), mm(2)),
              "A4 Detection Calibration Grid  —  210 × 297 mm",
              fill="black", font=font_md)

    # 하단 지시사항
    draw.text((mm(5), mm(291)),
              "● 클릭 순서: ①TL → ②TR → ③BL → ④BR (빨간 마커)   "
              "● 초록 십자: 검증용 위치 (52,74) (157,74) (52,223) (157,223)   "
              "● 인쇄: 100% 크기, 맞춤/비율조정 OFF",
              fill="#333333", font=font_sm)

    img.save(str(SHEET_FILE), dpi=(SHEET_DPI, SHEET_DPI))
    print(f"[sheet] 저장: {SHEET_FILE}")
    print(f"[sheet] 크기: {W}×{H}px  ({SHEET_DPI}DPI = A4 210×297mm)")
    print(f"[sheet] 프린트: '실제 크기(100%)' 또는 '배율 없음' 으로 출력하세요")
    print(f"[sheet] 캘리브레이션 마커(빨간): {[f'({x},{y})mm' for x,y in CALIB_MM]}")
    print(f"[sheet] 검증 포인트(초록):  (52.5,74)  (157.5,74)  (52.5,223)  (157.5,223)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 카메라 캘리브레이션
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate(camera_id: int) -> None:
    """카메라 캡처 화면에서 4개 빨간 마커를 클릭해 호모그래피를 계산합니다."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise SystemExit(f"[calibrate] 카메라 {camera_id}를 열 수 없습니다")

    clicked: list[list[float]] = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append([float(x), float(y)])
            mm = CALIB_MM[len(clicked) - 1]
            print(f"  [{len(clicked)}/4] 픽셀 ({x:4d},{y:4d})  →  A4 {mm} mm")

    win = "캘리브레이션 — 빨간 마커 4개를 순서대로 클릭"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    print("\n[calibrate] 프린트된 A4 시트를 카메라 아래 평평하게 놓으세요.")
    print("[calibrate] 빨간 마커를 다음 순서로 클릭하세요:")
    for i, (label, mm_pos) in enumerate(zip(CALIB_LABELS, CALIB_MM)):
        print(f"   {label}  =  {mm_pos} mm")
    print("[calibrate]  r = 초기화  |  Enter = 저장  |  q = 취소\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()
        n = len(clicked)

        # 이미 클릭된 점 표시
        for i, (px, py) in enumerate(clicked):
            cv2.drawMarker(vis, (int(px), int(py)), (0, 0, 220),
                           cv2.MARKER_CROSS, 26, 2)
            cv2.putText(vis, CALIB_LABELS[i], (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1)

        # 안내 텍스트
        if n < 4:
            msg = f"  {n+1}/4 클릭: {CALIB_LABELS[n]}"
            color = (30, 220, 30)
        else:
            msg = "  4/4 완료 — Enter 저장 / r 재시도"
            color = (0, 220, 220)
        cv2.putText(vis, msg, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            clicked.clear()
            print("[calibrate] 초기화 — TL부터 다시 클릭하세요")
        elif key == 13 and n == 4:   # Enter
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("[calibrate] 취소됨")
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(clicked) != 4:
        print("[calibrate] 포인트가 부족합니다.")
        return

    src = np.array(clicked,  dtype=np.float32)
    dst = np.array(CALIB_MM, dtype=np.float32)
    H, mask = cv2.findHomography(src, dst)

    # 캘리브레이션 품질 확인 (각 마커의 재투영 오차)
    print("\n[calibrate] 재투영 오차 확인:")
    for i, (px, py) in enumerate(clicked):
        pred_mm = cv2.perspectiveTransform(
            np.array([[[px, py]]], dtype=np.float32), H)[0][0]
        err = np.hypot(pred_mm[0] - CALIB_MM[i][0], pred_mm[1] - CALIB_MM[i][1])
        print(f"   {CALIB_LABELS[i]}  →  예측 ({pred_mm[0]:.2f},{pred_mm[1]:.2f})mm  오차={err:.2f}mm")

    data = {
        "H": H.tolist(),
        "calib_points_px": clicked,
        "calib_points_mm": [list(p) for p in CALIB_MM],
    }
    CALIB_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n[calibrate] 저장: {CALIB_FILE}")
    print("[calibrate] 다음 단계: python detect_a4.py --detect --model YOUR_MODEL.pt")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — 실시간 탐지 + A4 좌표
# ═══════════════════════════════════════════════════════════════════════════════

def _px_to_a4(px_x: float, px_y: float, H: np.ndarray) -> tuple[float, float]:
    """픽셀 좌표 → A4 mm 좌표 변환."""
    pt  = np.array([[[px_x, px_y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0][0][0]), float(out[0][0][1])


def _make_minimap(
    a4_x: float | None,
    a4_y: float | None,
    check_pts: list[tuple[float, float]],
    target_h: int,
) -> np.ndarray:
    """A4 미니맵 이미지 생성 (실시간 표시용)."""
    mw = int(A4_W_MM * MINI_SCALE)
    mh = int(A4_H_MM * MINI_SCALE)
    mini = np.full((mh, mw, 3), 245, dtype=np.uint8)

    # 50mm 격자
    for xm in range(0, int(A4_W_MM) + 1, 50):
        cv2.line(mini, (xm * MINI_SCALE, 0), (xm * MINI_SCALE, mh), (185, 185, 185), 1)
        cv2.putText(mini, str(xm), (xm * MINI_SCALE + 1, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 140), 1)
    for ym in range(0, int(A4_H_MM) + 1, 50):
        cv2.line(mini, (0, ym * MINI_SCALE), (mw, ym * MINI_SCALE), (185, 185, 185), 1)
        cv2.putText(mini, str(ym), (2, ym * MINI_SCALE + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 140), 1)
    cv2.rectangle(mini, (0, 0), (mw - 1, mh - 1), (0, 0, 0), 2)

    # 캘리브레이션 마커 (회색 사각형)
    for xm, ym in CALIB_MM:
        cv2.drawMarker(mini, (int(xm * MINI_SCALE), int(ym * MINI_SCALE)),
                       (150, 150, 150), cv2.MARKER_SQUARE, 7, 1)

    # 사전 설정 포인트 (주황 십자)
    for cx, cy in check_pts:
        mx, my = int(cx * MINI_SCALE), int(cy * MINI_SCALE)
        cv2.drawMarker(mini, (mx, my), (0, 100, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(mini, f"({cx:.0f},{cy:.0f})", (mx + 5, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 80, 200), 1)

    # 탐지된 위치 (빨간 원)
    if a4_x is not None and a4_y is not None:
        mx = int(np.clip(a4_x, 0, A4_W_MM) * MINI_SCALE)
        my = int(np.clip(a4_y, 0, A4_H_MM) * MINI_SCALE)
        cv2.circle(mini, (mx, my), 7, (0, 0, 210), -1)
        cv2.circle(mini, (mx, my), 8, (255, 255, 255), 1)
        cv2.putText(mini, f"({a4_x:.1f},{a4_y:.1f})", (mx + 9, my + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 180), 1)

    # 카메라 프레임 높이에 맞게 스케일
    if target_h > 0 and mh != target_h:
        scale = target_h / mh
        mini = cv2.resize(mini, (int(mw * scale), target_h),
                          interpolation=cv2.INTER_LINEAR)
    return mini


def detect(
    model_path: str,
    camera_id: int,
    check_pts: list[tuple[float, float]],
    conf_thresh: float = 0.30,
) -> None:
    """실시간 탐지 + A4 좌표 출력."""
    if not CALIB_FILE.exists():
        raise SystemExit(
            f"[detect] 캘리브레이션 파일 없음: {CALIB_FILE}\n"
            "먼저 실행하세요: python detect_a4.py --calibrate"
        )
    H = np.array(
        json.loads(CALIB_FILE.read_text(encoding="utf-8"))["H"],
        dtype=np.float64,
    )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics 필요: pip install ultralytics")

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise SystemExit(f"[detect] 카메라 {camera_id}를 열 수 없습니다")

    print(f"\n[detect] 모델  : {model_path}")
    print(f"[detect] 카메라: {camera_id}   신뢰도 임계값: {conf_thresh}")
    if check_pts:
        print(f"[detect] 검증 포인트(mm): {check_pts}")
    print("[detect] 키: q = 종료 | s = 스냅샷 저장\n")

    snap_idx = 0
    WIN = "A4 실시간 탐지  [q=종료  s=스냅샷]"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_thresh, verbose=False)[0]

        a4_x: float | None = None
        a4_y: float | None = None
        vis = frame.copy()

        # 최고 신뢰도 박스 1개만 처리
        boxes = sorted(results.boxes, key=lambda b: float(b.conf[0]), reverse=True)
        if boxes:
            box = boxes[0]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0
            conf  = float(box.conf[0])

            a4_x, a4_y = _px_to_a4(cx_px, cy_px, H)

            # 바운딩 박스
            cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 220, 30), 2)
            # 중심 십자
            cv2.drawMarker(vis, (int(cx_px), int(cy_px)), (0, 0, 230),
                           cv2.MARKER_CROSS, 24, 2)
            # 좌표 라벨
            coord_txt = f"A4: ({a4_x:.1f}, {a4_y:.1f}) mm    conf={conf:.2f}"
            cv2.putText(vis, coord_txt, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30, 230, 30), 2)

            # 사전 설정 포인트와 오차
            if check_pts:
                dists = [(np.hypot(a4_x - ex, a4_y - ey), (ex, ey))
                         for ex, ey in check_pts]
                err, near = min(dists)
                err_txt = f"가장 가까운 설정점: ({near[0]:.0f},{near[1]:.0f})mm   오차={err:.1f}mm"
                cv2.putText(vis, err_txt, (10, vis.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 210, 255), 2)
        else:
            cv2.putText(vis, "탐지 없음", (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

        # 카메라 화면 + A4 미니맵 합성
        mini     = _make_minimap(a4_x, a4_y, check_pts, vis.shape[0])
        combined = np.hstack([vis, mini])

        cv2.imshow(WIN, combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = HERE / f"snapshot_{snap_idx:04d}.jpg"
            cv2.imwrite(str(fname), combined)
            print(f"[snapshot] {fname}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_check_pts(raw: list[str]) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for s in raw:
        parts = s.split(",")
        if len(parts) != 2:
            raise SystemExit(f"잘못된 포인트 형식 '{s}'. 사용법: X,Y  예) 105,148.5")
        pts.append((float(parts[0]), float(parts[1])))
    return pts


def main() -> None:
    p = argparse.ArgumentParser(
        description="A4 기반 실시간 YOLO 탐지 + 좌표 출력",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python detect_a4.py --sheet
  python detect_a4.py --calibrate --camera 0
  python detect_a4.py --detect --model best.pt
  python detect_a4.py --detect --model best.pt --check 105,148.5 52.5,74
        """,
    )
    p.add_argument("--sheet",     action="store_true",
                   help="A4 캘리브레이션 시트 생성 (a4_sheet.png)")
    p.add_argument("--calibrate", action="store_true",
                   help="카메라 캘리브레이션 (빨간 마커 4개 클릭)")
    p.add_argument("--detect",    action="store_true",
                   help="실시간 탐지 (--model 필수)")
    p.add_argument("--model",     default="",
                   help="YOLO 모델 경로 (.pt 파일)")
    p.add_argument("--camera",    type=int, default=0,
                   help="카메라 장치 ID (기본값 0)")
    p.add_argument("--conf",      type=float, default=0.30,
                   help="탐지 신뢰도 임계값 (기본값 0.30)")
    p.add_argument("--check",     nargs="*", default=[], metavar="X,Y",
                   help="검증용 사전 설정 좌표(mm)  예: --check 105,148.5 52.5,74")
    args = p.parse_args()

    if args.sheet:
        generate_sheet()

    if args.calibrate:
        calibrate(args.camera)

    if args.detect:
        if not args.model:
            p.error("--detect 사용 시 --model MODEL.pt 필수")
        check_pts = _parse_check_pts(args.check or [])
        detect(args.model, args.camera, check_pts, args.conf)

    if not any([args.sheet, args.calibrate, args.detect]):
        p.print_help()


if __name__ == "__main__":
    main()
