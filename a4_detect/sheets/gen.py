"""
sheets/gen.py — A4 탐지 방법별 테스트 시트 생성

생성 파일 (프린트 후 물체를 테스트 포인트에 올려놓고 오차 측정):

  단일 방식 (캘리브레이션 마커 + 테스트 포인트)
  ─────────────────────────────────────────────
  sheet_edge.png         — 외곽선 전용: 마커 없음, 테스트 포인트 5개
  sheet_aruco.png        — ArUco 코너 마커 × 4,   테스트 포인트 3개
  sheet_color_dot.png    — 색상 원 × 4,            테스트 포인트 3개
  sheet_checkerboard.png — 체커보드 패턴,           테스트 포인트 3개
  sheet_grid.png         — 20mm 격자,              테스트 포인트 3개

  복합 방식 (테스트 포인트 1개, 캘리브레이션 조합별 4종)
  ─────────────────────────────────────────────────────
  sheet_comp_A_aruco.png        — ArUco만 (기준)
  sheet_comp_B_aruco_color.png  — ArUco + 색상점
  sheet_comp_C_aruco_grid.png   — ArUco + 격자
  sheet_comp_D_full.png         — ArUco + 색상점 + 격자 (풀 복합)

실행:
  python sheets/gen.py                  # 모든 시트 → sheets/output/
  python sheets/gen.py --out ./out      # 출력 위치 지정
  python sheets/gen.py --only aruco     # 특정 시트만
  python sheets/gen.py --one-point --only edge
                                      # 한 장당 표시점 1개인 좌표 실험 시트
  python sheets/gen.py --one-point --only composite --combo comp_B_aruco_color
                                      # 필요한 복합 조합만 1점 시트로 생성

프린트 주의: 반드시 '실제 크기(100%)' / '배율 없음'으로 출력하세요.
가능하면 함께 생성되는 print_ready_a4_sheets.pdf 를 인쇄하세요.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _configure_utf8_stdio() -> None:
    """Windows 기본 CP949 콘솔에서도 한글 help 출력이 깨지지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


_configure_utf8_stdio()

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plane_coord import (
    ARUCO_CENTER_MM, ARUCO_DICT_ID,
    CB_ORIGIN_MM, CHESSBOARD_COLS, CHESSBOARD_ROWS, SQUARE_MM,
    COLOR_POSITIONS_MM,
    GRID_SPACING_MM,
    A4_W_MM, A4_H_MM,
)
from eval import EVAL_TEST_PTS, EVAL_X_MM, EVAL_Y_MM  # 30점 격자 단일 정의 (eval/__init__.py)

# ── 공통 상수 ─────────────────────────────────────────────────────────────────
DPI   = 200
_W    = round(A4_W_MM * DPI / 25.4)   # 1654 px
_H    = round(A4_H_MM * DPI / 25.4)   # 2339 px
TARGET_DIAMETER_MM = 40.0
EDGE_BORDER_INSET_MM = 6.0


def _px(v: float) -> int:
    return round(v * DPI / 25.4)


# ── 테스트 포인트 정의 ────────────────────────────────────────────────────────
# (번호, x_mm, y_mm)

# 외곽선 전용: 마커 없음 → 포인트 5개로 전체 영역 커버 + 방향 확인
EDGE_TEST_PTS: list[tuple[int, float, float]] = [
    (1,  40.0,  50.0),   # ① 좌상
    (2, 170.0,  50.0),   # ② 우상
    (3, 105.0, 148.5),   # ③ 중앙
    (4,  40.0, 247.0),   # ④ 좌하
    (5, 170.0, 247.0),   # ⑤ 우하
]

# 단일 방식 (ArUco / 색상점 / 격자): 포인트 3개, 대각선 배치
SINGLE_TEST_PTS: list[tuple[int, float, float]] = [
    (1,  52.5,  74.0),   # ① 좌상
    (2, 105.0, 148.5),   # ② 중앙
    (3, 157.5, 223.0),   # ③ 우하
]

# 체커보드 전용: 패턴 아래 영역에 포인트 3개
# (패턴은 x=[0~180], y=[35~175] 범위 점유)
CHECKER_TEST_PTS: list[tuple[int, float, float]] = [
    (1,  40.0, 210.0),   # ① 좌 (40mm 원이 패턴을 가리지 않게 이격)
    (2, 105.0, 245.0),   # ② 중앙 하단
    (3, 170.0, 210.0),   # ③ 우 (40mm 원이 패턴을 가리지 않게 이격)
]

# 복합 방식: 포인트 1개 (중앙)
COMPOSITE_TEST_PT: list[tuple[int, float, float]] = [
    (1, 105.0, 148.5),
]

# 복합 조합 정의
COMPOSITE_COMBOS = [
    {
        "key":     "comp_A_aruco",
        "label":   "Composite A  —  ArUco only (baseline)",
        "aruco":   True,
        "color":   False,
        "grid":    False,
    },
    {
        "key":     "comp_B_aruco_color",
        "label":   "Composite B  —  ArUco + ColorDot",
        "aruco":   True,
        "color":   True,
        "grid":    False,
    },
    {
        "key":     "comp_C_aruco_grid",
        "label":   "Composite C  —  ArUco + Grid",
        "aruco":   True,
        "color":   False,
        "grid":    True,
    },
    {
        "key":     "comp_D_full",
        "label":   "Composite D  —  ArUco + ColorDot + Grid (Full)",
        "aruco":   True,
        "color":   True,
        "grid":    True,
    },
]

# 색상 원 위치 (복합 시트에서 ArUco와 겹치지 않도록)
_COMP_COLOR_MM: dict[str, tuple[float, float]] = {
    "red":    ( 55.0,  30.0),
    "green":  (155.0,  30.0),
    "blue":   ( 55.0, 267.0),
    "yellow": (155.0, 267.0),
}

CALIB_LAYOUTS: dict[str, dict[str, dict]] = {
    "standard": {
        "aruco": ARUCO_CENTER_MM,
        "color": COLOR_POSITIONS_MM,
    },
    "margin30": {
        "aruco": {
            0: ( 30.0,  30.0),
            1: (180.0,  30.0),
            2: ( 30.0, 267.0),
            3: (180.0, 267.0),
        },
        "color": {
            "red":    ( 30.0,  30.0),
            "green":  (180.0,  30.0),
            "blue":   ( 30.0, 267.0),
            "yellow": (180.0, 267.0),
        },
    },
    "margin35": {
        "aruco": {
            0: ( 35.0,  35.0),
            1: (175.0,  35.0),
            2: ( 35.0, 262.0),
            3: (175.0, 262.0),
        },
        "color": {
            "red":    ( 35.0,  35.0),
            "green":  (175.0,  35.0),
            "blue":   ( 35.0, 262.0),
            "yellow": (175.0, 262.0),
        },
    },
}

ARUCO_SIZE_VARIANTS_MM = [16.0, 20.0, 24.0]
COLOR_RADIUS_VARIANTS_MM = [4.0, 6.0, 8.0]


# ═════════════════════════════════════════════════════════════════════════════
# 공통 드로잉 헬퍼
# ═════════════════════════════════════════════════════════════════════════════

def _save(canvas: np.ndarray, path: Path) -> None:
    """PNG를 200DPI 메타데이터와 함께 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, dpi=(DPI, DPI))
    print(f"[gen] {path.name}  ({_W}x{_H}px, {DPI}DPI)")


def _write_print_pdf(image_paths: list[Path], out_path: Path) -> None:
    """PNG 시트들을 A4 실제 크기 페이지로 묶은 프린트용 PDF를 생성한다."""
    if not image_paths:
        return
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError:
        print("[gen] reportlab 없음: print_ready_a4_sheets.pdf 생성을 건너뜁니다")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    for image_path in sorted(image_paths, key=lambda p: p.name):
        c.drawImage(
            ImageReader(str(image_path)),
            0,
            0,
            width=A4_W_MM * mm,
            height=A4_H_MM * mm,
            preserveAspectRatio=False,
            mask="auto",
        )
        c.showPage()
    c.save()
    print(f"[gen] {out_path.name}  (PDF, {len(image_paths)} A4 pages)")


def _blank() -> np.ndarray:
    return np.full((_H, _W, 3), 255, dtype=np.uint8)


def _draw_border(c: np.ndarray, thickness: int = 3) -> None:
    cv2.rectangle(c, (1, 1), (_W - 2, _H - 2), (0, 0, 0), thickness)


def _draw_printable_edge_border(c: np.ndarray, thickness: int = 5) -> None:
    """일반 프린터 비인쇄 여백을 피해 edge 검출용 테두리를 안쪽에 그린다."""
    inset = _px(EDGE_BORDER_INSET_MM)
    cv2.rectangle(c, (inset, inset), (_W - inset, _H - inset), (0, 0, 0), thickness)


def _draw_grid(
    c: np.ndarray,
    spacing_mm: float,
    color_major=(160, 160, 160),
    color_minor=(215, 215, 215),
    major_every_mm: float = 50.0,
    labels: bool = True,
    line_thickness: int = 1,
) -> None:
    """격자선 + 50mm 단위 좌표 라벨."""
    sp = spacing_mm
    x_mm = 0.0
    while x_mm <= A4_W_MM + 0.1:
        is_major = (round(x_mm) % round(major_every_mm) == 0)
        c_line = color_major if is_major else color_minor
        cv2.line(c, (_px(x_mm), 0), (_px(x_mm), _H), c_line, line_thickness)
        if labels and is_major and x_mm > 0:
            cv2.putText(c, str(int(x_mm)), (_px(x_mm) + 2, 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 130, 130), 1)
        x_mm += sp

    y_mm = 0.0
    while y_mm <= A4_H_MM + 0.1:
        is_major = (round(y_mm) % round(major_every_mm) == 0)
        c_line = color_major if is_major else color_minor
        cv2.line(c, (0, _px(y_mm)), (_W, _px(y_mm)), c_line, line_thickness)
        if labels and is_major and y_mm > 0:
            cv2.putText(c, str(int(y_mm)), (3, _px(y_mm) + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 130, 130), 1)
        y_mm += sp


def _draw_top_indicator(c: np.ndarray, text: bool = True) -> None:
    """
    상단 방향 표시 (TOP) — 용지를 잘못 뒤집지 않도록.
    A4 상단 중앙 안쪽에 작게 표시.
    """
    cx = _W // 2
    y  = _px(6.5)
    if text:
        cv2.putText(c, "TOP", (cx - _px(4.5), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)
    # 작은 삼각형 (▲)
    tri = np.array([
        [cx,          y - _px(3.5)],
        [cx - _px(2), y + _px(0.5)],
        [cx + _px(2), y + _px(0.5)],
    ], dtype=np.int32)
    cv2.fillPoly(c, [tri], (160, 160, 160))


def _draw_test_points(
    c: np.ndarray,
    pts: list[tuple[int, float, float]],
    dot_color:    tuple = (30,  30,  30),
    cross_color:  tuple = (60,  60,  60),
    label_color:  tuple = (40,  40,  40),
    dot_r_mm:     float = 4.5,
    arm_mm:       float = 9.0,
) -> None:
    """
    테스트 포인트 마커 드로잉.
    - 채운 원(번호) + 십자선 + 좌표 라벨
    """
    r = _px(dot_r_mm)
    s = _px(arm_mm)

    for num, xm, ym in pts:
        cx, cy = _px(xm), _px(ym)

        # 십자선
        cv2.line(c, (cx - s, cy), (cx + s, cy), cross_color, 2)
        cv2.line(c, (cx, cy - s), (cx, cy + s), cross_color, 2)

        # 채운 원 (흰 테두리 포함)
        cv2.circle(c, (cx, cy), r + 1, (255, 255, 255), -1)
        cv2.circle(c, (cx, cy), r,     dot_color,       -1)

        # 번호 (흰 글씨)
        text  = str(num)
        scale = 0.52
        thick = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(c, text,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)

        # 좌표 라벨
        coord = f"({xm:.0f},{ym:.1f})" if ym != int(ym) else f"({xm:.0f},{ym:.0f})"
        cv2.putText(c, coord,
                    (cx + r + 4, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, label_color, 1)


def _draw_target_circle(
    c: np.ndarray,
    pt: tuple[int, float, float],
    diameter_mm: float = TARGET_DIAMETER_MM,
) -> None:
    """실물 객체 배치용 40mm 원. 중앙에는 포인트 번호만 표시한다."""
    num, xm, ym = pt
    cx, cy = _px(xm), _px(ym)
    r = _px(diameter_mm / 2.0)

    cv2.circle(c, (cx, cy), r, (0, 0, 0), 3)

    text = str(num)
    scale = 1.15 if num < 10 else 0.95
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(c, text,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick)


def _draw_aruco_markers(
    c: np.ndarray,
    size_mm: float = 20.0,
    labels: bool = True,
    positions: dict[int, tuple[float, float]] | None = None,
) -> None:
    positions = positions or ARUCO_CENTER_MM
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    mpx = _px(size_mm)
    for mid, (cx_mm, cy_mm) in positions.items():
        m = np.zeros((mpx, mpx), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, mid, mpx, m, 1)
        m_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        x0 = _px(cx_mm) - mpx // 2
        y0 = _px(cy_mm) - mpx // 2
        x1 = min(x0 + mpx, _W)
        y1 = min(y0 + mpx, _H)
        if x0 >= 0 and y0 >= 0:
            c[y0:y1, x0:x1] = m_bgr[:y1-y0, :x1-x0]
        if labels:
            cv2.putText(c, f"ID{mid}",
                        (_px(cx_mm) - mpx // 2, _px(cy_mm) + mpx // 2 + _px(3.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (40, 40, 140), 1)


def _draw_color_dots(
    c: np.ndarray,
    positions: dict[str, tuple[float, float]],
    r_mm: float = 5.5,
    labels: bool = True,
) -> None:
    CV_COLORS = {
        "red":    (  0,   0, 200),
        "green":  (  0, 150,   0),
        "blue":   (180,  40,   0),
        "yellow": (  0, 185, 190),
    }
    r = _px(r_mm)
    for color, (xm, ym) in positions.items():
        cx, cy = _px(xm), _px(ym)
        cv2.circle(c, (cx, cy), r, CV_COLORS[color], -1)
        cv2.circle(c, (cx, cy), r + 1, (0, 0, 0), 1)
        if labels:
            cv2.putText(c, f"{color[0].upper()}({xm:.0f},{ym:.0f})",
                        (cx + r + 3, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, CV_COLORS[color], 1)


def _draw_title(c: np.ndarray, text: str) -> None:
    cv2.putText(c, text, (_px(10), _px(9)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)


def _draw_footer(c: np.ndarray, text: str) -> None:
    cv2.putText(c, text, (_px(5), _H - _px(4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (70, 70, 70), 1)


def _draw_sheet_sequence(c: np.ndarray, seq: int, total: int | None = None) -> None:
    """Draw a small print-order number on the sheet without touching markers."""
    text = f"{seq:03d}" if total is None else f"{seq:03d}/{total:03d}"
    scale = 0.55
    thick = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x = (_W - tw) // 2
    y = _H - _px(5.5)
    pad = _px(1.5)
    cv2.rectangle(
        c,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (255, 255, 255),
        -1,
    )
    cv2.putText(c, text, (x, y), font, scale, (40, 40, 40), thick)


def _draw_checkerboard_pattern(c: np.ndarray, labels: bool = True) -> None:
    sq = _px(SQUARE_MM)
    total_cols = CHESSBOARD_COLS + 1
    total_rows = CHESSBOARD_ROWS + 1
    ox = _px(CB_ORIGIN_MM[0]) - sq
    oy = _px(CB_ORIGIN_MM[1]) - sq

    for r in range(total_rows):
        for col in range(total_cols):
            if (r + col) % 2 == 0:
                x0, y0 = ox + col * sq, oy + r * sq
                cv2.rectangle(c, (x0, y0), (x0 + sq, y0 + sq), (0, 0, 0), -1)

    if labels:
        cv2.circle(c, (_px(CB_ORIGIN_MM[0]), _px(CB_ORIGIN_MM[1])), _px(2.5), (0, 0, 200), -1)
        cv2.putText(c, f"Inner[0,0]=({CB_ORIGIN_MM[0]:.0f},{CB_ORIGIN_MM[1]:.0f})",
                    (_px(CB_ORIGIN_MM[0]) + _px(3), _px(CB_ORIGIN_MM[1]) - _px(2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 180), 1)


def _draw_grid_detection_lines(c: np.ndarray, labels: bool = True) -> None:
    """격자 시트용 진한 격자선 (탐지 대상). _draw_grid 의 색상 프리셋."""
    _draw_grid(
        c,
        GRID_SPACING_MM,
        color_major=(25, 25, 25),
        color_minor=(25, 25, 25),
        labels=labels,
        line_thickness=2,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. 외곽선 시트 — 마커 없음, 테스트 포인트 5개
# ═════════════════════════════════════════════════════════════════════════════

def gen_edge_sheet(out_dir: Path) -> Path:
    """
    흰 A4 + 굵은 테두리만.
    Canny 엣지 탐지기가 용지 외곽을 자동 검출.
    테스트 포인트 5개: 방향(좌우위아래) 파악 + 전체 영역 커버.
    """
    c = _blank()

    # 연한 격자 (시각 참고용 — 탐지에는 영향 없음)
    _draw_grid(c, 10.0, (210, 210, 210), (230, 230, 230), 50.0)

    # 굵은 외곽 테두리 (엣지 탐지 안정성 향상)
    _draw_printable_edge_border(c, thickness=5)

    # 방향 표시
    _draw_top_indicator(c)

    # 테스트 포인트 5개
    _draw_test_points(c, EDGE_TEST_PTS,
                      dot_color=(20, 20, 20), cross_color=(50, 50, 50))

    _draw_title(c, "EDGE SHEET  (exterior contour)  —  test pts: 5")
    _draw_footer(c,
        f"Calibration: printed edge border inset={EDGE_BORDER_INSET_MM:.0f}mm  "
        "|  Pts: 1~5 for orientation check  |  Print 100%")

    out = out_dir / "sheet_edge.png"
    _save(c, out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2. ArUco 시트 — 4 코너 마커, 테스트 포인트 3개
# ═════════════════════════════════════════════════════════════════════════════

def gen_aruco_sheet(out_dir: Path) -> Path:
    c = _blank()
    _draw_grid(c, 10.0)
    _draw_top_indicator(c)
    _draw_aruco_markers(c, size_mm=20.0)
    _draw_test_points(c, SINGLE_TEST_PTS)
    _draw_title(c, "ArUco SHEET  (DICT_4X4_50, size=20mm)  —  test pts: 3")
    _draw_footer(c,
        f"Calibration: ArUco "
        f"ID0=TL({ARUCO_CENTER_MM[0][0]:.0f},{ARUCO_CENTER_MM[0][1]:.0f})  "
        f"ID1=TR({ARUCO_CENTER_MM[1][0]:.0f},{ARUCO_CENTER_MM[1][1]:.0f})  "
        f"ID2=BL({ARUCO_CENTER_MM[2][0]:.0f},{ARUCO_CENTER_MM[2][1]:.0f})  "
        f"ID3=BR({ARUCO_CENTER_MM[3][0]:.0f},{ARUCO_CENTER_MM[3][1]:.0f})  |  Print 100%")
    out = out_dir / "sheet_aruco.png"
    _save(c, out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3. 색상점 시트 — 4색 원, 테스트 포인트 3개
# ═════════════════════════════════════════════════════════════════════════════

def gen_color_dot_sheet(out_dir: Path) -> Path:
    c = _blank()
    _draw_grid(c, 10.0)
    _draw_top_indicator(c)
    _draw_color_dots(c, COLOR_POSITIONS_MM, r_mm=6.0)
    _draw_test_points(c, SINGLE_TEST_PTS)
    _draw_title(c, "COLOR DOT SHEET  (R/G/B/Y markers)  —  test pts: 3")
    _draw_footer(c,
        "Calibration: R(25,25) G(185,25) B(25,272) Y(185,272)  r=6mm  |  Print 100%")
    out = out_dir / "sheet_color_dot.png"
    _save(c, out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. 체커보드 시트 — 7×5 내부코너 / 20mm 칸, 테스트 포인트 3개 (패턴 아래)
# ═════════════════════════════════════════════════════════════════════════════

def gen_checkerboard_sheet(out_dir: Path) -> Path:
    c = _blank()
    _draw_top_indicator(c)

    _draw_checkerboard_pattern(c)

    # 테스트 포인트 3개 (패턴 아래 영역)
    _draw_test_points(c, CHECKER_TEST_PTS)

    _draw_title(c, f"CHECKERBOARD SHEET  ({CHESSBOARD_COLS}x{CHESSBOARD_ROWS} inner, {SQUARE_MM:.0f}mm)  —  test pts: 3")
    _draw_footer(c,
        f"Calibration: inner TL=({CB_ORIGIN_MM[0]:.0f},{CB_ORIGIN_MM[1]:.0f})mm  "
        f"sq={SQUARE_MM:.0f}mm  |  Test pts below pattern  |  Print 100%")
    _draw_border(c)

    out = out_dir / "sheet_checkerboard.png"
    _save(c, out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5. 그리드 시트 — 20mm 격자, 테스트 포인트 3개
# ═════════════════════════════════════════════════════════════════════════════

def gen_grid_sheet(out_dir: Path) -> Path:
    c = _blank()

    # 진한 격자 (탐지용) — 간격 GRID_SPACING_MM
    _draw_grid_detection_lines(c)

    _draw_top_indicator(c)
    _draw_test_points(c, SINGLE_TEST_PTS,
                      dot_color=(0, 0, 0), cross_color=(0, 0, 0))
    _draw_border(c, thickness=3)
    _draw_title(c, f"GRID SHEET  ({GRID_SPACING_MM:.0f}mm spacing)  —  test pts: 3")
    _draw_footer(c, f"Calibration: {GRID_SPACING_MM:.0f}mm grid lines  |  Origin(0,0)=top-left  |  Print 100%")

    out = out_dir / "sheet_grid.png"
    _save(c, out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 6. 복합 시트 — 테스트 포인트 1개, 조합 방식별 4종
# ═════════════════════════════════════════════════════════════════════════════

def gen_composite_sheets(out_dir: Path) -> list[Path]:
    """
    ArUco / 색상점 / 격자를 다양하게 조합한 4종 복합 시트.
    테스트 포인트는 모두 동일 위치(중앙) — 방법 간 정확도 비교용.
    """
    paths = []
    for combo in COMPOSITE_COMBOS:
        c = _blank()

        # 격자
        if combo["grid"]:
            _draw_grid(c, 10.0,
                       color_major=(105, 105, 105),
                       color_minor=(175, 175, 175))
        else:
            _draw_grid(c, 10.0,
                       color_major=(200, 200, 200),
                       color_minor=(225, 225, 225))

        # ArUco 마커 (aruco.py 와 통일: 20mm)
        if combo["aruco"]:
            _draw_aruco_markers(c, size_mm=20.0)

        # 색상 원 (ArUco와 겹치지 않는 위치)
        if combo["color"]:
            _draw_color_dots(c, _COMP_COLOR_MM, r_mm=5.5)

        _draw_top_indicator(c)

        # 테스트 포인트 1개 (중앙)
        _draw_test_points(c, COMPOSITE_TEST_PT,
                          dot_color=(180, 0, 0), cross_color=(140, 0, 0))

        _draw_border(c, thickness=3)
        _draw_title(c, combo["label"])

        methods_used = (
            (["ArUco"] if combo["aruco"] else []) +
            (["ColorDot"] if combo["color"] else []) +
            (["Grid"] if combo["grid"] else [])
        )
        _draw_footer(c,
            f"Methods: {' + '.join(methods_used)}  |  Test pt: (105,148.5) center  "
            f"|  Print 100%  |  {combo['key']}")

        out = out_dir / f"sheet_{combo['key']}.png"
        _save(c, out)
        paths.append(out)

    return paths


# ═════════════════════════════════════════════════════════════════════════════
# 7. 좌표 오차 실험 시트 — ArUco 캘리브레이션 + 30 테스트 포인트 (5×6 격자)
# ═════════════════════════════════════════════════════════════════════════════

# 30점 격자: EVAL_X_MM × EVAL_Y_MM — eval/__init__.py 에서 임포트 (단일 정의)

QUICK_TEST_PTS: list[tuple[int, float, float]] = [
    # 40mm 캡(반경 20mm)이 ArUco 마커(중심 ±10mm)와 겹치지 않으려면
    # 최근접 마커 모서리까지 20mm 이상 이격이 필요.
    # 각 코너 포인트는 해당 ArUco 마커 모서리까지 ≈39mm 이격.
    (1, 105.0, 148.5),   # ① 중앙
    (2,  60.0,  65.0),   # ② 좌상  (TL ArUco 모서리(35,35)까지 39mm)
    (3, 150.0,  65.0),   # ③ 우상  (TR ArUco 모서리(175,35)까지 39mm)
    (4,  60.0, 232.0),   # ④ 좌하  (BL ArUco 모서리(35,262)까지 39mm)
    (5, 150.0, 232.0),   # ⑤ 우하  (BR ArUco 모서리(175,262)까지 39mm)
]


def gen_eval_sheet(out_dir: Path) -> Path:
    """
    좌표 오차 실험용 시트.
    ArUco 캘리브레이션 마커(ID0~3) + 30개 테스트 포인트 (5×6 격자).
    번호 1~30: 왼→오, 위→아래 순서.
    물체를 각 번호 위에 올려놓고 YOLO 예측 좌표와 비교.
    """
    c = _blank()
    _draw_grid(c, 10.0)               # 연한 10mm 격자
    _draw_aruco_markers(c, size_mm=20.0)  # A4 검출용 ArUco (aruco.py 와 통일: 20mm)
    _draw_top_indicator(c)

    # 테스트 포인트 30개 (번호 + 좌표 라벨)
    _draw_test_points(
        c, EVAL_TEST_PTS,
        dot_color   = (20,  20,  20),
        cross_color = (80,  80,  80),
        label_color = (40,  40,  40),
        dot_r_mm    = 3.8,
        arm_mm      = 7.0,
    )

    # X / Y 축 라벨 (격자 구조 이해용)
    for xm in EVAL_X_MM:
        cv2.putText(c, f"X={xm:.0f}", (_px(xm) - _px(3), _px(38)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 100), 1)
    for ym in EVAL_Y_MM:
        cv2.putText(c, f"Y={ym:.0f}", (_px(3), _px(ym) + _px(1.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 100), 1)

    _draw_border(c, thickness=3)
    _draw_title(c, "COORD EVAL SHEET  (ArUco + 30 pts, 5x6)  for mm error test")
    x_str = " ".join(f"{x:.0f}" for x in EVAL_X_MM)
    y_str = " ".join(f"{y:.0f}" for y in EVAL_Y_MM)
    _draw_footer(c,
        f"X: {x_str} mm  |  Y: {y_str} mm"
        f"  |  {len(EVAL_TEST_PTS)} pts x 3 repeats = {len(EVAL_TEST_PTS)*3} samples  |  Print 100%")

    out = out_dir / "sheet_eval_30pt.png"
    _save(c, out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 8. 좌표 오차 실험 시트 묶음 — 한 장당 테스트 포인트 1개
# ═════════════════════════════════════════════════════════════════════════════

ONE_POINT_METHODS = ["edge", "aruco", "color_dot", "checkerboard", "grid"]
ONE_POINT_CHOICES = ONE_POINT_METHODS + ["composite"]


def _draw_method_base(c: np.ndarray, method: str, combo: dict | None = None) -> str:
    """좌표 실험용 1점 시트의 A4 탐지 기준 요소를 그린다."""
    if method == "edge":
        _draw_grid(c, 10.0, (225, 225, 225), (242, 242, 242), 50.0, labels=False)
        _draw_printable_edge_border(c, thickness=5)
        _draw_top_indicator(c, text=False)
        return "Edge"

    if method == "aruco":
        _draw_grid(c, 10.0, labels=False)
        _draw_aruco_markers(c, size_mm=20.0, labels=False)
        _draw_top_indicator(c, text=False)
        _draw_border(c, thickness=3)
        return "ArUco"

    if method == "color_dot":
        _draw_grid(c, 10.0, labels=False)
        _draw_color_dots(c, COLOR_POSITIONS_MM, r_mm=6.0, labels=False)
        _draw_top_indicator(c, text=False)
        _draw_border(c, thickness=3)
        return "ColorDot"

    if method == "checkerboard":
        _draw_checkerboard_pattern(c, labels=False)
        _draw_top_indicator(c, text=False)
        _draw_border(c, thickness=3)
        return "Checkerboard"

    if method == "grid":
        _draw_grid_detection_lines(c, labels=False)
        _draw_top_indicator(c, text=False)
        _draw_border(c, thickness=3)
        return "Grid"

    if method == "composite":
        if combo is None:
            raise ValueError("composite one-point sheet requires combo")
        if combo["grid"]:
            _draw_grid(c, 10.0,
                       color_major=(105, 105, 105),
                       color_minor=(175, 175, 175),
                       labels=False)
        else:
            _draw_grid(c, 10.0,
                       color_major=(210, 210, 210),
                       color_minor=(232, 232, 232),
                       labels=False)
        if combo["aruco"]:
            _draw_aruco_markers(c, size_mm=20.0, labels=False)
        if combo["color"]:
            _draw_color_dots(c, _COMP_COLOR_MM, r_mm=5.5, labels=False)
        _draw_top_indicator(c, text=False)
        _draw_border(c, thickness=3)
        return combo["label"]

    raise ValueError(f"unknown method: {method}")


def _point_name(pt: tuple[int, float, float]) -> str:
    num, x, y = pt
    return f"pt{num:02d}_{x:.0f}x_{y:.0f}y"


def _one_point_pts_for_method(method: str) -> list[tuple[int, float, float]]:
    # 체커보드는 패턴 위에 물체가 올라가면 A4 검출 자체를 가릴 수 있으므로
    # 패턴 아래 안전 영역 포인트만 사용한다.
    return CHECKER_TEST_PTS if method == "checkerboard" else QUICK_TEST_PTS


def _selected_one_point_series(
    only: str | None,
    combo_key: str,
) -> list[tuple[str, dict | None]]:
    if only and only not in ONE_POINT_CHOICES:
        raise ValueError(
            "--one-point requires --only edge|aruco|color_dot|checkerboard|grid|composite"
        )

    methods = [only] if only else ONE_POINT_METHODS + ["composite"]
    combos = [
        combo for combo in COMPOSITE_COMBOS
        if combo_key == "all" or combo["key"] == combo_key
    ]
    if combo_key != "all" and not combos:
        raise ValueError(f"unknown composite combo: {combo_key}")

    series: list[tuple[str, dict | None]] = []
    for method in methods:
        if method == "composite":
            series.extend(("composite", combo) for combo in combos)
        else:
            series.append((method, None))
    return series


def _one_point_sheet_count(only: str | None, combo_key: str) -> int:
    return sum(
        len(_one_point_pts_for_method(method))
        for method, _ in _selected_one_point_series(only, combo_key)
    )


def gen_one_point_eval_sheets(
    out_dir: Path,
    only: str | None = None,
    combo_key: str = "all",
    seq_start: int = 1,
    total_override: int | None = None,
) -> list[Path]:
    """
    실제 좌표 오차 실험용 시트 묶음.
    한 PNG에는 A4 검출 기준 요소 + 테스트 포인트 1개만 들어간다.
    """
    series = _selected_one_point_series(only, combo_key)
    total = total_override or sum(len(_one_point_pts_for_method(method)) for method, _ in series)
    seq = seq_start
    paths: list[Path] = []
    for method, combo in series:
        batch = _gen_one_point_method_series(out_dir, method, combo, seq_start=seq, total=total)
        paths.extend(batch)
        seq += len(batch)
    return paths


def _gen_one_point_method_series(
    out_dir: Path,
    method: str,
    combo: dict | None,
    seq_start: int = 1,
    total: int | None = None,
) -> list[Path]:
    key = combo["key"] if combo else method
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for offset, pt in enumerate(_one_point_pts_for_method(method)):
        seq = seq_start + offset
        num, x, y = pt
        c = _blank()
        _draw_method_base(c, method, combo)
        _draw_target_circle(c, pt, diameter_mm=TARGET_DIAMETER_MM)
        _draw_sheet_sequence(c, seq, total)

        out = out_dir / f"{seq:03d}_{key}_{_point_name(pt)}.png"
        _save(c, out)
        paths.append(out)

    return paths


def _calib_variant_sheet_count() -> int:
    return len(CALIB_LAYOUTS) * (len(ARUCO_SIZE_VARIANTS_MM) + len(COLOR_RADIUS_VARIANTS_MM))


def gen_calibration_variant_sheets(
    out_dir: Path,
    seq_start: int = 1,
    total_override: int | None = None,
) -> list[Path]:
    """
    A4 기준점 자체의 크기/위치 안정성을 보기 위한 출력 시트.
    중앙 40mm 표시 원 1개만 넣고, ArUco / 색상점의 크기와 배치를 바꾼다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    target_pt = (1, 105.0, 148.5)

    tasks: list[tuple[str, str, float, dict]] = []
    for layout_key in CALIB_LAYOUTS:
        for size_mm in ARUCO_SIZE_VARIANTS_MM:
            tasks.append(("aruco", layout_key, size_mm, CALIB_LAYOUTS[layout_key]))
        for radius_mm in COLOR_RADIUS_VARIANTS_MM:
            tasks.append(("color_dot", layout_key, radius_mm, CALIB_LAYOUTS[layout_key]))

    total = total_override or len(tasks)
    for seq, (kind, layout_key, value_mm, layout) in enumerate(tasks, start=seq_start):
        c = _blank()
        _draw_grid(c, 10.0, color_major=(220, 220, 220), color_minor=(242, 242, 242), labels=False)
        if kind == "aruco":
            _draw_aruco_markers(c, size_mm=value_mm, labels=False, positions=layout["aruco"])
            suffix = f"aruco_{layout_key}_size{value_mm:.0f}mm"
        else:
            _draw_color_dots(c, layout["color"], r_mm=value_mm, labels=False)
            suffix = f"color_dot_{layout_key}_radius{value_mm:.0f}mm"
        _draw_top_indicator(c, text=False)
        _draw_border(c, thickness=3)
        _draw_target_circle(c, target_pt)
        _draw_sheet_sequence(c, seq, total)
        out = out_dir / f"{seq:03d}_calib_{suffix}_pt01.png"
        _save(c, out)
        paths.append(out)

    return paths


# ═════════════════════════════════════════════════════════════════════════════
# 9. 카메라 렌즈 왜곡 보정용 체커보드 시트 (PDF)
# ═════════════════════════════════════════════════════════════════════════════

_CALIB_GRID_INNER = (9, 6)   # 내부 코너 수 (cols, rows)  →  칸 수 10×7
_CALIB_SQUARE_MM  = 25.0     # 체커 한 칸 크기 (mm)


def gen_calib_checkerboard_sheet(
    out_dir: Path,
    grid: tuple[int, int] = _CALIB_GRID_INNER,
    square_mm: float = _CALIB_SQUARE_MM,
    out_path: Path | None = None,
) -> Path:
    """
    카메라 렌즈 왜곡 보정용 체커보드 시트를 A4 PDF로 생성.

    기본값: 내부 코너 9×6 (칸 수 10×7), 한 칸 25mm.
    PDF 포맷 사용 이유: mm 단위 직접 지정 → 100% 인쇄 시 실제 크기 보장.
    calibrate_camera.py --gen-sheet 과 동일한 출력.

    out_path 를 지정하면 해당 경로에 직접 저장 (out_dir 무시).
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError:
        raise SystemExit("reportlab 필요: pip install reportlab")

    cols, rows = grid
    ncols, nrows = cols + 1, rows + 1          # 칸 수 = 내부코너 + 1
    total_w_mm = ncols * square_mm
    total_h_mm = nrows * square_mm

    A4_W_MM, A4_H_MM = 210.0, 297.0
    page_w_mm, page_h_mm = A4_W_MM, A4_H_MM
    pagesize = A4
    orientation = "portrait"
    if total_w_mm <= A4_H_MM and total_h_mm <= A4_W_MM:
        page_w_mm, page_h_mm = A4_H_MM, A4_W_MM
        pagesize = landscape(A4)
        orientation = "landscape"

    if total_w_mm > page_w_mm or total_h_mm > page_h_mm:
        raise ValueError(
            f"체커보드 {total_w_mm:.0f}×{total_h_mm:.0f}mm 가 A4를 초과합니다. "
            f"grid 또는 square_mm 을 줄이세요."
        )

    if out_path is not None:
        resolved = Path(out_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved = out_dir / f"sheet_checkerboard_calib_{cols}x{rows}_{square_mm:.0f}mm.pdf"

    off_x = (page_w_mm - total_w_mm) / 2.0
    off_y = (page_h_mm - total_h_mm) / 2.0

    c = rl_canvas.Canvas(str(resolved), pagesize=pagesize)
    for row in range(nrows):
        for col in range(ncols):
            if (row + col) % 2 == 0:
                x = (off_x + col * square_mm) * mm
                # reportlab Y축: 아래=0, 위=max
                y = (page_h_mm - off_y - (row + 1) * square_mm) * mm
                c.rect(x, y, square_mm * mm, square_mm * mm, fill=1, stroke=0)

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(
        page_w_mm / 2 * mm, 8 * mm,
        f"Camera Calib Checkerboard  |  grid={cols}×{rows} inner corners"
        f"  |  square={square_mm:.0f}mm  |  {orientation}  |  반드시 100%(실제 크기)로 인쇄",
    )
    c.save()
    print(f"[gen] {resolved.name}  (PDF, {cols}x{rows} inner, {square_mm:.0f}mm/sq, {orientation})")
    return resolved


# ═════════════════════════════════════════════════════════════════════════════
# 전체 생성
# ═════════════════════════════════════════════════════════════════════════════

_SINGLE_GENERATORS = {
    "edge":         gen_edge_sheet,
    "aruco":        gen_aruco_sheet,
    "color_dot":    gen_color_dot_sheet,
    "checkerboard": gen_checkerboard_sheet,
    "grid":         gen_grid_sheet,
    "eval":         gen_eval_sheet,
}


def gen_all_sheets(
    out_dir: Path | None = None,
    only: str | None     = None,
    one_point: bool      = False,
    combo_key: str       = "all",
    calib_variants: bool = False,
    calib_sheet: bool    = False,
) -> None:
    if out_dir is None:
        out_dir = _HERE / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[gen] output: {out_dir}\n")
    paths: list[Path] = []

    # ── 캘리브레이션 체커보드 단독 생성 ───────────────────────────────────────
    if only == "calib_checkerboard":
        gen_calib_checkerboard_sheet(out_dir)
        print("\n[gen] done  total=1 sheets (PDF)")
        print("[gen] 반드시 100%(실제 크기 / 배율 없음)로 인쇄하세요")
        return

    if calib_variants and one_point:
        one_point_only = None if only in (None, "eval", "eval_one_point") else only
        total = _one_point_sheet_count(one_point_only, combo_key) + _calib_variant_sheet_count()
        paths = gen_one_point_eval_sheets(
            out_dir,
            only=one_point_only,
            combo_key=combo_key,
            seq_start=1,
            total_override=total,
        )
        paths.extend(gen_calibration_variant_sheets(
            out_dir,
            seq_start=len(paths) + 1,
            total_override=total,
        ))
        n = len(paths)
    elif calib_variants:
        paths = gen_calibration_variant_sheets(out_dir)
        n = len(paths)
    elif one_point:
        one_point_only = None if only in (None, "eval", "eval_one_point") else only
        paths = gen_one_point_eval_sheets(out_dir, only=one_point_only, combo_key=combo_key)
        n = len(paths)
    elif only and only in _SINGLE_GENERATORS:
        paths = [_SINGLE_GENERATORS[only](out_dir)]
        n = 1
    elif only == "composite":
        paths = gen_composite_sheets(out_dir)
        n = len(COMPOSITE_COMBOS)
    else:
        paths = []
        for fn in _SINGLE_GENERATORS.values():
            paths.append(fn(out_dir))
        paths.extend(gen_composite_sheets(out_dir))
        n = len(_SINGLE_GENERATORS) + len(COMPOSITE_COMBOS)

    # ── 캘리브레이션 체커보드 추가 생성 ───────────────────────────────────────
    if calib_sheet:
        gen_calib_checkerboard_sheet(out_dir)
        n += 1

    if paths:
        _write_print_pdf(paths, out_dir / "print_ready_a4_sheets.pdf")

    print(f"\n[gen] done  total={n} sheets")
    print("[gen] Print at 100% scale / no scaling / no fit-to-page")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="A4 탐지 방법별 테스트 시트 생성",
        epilog=(
            "예시:\n"
            "  python sheets/gen.py                          # 모든 연구 시트\n"
            "  python sheets/gen.py --calib-sheet            # 연구 시트 + 카메라 캘리브레이션 PDF\n"
            "  python sheets/gen.py --only calib_checkerboard # 캘리브레이션 PDF만\n"
            "  python sheets/gen.py --only aruco             # ArUco 시트만\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out",  default=None,
                   help="출력 디렉터리 (기본: sheets/output)")
    p.add_argument("--only", default=None,
                   choices=(
                       list(_SINGLE_GENERATORS.keys())
                       + ["composite", "eval_one_point", "calib_checkerboard"]
                   ),
                   help="특정 시트만 생성 (calib_checkerboard=캘리브레이션 PDF)")
    p.add_argument("--one-point", action="store_true",
                   help="좌표 실험용: 한 장당 테스트 포인트 1개씩 생성")
    p.add_argument("--combo", default="all",
                   choices=["all"] + [combo["key"] for combo in COMPOSITE_COMBOS],
                   help="--one-point --only composite 에서 생성할 복합 조합")
    p.add_argument("--calib-variants", action="store_true",
                   help="ArUco/색상점 크기와 위치 변형 테스트 시트 생성")
    p.add_argument("--calib-sheet", action="store_true",
                   help=(
                       "카메라 렌즈 왜곡 보정용 체커보드 PDF 추가 생성 "
                       f"(기본: {_CALIB_GRID_INNER[0]}×{_CALIB_GRID_INNER[1]} inner, "
                       f"{_CALIB_SQUARE_MM:.0f}mm/sq)"
                   ))
    args = p.parse_args()
    only = None if args.only == "eval_one_point" else args.only
    gen_all_sheets(
        out_dir=Path(args.out) if args.out else None,
        only=only,
        one_point=args.one_point or args.only == "eval_one_point",
        combo_key=args.combo,
        calib_variants=args.calib_variants,
        calib_sheet=args.calib_sheet,
    )
