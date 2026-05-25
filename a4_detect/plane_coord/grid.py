"""
grid.py — Method 5: 그리드 라인 기반 A4 탐지

수평/수직 격자선을 Hough 변환으로 검출 →
교차점 군집화 → 정규 격자 구조 매칭 → 호모그래피 계산.

장점: 별도 마커 불필요, 그리드 시트만 있으면 동작
     많은 대응점(교차점)으로 robust한 호모그래피 추정 가능
단점: 격자가 선명하게 인쇄되어야 함
     조명 불균일 시 일부 선이 누락됨
     계산량이 다른 방법보다 많음 (Hough 변환)

⚠️  사용 제한 — 비교 실험 전용:
     그리드 방식은 격자 원점을 카메라 뷰 내에서 절대 위치로 확정할 수 없어
     검출 성공 시에도 A4 좌표 원점(0,0)이 잘못 추정될 수 있습니다.
     실제 좌표 측정 실험(--eval)에는 aruco 또는 composite 방식을 사용하세요.
     --compare / --benchmark / --precheck --all-methods 비교 시에만 포함하세요.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import A4_W_MM, A4_H_MM, BaseA4Detector, DetectResult

# ── 그리드 파라미터 (시트와 반드시 일치) ──────────────────────────────────────
GRID_SPACING_MM: float = 20.0   # 격자 간격 (mm)

# A4 내 그리드 시작 위치 (mm) — 시트에서 그리드가 (0,0)에서 시작하면 (0,0)
GRID_ORIGIN_MM: tuple[float, float] = (0.0, 0.0)


class GridDetector(BaseA4Detector):
    """
    Hough 직선 + 교차점 군집화 기반 A4 탐지.

    Parameters
    ----------
    hough_thresh   : HoughLinesP threshold (교차점 수 임계값)
    min_line_len   : HoughLinesP 최소 선 길이 (px)
    max_line_gap   : HoughLinesP 선 간격 허용값 (px)
    angle_tol_deg  : 수평/수직 판단 각도 허용 오차 (°)
    min_lines      : 각 방향 최소 필요 라인 수
    cluster_ratio  : 군집화 임계값 = frame 단변 × ratio
    min_intersections : 호모그래피에 필요한 최소 교차점 수
    """

    def __init__(
        self,
        hough_thresh: int        = 55,
        min_line_len: int        = 40,
        max_line_gap: int        = 10,
        angle_tol_deg: float     = 18.0,
        min_lines: int           = 4,
        cluster_ratio: float     = 0.025,
        min_intersections: int   = 9,
    ):
        super().__init__("grid")
        self.hough_thresh       = hough_thresh
        self.min_line_len       = min_line_len
        self.max_line_gap       = max_line_gap
        self.angle_tol_deg      = angle_tol_deg
        self.min_lines          = min_lines
        self.cluster_ratio      = cluster_ratio
        self.min_intersections  = min_intersections

    def detect(self, frame: np.ndarray) -> DetectResult:
        result = DetectResult()
        h, w   = frame.shape[:2]
        debug  = frame.copy()

        # ── 전처리 ──────────────────────────────────────────────────────────
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 적응형 이진화 → 라인 강조
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 8,
        )
        edges  = cv2.Canny(binary, 30, 100)

        # ── Hough 직선 변환 ──────────────────────────────────────────────────
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            self.hough_thresh,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            result.note      = "Hough 직선 없음"
            result.debug_img = debug
            return result

        h_lines, v_lines = _classify_lines(lines, self.angle_tol_deg)

        for x1, y1, x2, y2 in h_lines:
            cv2.line(debug, (x1,y1), (x2,y2), (200, 80, 0), 1)
        for x1, y1, x2, y2 in v_lines:
            cv2.line(debug, (x1,y1), (x2,y2), (0, 80, 200), 1)

        if len(h_lines) < self.min_lines or len(v_lines) < self.min_lines:
            result.note      = (f"라인 부족: H={len(h_lines)} V={len(v_lines)} "
                                f"(최소 {self.min_lines})")
            result.debug_img = debug
            return result

        # ── 교차점 계산 ──────────────────────────────────────────────────────
        intersections = _compute_intersections(h_lines, v_lines, w, h)

        if len(intersections) < self.min_intersections:
            result.note      = f"교차점 {len(intersections)}개 (최소 {self.min_intersections})"
            result.debug_img = debug
            return result

        # ── 교차점 군집화 → 격자 구조 추출 ──────────────────────────────────
        cluster_th = min(h, w) * self.cluster_ratio
        x_groups   = _merge_close([p[0] for p in intersections], cluster_th)
        y_groups   = _merge_close([p[1] for p in intersections], cluster_th)

        if len(x_groups) < 2 or len(y_groups) < 2:
            result.note      = "그리드 군집화 실패"
            result.debug_img = debug
            return result

        # ── 픽셀 간격으로 mm 격자 오프셋 추정 ───────────────────────────────
        grid_px, grid_mm = _build_grid_correspondences(
            x_groups, y_groups, w, h
        )

        if len(grid_px) < 4:
            result.note      = f"유효 격자점 {len(grid_px)}개 (최소 4)"
            result.debug_img = debug
            return result

        for px, py in grid_px:
            cv2.circle(debug, (int(px), int(py)), 3, (0, 240, 0), -1)

        H = self._find_homography(
            np.array(grid_px, dtype=np.float32),
            np.array(grid_mm, dtype=np.float32),
        )
        if H is None:
            result.note      = "호모그래피 계산 실패"
            result.debug_img = debug
            return result

        src_px = np.array(grid_px, dtype=np.float32)
        dst_mm = np.array(grid_mm, dtype=np.float32)

        result.H          = H
        result.ref_pts_px = src_px
        result.ref_pts_mm = dst_mm
        result.confidence = min(len(grid_px) / 25.0, 1.0)
        result.debug_img  = debug
        return result


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def _classify_lines(
    lines: np.ndarray,
    angle_tol: float,
) -> tuple[list, list]:
    """선을 수평(h)/수직(v)으로 분류."""
    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < angle_tol or angle > (180.0 - angle_tol):
            h_lines.append((x1, y1, x2, y2))
        elif (90.0 - angle_tol) < angle < (90.0 + angle_tol):
            v_lines.append((x1, y1, x2, y2))
    return h_lines, v_lines


def _line_intersection(
    l1: tuple[int,int,int,int],
    l2: tuple[int,int,int,int],
) -> tuple[float, float] | None:
    """두 직선(무한 연장)의 교점.  평행이면 None."""
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2
    d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(d) < 1e-8:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / d
    return (x1 + t*(x2-x1), y1 + t*(y2-y1))


def _compute_intersections(
    h_lines: list,
    v_lines: list,
    img_w: int,
    img_h: int,
) -> list[tuple[float, float]]:
    """모든 h×v 교차점 중 이미지 내부에 있는 것만 반환."""
    pts = []
    for hl in h_lines:
        for vl in v_lines:
            pt = _line_intersection(hl, vl)
            if pt and 0 <= pt[0] < img_w and 0 <= pt[1] < img_h:
                pts.append(pt)
    return pts


def _merge_close(values: list[float], threshold: float) -> list[float]:
    """값들을 정렬 후 threshold 이내의 값들을 평균으로 군집화."""
    if not values:
        return []
    groups = []
    for v in sorted(values):
        if groups and abs(v - groups[-1]) < threshold:
            # 이동 평균으로 군집 중심 갱신
            groups[-1] = (groups[-1] + v) / 2.0
        else:
            groups.append(float(v))
    return groups


def _build_grid_correspondences(
    x_groups: list[float],
    y_groups: list[float],
    img_w: int,
    img_h: int,
) -> tuple[list, list]:
    """
    픽셀 군집 위치 → GRID_SPACING_MM 간격 격자 mm 좌표 대응.

    전략:
    1. 픽셀 간격(px/step) 으로 mm/px 스케일 추정
    2. 가장 작은 x_group → GRID_ORIGIN_MM[0] 에 대응
    3. A4 범위(0~210, 0~297) 안에 드는 격자점만 사용
    """
    grid_px, grid_mm = [], []

    if len(x_groups) < 2 or len(y_groups) < 2:
        return grid_px, grid_mm

    # 픽셀 간격 추정 (중앙값)
    dx_px = float(np.median(np.diff(x_groups)))
    dy_px = float(np.median(np.diff(y_groups)))

    if dx_px <= 0 or dy_px <= 0:
        return grid_px, grid_mm

    # 스케일: px → mm
    scale_x = GRID_SPACING_MM / dx_px
    scale_y = GRID_SPACING_MM / dy_px

    # 원점(mm): x_groups[0] 에 대응하는 A4 mm
    # 격자가 GRID_ORIGIN_MM 에서 시작한다고 가정
    x0_mm = GRID_ORIGIN_MM[0]
    y0_mm = GRID_ORIGIN_MM[1]

    for xi, xg in enumerate(x_groups):
        xmm = x0_mm + xi * GRID_SPACING_MM
        for yi, yg in enumerate(y_groups):
            ymm = y0_mm + yi * GRID_SPACING_MM
            if 0.0 <= xmm <= A4_W_MM and 0.0 <= ymm <= A4_H_MM:
                grid_px.append((xg, yg))
                grid_mm.append((xmm, ymm))

    return grid_px, grid_mm
