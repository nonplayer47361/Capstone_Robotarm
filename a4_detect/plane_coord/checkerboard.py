"""
checkerboard.py — Method 4: 체커보드 패턴 기반 A4 탐지

시트에 프린트된 체커보드 패턴의 내부 코너를 서브픽셀 정밀도로 검출하여
알려진 실제 좌표와 매칭 → 호모그래피 계산.

장점: 서브픽셀 정밀도로 가장 높은 정확도 기대
     OpenCV 내장 알고리즘 사용, 코너 수가 많아 과결정(overdetermined) 문제
단점: 패턴 전체가 카메라에 보여야 함 (부분 가림 시 실패)
     패턴 크기/위치 파라미터가 시트와 정확히 일치해야 함
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import BaseA4Detector, DetectResult

# ── 체커보드 파라미터 (시트와 반드시 일치) ────────────────────────────────────
# 내부 코너 수 (열 × 행):  전체 칸 수 = (COLS+1) × (ROWS+1)
CHESSBOARD_COLS = 7     # 내부 X 코너 수
CHESSBOARD_ROWS = 5     # 내부 Y 코너 수
SQUARE_MM       = 20.0  # 한 칸 크기 (mm)

# 내부 TL 코너(0번)의 A4 좌표 (mm)
# = 체커보드 시작 위치 (첫 번째 내부 코너)
CB_ORIGIN_MM: tuple[float, float] = (35.0, 65.0)

_PATTERN = (CHESSBOARD_COLS, CHESSBOARD_ROWS)


def _build_object_points() -> np.ndarray:
    """체커보드 내부 코너의 A4 mm 좌표 배열 (ROWS*COLS, 2)."""
    pts = []
    for r in range(CHESSBOARD_ROWS):
        for c in range(CHESSBOARD_COLS):
            pts.append([
                CB_ORIGIN_MM[0] + c * SQUARE_MM,
                CB_ORIGIN_MM[1] + r * SQUARE_MM,
            ])
    return np.array(pts, dtype=np.float32)


_OBJ_PTS = _build_object_points()


class CheckerboardDetector(BaseA4Detector):
    """
    체커보드 패턴 기반 A4 탐지.

    Parameters
    ----------
    use_subpix : 서브픽셀 정밀화 사용 여부
    fast_check : CALIB_CB_FAST_CHECK 플래그 (속도 향상, 간헐적 미검출)
    """

    def __init__(
        self,
        use_subpix: bool  = True,
        fast_check: bool  = True,
    ):
        super().__init__("checkerboard")
        self.use_subpix = use_subpix

        self._flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | (cv2.CALIB_CB_FAST_CHECK if fast_check else 0)
        )
        self._subpix_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001,
        )

    def detect(self, frame: np.ndarray) -> DetectResult:
        result = DetectResult()
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug  = frame.copy()

        found, corners = cv2.findChessboardCorners(gray, _PATTERN, self._flags)

        if not found:
            result.note      = f"체커보드 패턴 미검출 ({CHESSBOARD_COLS}×{CHESSBOARD_ROWS})"
            result.debug_img = debug
            return result

        # ── 서브픽셀 정밀화 ────────────────────────────────────────────────
        if self.use_subpix:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self._subpix_criteria
            )

        cv2.drawChessboardCorners(debug, _PATTERN, corners, found)

        src_px = corners.reshape(-1, 2)   # (ROWS*COLS, 2)
        dst_mm = _OBJ_PTS                  # (ROWS*COLS, 2)

        H = self._find_homography(src_px, dst_mm, method=0)  # 최소제곱법 (모두 정확한 점)
        if H is None:
            result.note      = "호모그래피 계산 실패"
            result.debug_img = debug
            return result

        # 대표 4 코너: 첫 번째, 마지막, 첫 행 끝, 마지막 행 시작
        n_pts = len(src_px)
        result.H          = H
        result.corners_px = src_px[[0, CHESSBOARD_COLS - 1,
                                     n_pts - CHESSBOARD_COLS, n_pts - 1]]
        result.ref_pts_px = src_px
        result.ref_pts_mm = dst_mm
        result.confidence = 1.0          # 패턴이 검출되면 신뢰도 최대
        result.debug_img  = debug
        return result
