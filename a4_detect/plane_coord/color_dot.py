"""
color_dot.py — Method 2: 색상점(컬러 원형 마커) 기반 A4 탐지

시트에 프린트된 4개 색상 원(빨강/초록/파랑/노랑)을
HSV 색공간에서 검출하여 호모그래피 계산.

장점: 구현 간단, 색상이 뚜렷하면 빠르게 동작
단점: 조명 변화(색온도, 그림자)에 민감
     비슷한 색상이 배경에 있으면 오탐
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import BaseA4Detector, DetectResult

# ── 시트에 프린트된 색상 원 위치 (A4 mm) ────────────────────────────────────
# sheet_color_dot.png 및 sheet_composite.png 와 동일한 값이어야 함
COLOR_POSITIONS_MM: dict[str, tuple[float, float]] = {
    "red":    ( 25.0,  25.0),   # TL
    "green":  (185.0,  25.0),   # TR
    "blue":   ( 25.0, 272.0),   # BL
    "yellow": (185.0, 272.0),   # BR
}

# ── HSV 탐지 범위 ─────────────────────────────────────────────────────────────
# 형식: [(lo1, hi1), (lo2, hi2)]  — 빨강은 Hue 가 0과 180 주변 2구간
_HSV_RANGES: dict[str, list[tuple[tuple[int,int,int], tuple[int,int,int]]]] = {
    "red": [
        ((  0, 110, 80), ( 10, 255, 255)),
        ((165, 110, 80), (180, 255, 255)),
    ],
    "green": [
        (( 38,  70, 50), ( 85, 255, 255)),
    ],
    "blue": [
        ((100,  80, 50), (140, 255, 255)),
    ],
    "yellow": [
        (( 18, 100, 80), ( 38, 255, 255)),
    ],
}

# 시각화용 BGR 색
_DEBUG_BGR: dict[str, tuple[int, int, int]] = {
    "red":    (  0,  0, 200),
    "green":  (  0, 180,  0),
    "blue":   (200,  0,   0),
    "yellow": (  0, 200, 200),
}

# 색 처리 순서: TL TR BL BR 대응
_COLOR_ORDER = ["red", "green", "blue", "yellow"]


class ColorDotDetector(BaseA4Detector):
    """
    색상 마커(컬러 원) 기반 A4 평면 탐지.

    Parameters
    ----------
    min_blob_area   : 유효 블롭 최소 면적 (px²)
    max_blob_area   : 유효 블롭 최대 면적 (px²)
    morph_ksize     : 모폴로지 열림 커널 크기
    require_all     : True → 4개 모두 감지해야 OK. False → 4개 이상이면 OK
    """

    def __init__(
        self,
        min_blob_area: int   = 20,
        max_blob_area: int   = 5000,
        morph_ksize: int     = 5,
        require_all: bool    = True,
    ):
        super().__init__("color_dot")
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.morph_ksize   = morph_ksize
        self.require_all   = require_all

    def detect(self, frame: np.ndarray) -> DetectResult:
        result = DetectResult()
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        debug  = frame.copy()

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize)
        )

        found_px: dict[str, tuple[float, float]] = {}

        for color in _COLOR_ORDER:
            mask = self._build_mask(hsv, color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_cnt  = None
            best_area = 0.0
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if self.min_blob_area < area < self.max_blob_area and area > best_area:
                    best_area = area
                    best_cnt  = cnt

            if best_cnt is None:
                continue

            M = cv2.moments(best_cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                found_px[color] = (cx, cy)

                # 시각화
                bgr = _DEBUG_BGR[color]
                cv2.circle(debug, (int(cx), int(cy)), 12, bgr, 2)
                cv2.drawMarker(debug, (int(cx), int(cy)), bgr,
                               cv2.MARKER_CROSS, 18, 2)
                xm, ym = COLOR_POSITIONS_MM[color]
                cv2.putText(debug, f"{color} ({xm:.0f},{ym:.0f})",
                            (int(cx) + 14, int(cy) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, bgr, 1)

        # ── 결과 평가 ────────────────────────────────────────────────────────
        n_found = len(found_px)
        if self.require_all and n_found < 4:
            result.note      = f"색상점 {n_found}/4 감지됨  ({list(found_px.keys())})"
            result.debug_img = debug
            return result
        if n_found < 4:
            result.note      = f"색상점 {n_found}/4 (require_all=False 이지만 부족)"
            result.debug_img = debug
            return result

        src_px = np.array([found_px[c]                     for c in _COLOR_ORDER], dtype=np.float32)
        dst_mm = np.array([COLOR_POSITIONS_MM[c]            for c in _COLOR_ORDER], dtype=np.float32)

        H = self._find_homography(src_px, dst_mm)
        if H is None:
            result.note      = "호모그래피 계산 실패"
            result.debug_img = debug
            return result

        result.H          = H
        result.corners_px = src_px
        result.ref_pts_px = src_px
        result.ref_pts_mm = dst_mm
        result.confidence = n_found / 4.0
        result.debug_img  = debug
        return result

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────
    def _build_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        """해당 색상의 HSV 마스크 생성 (빨강은 2구간 OR)."""
        ranges = _HSV_RANGES[color]
        mask   = cv2.inRange(hsv, np.array(ranges[0][0]), np.array(ranges[0][1]))
        for lo, hi in ranges[1:]:
            mask = cv2.bitwise_or(
                mask, cv2.inRange(hsv, np.array(lo), np.array(hi))
            )
        return mask
