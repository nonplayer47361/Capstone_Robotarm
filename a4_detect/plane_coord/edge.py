"""
edge.py — Method 1: 외곽선(엣지) 기반 A4 탐지

전처리(그레이→블러→캐니) → 컨투어 추출 → 가장 큰 사각형 → 4 코너 → 호모그래피.
인쇄된 보조 테두리가 아니라 실제 A4 용지의 물리적 외곽선을 기준으로 삼습니다.

장점: 마커 불필요, 흰 A4 용지만 있으면 동작
단점: 배경이 복잡하거나 조명이 고르지 않으면 실패
     부분 가림(occlusion) 시 취약
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import BaseA4Detector, DetectResult, CORNERS_MM


class EdgeDetector(BaseA4Detector):
    """
    Canny 에지 + 컨투어 기반 A4 외곽선 탐지.

    Parameters
    ----------
    canny_lo        : Canny 하한 임계값
    canny_hi        : Canny 상한 임계값
    blur_ksize      : 가우시안 블러 커널 크기 (홀수)
    min_area_ratio  : 프레임 대비 최소 사각형 면적 비율 (0~1)
    poly_eps_ratio  : approxPolyDP epsilon = 둘레 × eps_ratio
    """

    def __init__(
        self,
        canny_lo: int       = 30,
        canny_hi: int       = 120,
        blur_ksize: int     = 5,
        min_area_ratio: float = 0.15,
        poly_eps_ratio: float = 0.02,
        corner_margin_px: int = 8,
    ):
        super().__init__("edge")
        self.canny_lo       = canny_lo
        self.canny_hi       = canny_hi
        self.blur_ksize     = blur_ksize
        self.min_area_ratio = min_area_ratio
        self.poly_eps_ratio = poly_eps_ratio
        self.corner_margin_px = corner_margin_px

    def detect(self, frame: np.ndarray) -> DetectResult:
        result = DetectResult()
        h, w   = frame.shape[:2]
        min_area = h * w * self.min_area_ratio

        # ── 전처리 ──────────────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        edges = cv2.Canny(blur, self.canny_lo, self.canny_hi)

        # 끊긴 에지 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges  = cv2.dilate(edges, kernel, iterations=1)

        # ── 컨투어 탐지 ─────────────────────────────────────────────────────
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug = frame.copy()
        cv2.drawContours(debug, cnts, -1, (80, 80, 80), 1)

        best_quad = None
        best_area = 0.0

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.poly_eps_ratio * peri, True)
            if len(approx) == 4 and area > best_area:
                best_area = area
                best_quad = approx

        if best_quad is None:
            result.note      = f"사각형 컨투어 없음 (min_area={min_area:.0f}px²)"
            result.debug_img = debug
            return result

        # ── 코너 정렬: TL TR BL BR ──────────────────────────────────────────
        pts = self._order_quad(best_quad.reshape(4, 2).astype(np.float32))

        margin = float(self.corner_margin_px)
        inside = (
            (pts[:, 0] >= margin).all()
            and (pts[:, 0] <= (w - 1 - margin)).all()
            and (pts[:, 1] >= margin).all()
            and (pts[:, 1] <= (h - 1 - margin)).all()
        )
        if not inside:
            cv2.polylines(debug, [pts.astype(int)], True, (0, 140, 255), 2)
            result.note = "A4 corner too close to frame border"
            result.debug_img = debug
            return result

        cv2.polylines(debug, [pts.astype(int)], True, (0, 240, 60), 3)
        for i, (x, y) in enumerate(pts):
            cv2.circle(debug, (int(x), int(y)), 7, (0, 0, 220), -1)
            cv2.putText(debug, ["TL","TR","BL","BR"][i],
                        (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1)

        # ── 호모그래피 ───────────────────────────────────────────────────────
        H = self._find_homography(pts, CORNERS_MM)
        if H is None:
            result.note      = "호모그래피 계산 실패"
            result.debug_img = debug
            return result

        result.H          = H
        result.corners_px = pts
        result.ref_pts_px = pts
        result.ref_pts_mm = CORNERS_MM.copy()
        result.confidence = min(best_area / (h * w), 1.0)
        result.debug_img  = debug
        return result
