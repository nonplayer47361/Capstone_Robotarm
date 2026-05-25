"""
base.py — A4 평면좌표계 탐지기 공통 인터페이스

모든 탐지 방법은 BaseA4Detector 를 상속하고
detect(frame) → DetectResult 를 구현합니다.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

# ── A4 상수 ──────────────────────────────────────────────────────────────────
A4_W_MM = 210.0
A4_H_MM = 297.0

# A4 용지 4 코너 (A4 좌표, mm):  TL TR BL BR
CORNERS_MM = np.array([
    [  0.0,   0.0],   # TL
    [210.0,   0.0],   # TR
    [  0.0, 297.0],   # BL
    [210.0, 297.0],   # BR
], dtype=np.float32)


# ── 탐지 결과 ─────────────────────────────────────────────────────────────────
@dataclass
class DetectResult:
    """
    단일 프레임에 대한 A4 탐지 결과.

    H            : 픽셀 → A4 mm 변환 호모그래피 (3×3).  None 이면 탐지 실패
    corners_px   : 대표 코너 픽셀 좌표 배열 (N×2).  호모그래피 계산에 사용된 점들
    ref_pts_px   : 모든 대응점 픽셀 (더 많을수록 reprojection error 신뢰도 ↑)
    ref_pts_mm   : ref_pts_px 에 대응되는 A4 mm 좌표
    confidence   : 탐지 신뢰도 0~1 (방법별 주관적 점수)
    method_name  : 탐지 방법 이름
    elapsed_ms   : 처리 시간 (ms)
    repro_err_mm : 재투영 오차 평균 (mm)
    debug_img    : 시각화 이미지 (BGR)
    note         : 실패/경고 메시지
    """
    H            : np.ndarray | None     = None
    corners_px   : np.ndarray | None     = None
    ref_pts_px   : np.ndarray | None     = None
    ref_pts_mm   : np.ndarray | None     = None
    confidence   : float                 = 0.0
    method_name  : str                   = ""
    elapsed_ms   : float                 = 0.0
    repro_err_mm : float                 = float("inf")
    debug_img    : np.ndarray | None     = None
    note         : str                   = ""
    # 지연 계산 — H 역행렬 캐시 (mm_to_px 최초 호출 시 생성)
    _H_inv: np.ndarray | None = field(default=None, repr=False, compare=False)

    # ── 속성 ─────────────────────────────────────────────────────────────────
    @property
    def ok(self) -> bool:
        """호모그래피가 유효하면 True."""
        return self.H is not None

    # ── 좌표 변환 ─────────────────────────────────────────────────────────────
    def px_to_mm(self, px_x: float, px_y: float) -> tuple[float, float]:
        """픽셀 (px_x, px_y) → A4 mm 좌표."""
        if self.H is None:
            return float("nan"), float("nan")
        pt  = np.array([[[float(px_x), float(px_y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0][0][0]), float(out[0][0][1])

    def mm_to_px(self, mm_x: float, mm_y: float) -> tuple[float, float]:
        """A4 mm → 픽셀 좌표 (H 역행렬 사용, 역행렬은 첫 호출 시 캐싱)."""
        if self.H is None:
            return float("nan"), float("nan")
        if self._H_inv is None:
            self._H_inv = np.linalg.inv(self.H)
        pt  = np.array([[[float(mm_x), float(mm_y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H_inv)
        return float(out[0][0][0]), float(out[0][0][1])

    # ── 재투영 오차 계산 ───────────────────────────────────────────────────────
    def calc_reprojection_error(
        self,
        src_px: np.ndarray,
        dst_mm: np.ndarray,
    ) -> float:
        """src_px → H → 예측 mm 와 dst_mm 사이의 평균 오차(mm)."""
        if self.H is None:
            return float("inf")
        pts  = src_px.reshape(-1, 1, 2).astype(np.float32)
        pred = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        errs = np.hypot(pred[:, 0] - dst_mm[:, 0], pred[:, 1] - dst_mm[:, 1])
        return float(errs.mean())


# ── 탐지기 기반 클래스 ─────────────────────────────────────────────────────────
class BaseA4Detector(ABC):
    """모든 A4 탐지 방법의 추상 기반 클래스."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectResult:
        """
        BGR 이미지 프레임을 받아 DetectResult 반환.
        성공 시 result.H 에 픽셀→mm 호모그래피 포함.
        """
        ...

    def timed_detect(self, frame: np.ndarray) -> DetectResult:
        """detect() 실행 후 elapsed_ms / method_name 자동 기록."""
        t0     = time.perf_counter()
        result = self.detect(frame)
        result.elapsed_ms  = (time.perf_counter() - t0) * 1000.0
        result.method_name = self.name

        # 재투영 오차 계산
        if result.ok and result.ref_pts_px is not None and result.ref_pts_mm is not None:
            result.repro_err_mm = result.calc_reprojection_error(
                result.ref_pts_px, result.ref_pts_mm
            )
        return result

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _find_homography(
        src_px: np.ndarray,
        dst_mm: np.ndarray,
        method: int = cv2.RANSAC,
        ransac_thresh: float = 5.0,
    ) -> np.ndarray | None:
        """최소 4점으로 호모그래피 계산.  실패 시 None."""
        if len(src_px) < 4:
            return None
        H, _ = cv2.findHomography(
            src_px.astype(np.float32),
            dst_mm.astype(np.float32),
            method,
            ransac_thresh,
        )
        return H  # findHomography 실패 시 None 반환

    @staticmethod
    def _order_quad(pts: np.ndarray) -> np.ndarray:
        """
        4개 점을 TL, TR, BL, BR 순서로 정렬.
        좌표합(x+y)이 최소 = TL, 최대 = BR
        좌표차(x-y)가 최대 = TR, 최소 = BL
        """
        pts = pts.reshape(4, 2)
        s   = pts.sum(axis=1)
        d   = np.diff(pts, axis=1).ravel()
        return np.array([
            pts[np.argmin(s)],   # TL
            pts[np.argmax(d)],   # TR
            pts[np.argmin(d)],   # BL
            pts[np.argmax(s)],   # BR
        ], dtype=np.float32)
