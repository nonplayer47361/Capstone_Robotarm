"""
aruco.py — Method 3: ArUco 마커 기반 A4 탐지

시트 4 코너에 프린트된 ArUco 마커(DICT_4X4_50, ID 0~3)를 검출하여
각 마커 중심의 픽셀 좌표 → 알려진 A4 mm 좌표 → 호모그래피 계산.

장점: OpenCV 내장, 조명 변화에 강함, 서브픽셀 정밀도
단점: 마커 크기가 작거나 기울기가 심하면 미검출
     직접 프린트 필요
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import BaseA4Detector, DetectResult

# ── ArUco 마커 설정 ───────────────────────────────────────────────────────────
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# 마커 ID → A4 상 중심 위치 (mm)
# sheet_aruco.png / sheet_composite.png 와 동일
ARUCO_CENTER_MM: dict[int, tuple[float, float]] = {
    0: ( 25.0,  25.0),   # TL
    1: (185.0,  25.0),   # TR
    2: ( 25.0, 272.0),   # BL
    3: (185.0, 272.0),   # BR
}

# 처리 순서: ARUCO_CENTER_MM 삽입 순서(TL TR BL BR)와 항상 동기화
_ID_ORDER = list(ARUCO_CENTER_MM)


class ArucoDetector(BaseA4Detector):
    """
    ArUco 마커 기반 A4 탐지.

    Parameters
    ----------
    min_markers   : 호모그래피를 계산하기 위한 최소 마커 수 (기본 4, 최소 4)
    use_corners   : True → 마커의 4 꼭짓점 전체를 대응점으로 사용 (정밀도 ↑)
                    False → 마커 중심만 사용 (속도 ↑)
    marker_size_mm: 출력물에 프린트된 ArUco 마커 한 변 길이 (mm).
                    gen.py 의 ARUCO_SIZE_VARIANTS_MM = [16, 20, 24] 중 하나.
                    기본값 20.0 — 표준 sheet_aruco.png / sheet_composite.png 에 해당.
                    ⚠️  출력 시 사용한 크기와 반드시 일치해야 합니다.
    """

    def __init__(
        self,
        min_markers: int   = 4,
        use_corners: bool  = True,
        marker_size_mm: float = 20.0,
    ):
        super().__init__("aruco")
        self.min_markers    = max(min_markers, 4)
        self.use_corners    = use_corners
        self.marker_size_mm = marker_size_mm

        aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        params       = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def detect(self, frame: np.ndarray) -> DetectResult:
        result = DetectResult()
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug  = frame.copy()

        corners_list, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            result.note      = "ArUco 마커 없음"
            result.debug_img = debug
            return result

        cv2.aruco.drawDetectedMarkers(debug, corners_list, ids)

        # ── 마커별 중심 / 코너 수집 ────────────────────────────────────────
        found_center: dict[int, np.ndarray]  = {}
        found_corners: dict[int, np.ndarray] = {}

        for corners, id_arr in zip(corners_list, ids):
            mid = int(id_arr[0])
            if mid in ARUCO_CENTER_MM:
                c4 = corners[0]             # shape (4, 2) — 마커의 4 꼭짓점
                found_center[mid]  = c4.mean(axis=0)
                found_corners[mid] = c4     # TL TR BR BL (ArUco 순서)

        n_found = len(found_center)
        if n_found < self.min_markers:
            result.note      = f"ArUco {n_found}/{self.min_markers} 감지됨  ({list(found_center.keys())})"
            result.debug_img = debug
            return result

        # ── 호모그래피용 대응점 구성 ────────────────────────────────────────
        if self.use_corners and n_found == 4:
            # 각 마커의 4 꼭짓점 → 더 많은 대응점 (최대 16개)
            src_pts, dst_pts = _corners_correspondence(
                found_corners, self.marker_size_mm
            )
        else:
            # 마커 중심만 사용 (4개)
            src_pts = np.array([found_center[i] for i in _ID_ORDER if i in found_center],
                                dtype=np.float32)
            dst_pts = np.array([ARUCO_CENTER_MM[i] for i in _ID_ORDER if i in found_center],
                                dtype=np.float32)

        H = self._find_homography(src_pts, dst_pts)
        if H is None:
            result.note      = "호모그래피 계산 실패"
            result.debug_img = debug
            return result

        # 대표 코너 (4개 마커 중심)
        corners_px = np.array(
            [found_center[i] for i in _ID_ORDER if i in found_center],
            dtype=np.float32
        )

        result.H          = H
        result.corners_px = corners_px
        result.ref_pts_px = src_pts
        result.ref_pts_mm = dst_pts
        result.confidence = n_found / 4.0
        result.debug_img  = debug
        return result


# ── 마커 꼭짓점 대응 구성 ─────────────────────────────────────────────────────
# ArUco 마커 꼭짓점 순서: TL TR BR BL (반시계)
# _MARKER_CORNER_OFFSETS_MM : 마커 크기 1.0mm 기준 단위 오프셋.
#   실제 사용 시 marker_size_mm 을 곱해서 쓴다.
#   gen.py ARUCO_SIZE_VARIANTS_MM = [16, 20, 24] 중 하나와 일치해야 함.
_MARKER_CORNER_OFFSETS_UNIT = np.array([
    [-0.5, -0.5],  # TL
    [ 0.5, -0.5],  # TR
    [ 0.5,  0.5],  # BR
    [-0.5,  0.5],  # BL
], dtype=np.float32)


def _corners_correspondence(
    found_corners: dict[int, np.ndarray],
    marker_size_mm: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    4개 마커의 4 꼭짓점 → 픽셀/mm 대응점 배열 반환. (최대 16대응점)

    Parameters
    ----------
    found_corners  : {marker_id: (4,2) px corners}
    marker_size_mm : 출력물 마커 한 변 길이 (mm). ArucoDetector.marker_size_mm 과 일치.
    """
    offsets = _MARKER_CORNER_OFFSETS_UNIT * marker_size_mm
    src_list, dst_list = [], []
    for mid in _ID_ORDER:
        if mid not in found_corners:
            continue
        cx_mm, cy_mm = ARUCO_CENTER_MM[mid]
        c4_px = found_corners[mid]         # (4, 2) px
        for k in range(4):
            src_list.append(c4_px[k])
            dst_list.append([cx_mm + offsets[k, 0], cy_mm + offsets[k, 1]])

    return (
        np.array(src_list, dtype=np.float32),
        np.array(dst_list, dtype=np.float32),
    )
