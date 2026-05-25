"""
plane_coord — A4 평면좌표계 탐지 방법 모음

각 방법은 BaseA4Detector 를 상속하며 detect(frame) → DetectResult 를 구현합니다.

탐지 방법:
  edge         : A4 외곽선(Canny 에지) 기반
  color_dot    : 색상 마커(컬러 원) 기반
  aruco        : ArUco 마커 기반 (DICT_4X4_50)
  grid         : 그리드 라인 기반 (Hough 변환)
  composite    : 복합 (위 방법 조합)

빠른 사용 예:
    from plane_coord import METHODS
    detector = METHODS["aruco"]()
    result   = detector.timed_detect(frame)
    if result.ok:
        x_mm, y_mm = result.px_to_mm(cx_px, cy_px)
"""
from .base         import BaseA4Detector, DetectResult, A4_W_MM, A4_H_MM, CORNERS_MM
from .edge         import EdgeDetector
from .color_dot    import ColorDotDetector, COLOR_POSITIONS_MM
from .aruco        import ArucoDetector, ARUCO_CENTER_MM, ARUCO_DICT_ID
from .grid         import GridDetector, GRID_SPACING_MM
from .composite    import CompositeDetector
from .camera_calib import CameraCalib, maybe_undistort

# 이름으로 탐지기를 생성하는 레지스트리
METHODS: dict[str, type[BaseA4Detector]] = {
    "edge":         EdgeDetector,
    "color_dot":    ColorDotDetector,
    "aruco":        ArucoDetector,
    "grid":         GridDetector,
    "composite":    CompositeDetector,
}

__all__ = [
    # 기반 클래스
    "BaseA4Detector", "DetectResult",
    "A4_W_MM", "A4_H_MM", "CORNERS_MM",
    # 탐지기
    "EdgeDetector",
    "ColorDotDetector",
    "ArucoDetector",
    "GridDetector",
    "CompositeDetector",
    # 탐지기 레지스트리
    "METHODS",
    # 방법별 설정 상수 (시트 생성기와 공유)
    "COLOR_POSITIONS_MM",
    "ARUCO_CENTER_MM",
    "ARUCO_DICT_ID",
    "GRID_SPACING_MM",
    # 카메라 캘리브레이션
    "CameraCalib",
    "maybe_undistort",
]
