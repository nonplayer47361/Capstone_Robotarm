"""
camera_calib.py — 카메라 렌즈 왜곡 보정 인프라

팀원마다 카메라가 달라도 동일한 방법으로 캘리브레이션 파라미터를 저장·불러와
A4 검출 전에 프레임에 적용할 수 있도록 합니다.

주요 특징:
  - JSON 파일 1개로 카메라별 파라미터 저장 (이식성)
  - remap 맵 사전계산 → 매 프레임 undistort 속도 최적화
  - alpha=0 크롭 방식: 왜곡 보정 후 black border 없이 출력
  - CameraCalib.load() / .save() 로 파일 I/O

사용 예:
    from plane_coord.camera_calib import CameraCalib

    calib = CameraCalib.load("calib_camera0.json")
    undistorted_frame = calib.undistort(frame)
    # 이후 undistorted_frame 을 A4 검출기에 전달
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraCalib:
    """
    단일 카메라의 내부 파라미터 + 왜곡 계수.

    Attributes
    ----------
    camera_matrix : (3,3) 내부 파라미터 행렬  [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    dist_coeffs   : (1,N) 왜곡 계수  [k1, k2, p1, p2, k3, ...]
    image_size    : (width, height) 캘리브레이션에 사용된 이미지 크기
    rms_px        : 캘리브레이션 RMS 재투영 오차 (px)  — 낮을수록 좋음, 목표 < 1.0

    Notes
    -----
    - undistort() 첫 호출 시 remap 맵을 자동으로 생성합니다 (지연 초기화).
    - 이미지 크기가 달라지면 remap 맵을 다시 생성해야 합니다.
      그런 경우 reset_maps() 를 호출하세요.
    """
    camera_matrix: np.ndarray           # (3,3)
    dist_coeffs:   np.ndarray           # (1,N)
    image_size:    tuple[int, int]      # (width, height)
    rms_px:        float = 0.0

    # 지연 초기화 — JSON/dataclass 직렬화에서 제외
    _map1: np.ndarray | None = field(default=None, repr=False, compare=False)
    _map2: np.ndarray | None = field(default=None, repr=False, compare=False)

    # ── remap 맵 ─────────────────────────────────────────────────────────────

    def _build_maps(self) -> None:
        """remap 맵을 (처음 한 번) 사전 계산합니다."""
        w, h = self.image_size
        # alpha=0: 유효 픽셀만 남도록 크롭 (black border 없음)
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=0
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs,
            None, new_K,
            (w, h), cv2.CV_16SC2,
        )

    def reset_maps(self) -> None:
        """이미지 크기가 바뀌었을 때 remap 맵을 초기화합니다."""
        self._map1 = self._map2 = None

    # ── 왜곡 보정 ─────────────────────────────────────────────────────────────

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임에 렌즈 왜곡 보정을 적용하고 보정된 프레임을 반환합니다.

        remap 방식 (initUndistortRectifyMap + remap) 을 사용해
        매 프레임 호출해도 빠릅니다.
        """
        h, w = frame.shape[:2]
        if (w, h) != self.image_size:
            # 이미지 크기가 캘리브레이션 당시와 다른 경우 재계산
            self.image_size = (w, h)
            self.reset_maps()

        if self._map1 is None:
            self._build_maps()

        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

    # ── 파일 I/O ──────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """JSON 파일로 저장합니다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs":   self.dist_coeffs.tolist(),
            "image_size":    list(self.image_size),
            "rms_px":        float(self.rms_px),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[calib] 저장 → {path}  (rms={self.rms_px:.4f} px)")

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalib":
        """JSON 파일에서 불러옵니다."""
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"캘리브레이션 파일 없음: {path}") from None
        return cls(
            camera_matrix = np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs   = np.array(data["dist_coeffs"],   dtype=np.float64),
            image_size    = tuple(data["image_size"]),
            rms_px        = float(data.get("rms_px", 0.0)),
        )

    # ── 요약 ─────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        K = self.camera_matrix
        d = self.dist_coeffs.ravel()
        w, h = self.image_size
        return (
            f"CameraCalib  {w}×{h}  rms={self.rms_px:.4f}px\n"
            f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
            f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}\n"
            f"  dist=[{', '.join(f'{v:.5f}' for v in d)}]"
        )


def maybe_undistort(frame: np.ndarray, calib: CameraCalib | None) -> np.ndarray:
    """calib 가 None 이 아닌 경우에만 왜곡 보정을 적용합니다.

    프레임 루프 내 반복되는 ``if calib is not None: frame = calib.undistort(frame)``
    패턴을 한 줄로 줄이기 위한 헬퍼입니다.
    """
    return calib.undistort(frame) if calib is not None else frame
