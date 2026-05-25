"""
composite.py — Method 6: 복합(Composite) A4 탐지

여러 탐지 방법을 조합해 가장 신뢰할 수 있는 결과를 선택합니다.

Mode A (priority_fallback):
  우선순위 순서대로 시도 → 첫 번째 성공한 결과 반환.
  빠르고 단순하지만 여러 결과를 비교하지 않음.

Mode B (best_of_all):
  모든 방법을 실행 → 재투영 오차가 가장 낮은 결과 반환.
  가장 정확하지만 처리 시간이 가장 많이 걸림.

Mode C (vote_H):
  여러 방법에서 계산된 H를 앙상블:
  - 각 성공한 H 로 코너를 변환
  - 변환 결과의 가중 평균으로 최종 H 계산 (검증 포인트 기준 오차 최소화)
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import A4_W_MM, A4_H_MM, CORNERS_MM, BaseA4Detector, DetectResult
from .aruco        import ArucoDetector
from .color_dot    import ColorDotDetector
from .edge         import EdgeDetector
from .grid         import GridDetector

# 기본 우선순위: 정확도 높은 순
DEFAULT_PRIORITY = ["aruco", "color_dot", "edge", "grid"]

# vote_H 앙상블 기준점: A4 코너 4개 + 중앙점 (매 프레임 재생성 방지)
_VOTE_REF_MM: np.ndarray = np.vstack(
    [CORNERS_MM, [[105.0, 148.5]]]
).astype(np.float32)

# 각 방법의 신뢰도 가중치 (앙상블 시 사용)
_METHOD_WEIGHTS: dict[str, float] = {
    "aruco":        1.0,
    "color_dot":    0.80,
    "edge":         0.60,
    "grid":         0.55,
}

_ALL_DETECTORS: dict[str, type[BaseA4Detector]] = {
    "aruco":        ArucoDetector,
    "color_dot":    ColorDotDetector,
    "edge":         EdgeDetector,
    "grid":         GridDetector,
}


class CompositeDetector(BaseA4Detector):
    """
    여러 탐지 방법을 조합한 복합 탐지기.

    Parameters
    ----------
    methods    : 사용할 방법 이름 리스트 (순서 = 우선순위)
    mode       : 'priority_fallback' | 'best_of_all' | 'vote_H'
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        mode: str = "priority_fallback",
    ):
        super().__init__("composite")
        if methods is None:
            methods = DEFAULT_PRIORITY
        self.mode = mode

        # 요청된 방법 중 구현된 것만 초기화
        self._detectors: list[tuple[str, BaseA4Detector]] = [
            (name, _ALL_DETECTORS[name]())
            for name in methods
            if name in _ALL_DETECTORS
        ]

    def detect(self, frame: np.ndarray) -> DetectResult:
        if self.mode == "best_of_all":
            return self._best_of_all(frame)
        if self.mode == "vote_H":
            return self._vote_H(frame)
        return self._priority_fallback(frame)   # 기본

    def detect_all(self, frame: np.ndarray) -> dict[str, DetectResult]:
        """모든 방법 실행 → {method_name: DetectResult} 반환 (비교 연구용)."""
        results = {}
        for name, det in self._detectors:
            r = det.detect(frame)
            r.method_name = name
            results[name] = r
        return results

    # ── Mode A ───────────────────────────────────────────────────────────────
    def _priority_fallback(self, frame: np.ndarray) -> DetectResult:
        """우선순위 순으로 시도 → 첫 성공 결과 반환."""
        debug = frame.copy()
        fail_notes = []

        for name, det in self._detectors:
            r = det.detect(frame)
            if r.ok:
                r.method_name = f"composite[{name}]"
                debug_img = r.debug_img if r.debug_img is not None else frame.copy()
                _put_method_label(debug_img, name, ok=True)
                r.debug_img = debug_img
                return r
            fail_notes.append(f"{name}:{r.note}")

        result = DetectResult(note=f"모든 방법 실패 — {' | '.join(fail_notes)}")
        result.debug_img = _draw_fail_overlay(debug, fail_notes)
        return result

    # ── Mode B ───────────────────────────────────────────────────────────────
    def _best_of_all(self, frame: np.ndarray) -> DetectResult:
        """모든 방법 실행 → 재투영 오차가 가장 낮은 결과 반환."""
        candidates: list[DetectResult] = []

        for name, det in self._detectors:
            r = det.detect(frame)
            r.method_name = name
            if r.ok and r.ref_pts_px is not None:
                r.repro_err_mm = r.calc_reprojection_error(r.ref_pts_px, r.ref_pts_mm)
                candidates.append(r)

        if not candidates:
            result = DetectResult(note="모든 방법 실패")
            result.debug_img = frame.copy()
            return result

        best = min(candidates, key=lambda x: x.repro_err_mm)
        best.method_name = f"composite_best[{best.method_name}]"
        debug_img = best.debug_img if best.debug_img is not None else frame.copy()
        _put_method_label(
            debug_img,
            f"{best.method_name} err={best.repro_err_mm:.2f}mm",
            ok=True,
        )
        best.debug_img = debug_img
        return best

    # ── Mode C ───────────────────────────────────────────────────────────────
    def _vote_H(self, frame: np.ndarray) -> DetectResult:
        """
        성공한 방법들의 H 를 앙상블:
        각 H 로 A4 코너 4점을 픽셀에서 mm로 변환 →
        가중 평균으로 합의된 mm 좌표 계산 →
        합의된 좌표에서 최종 H 재계산.
        """
        valid: list[tuple[float, DetectResult]] = []

        for name, det in self._detectors:
            r = det.detect(frame)
            r.method_name = name
            if r.ok and r.corners_px is not None:
                w = _METHOD_WEIGHTS.get(name, 0.5)
                valid.append((w, r))

        if not valid:
            result = DetectResult(note="앙상블 실패: 성공한 방법 없음")
            result.debug_img = frame.copy()
            return result

        if len(valid) == 1:
            r = valid[0][1]
            r.method_name = f"composite_vote[{r.method_name}]"
            return r

        # ── 각 방법에서 코너 픽셀을 mm로 변환 → 가중 평균 ──────────────────
        # 각 기준점을 역변환(mm→px)하여 픽셀 위치 수집
        ref_mm = _VOTE_REF_MM
        weighted_px: list[tuple[float, np.ndarray]] = []
        for w, r in valid:
            try:
                H_inv = np.linalg.inv(r.H)
                pts   = cv2.perspectiveTransform(
                    ref_mm.reshape(1, -1, 2), H_inv
                ).reshape(-1, 2)
                weighted_px.append((w, pts))
            except np.linalg.LinAlgError:
                continue

        if not weighted_px:
            result = DetectResult(note="앙상블: H 역변환 실패")
            result.debug_img = frame.copy()
            return result

        # 가중 평균 픽셀 좌표
        total_w     = sum(w for w, _ in weighted_px)
        avg_px      = sum(w * pts for w, pts in weighted_px) / total_w  # (5, 2)

        # 앙상블 픽셀 ↔ 알려진 mm 로 최종 H 계산
        H_ensemble = self._find_homography(avg_px, ref_mm)
        if H_ensemble is None:
            # 앙상블 실패 시 최고 가중치 방법 반환
            best = max(valid, key=lambda x: x[0])[1]
            best.method_name = f"composite_vote_fallback[{best.method_name}]"
            return best

        # ── 결과 구성 ──────────────────────────────────────────────────────
        result = DetectResult()
        result.H          = H_ensemble
        result.corners_px = avg_px[:4]
        result.confidence = min(sum(w for w, _ in weighted_px) / len(_ALL_DETECTORS), 1.0)
        result.method_name = (
            f"composite_vote[{'+'.join(r.method_name for _,r in valid)}]"
        )
        # 디버그 이미지: 가장 신뢰도 높은 방법의 것 사용
        best_r = max(valid, key=lambda x: x[0])[1]
        result.debug_img = best_r.debug_img if best_r.debug_img is not None else frame.copy()
        _put_method_label(result.debug_img, result.method_name, ok=True)
        return result


# ── 시각화 헬퍼 ───────────────────────────────────────────────────────────────

def _put_method_label(img: np.ndarray, label: str, ok: bool) -> None:
    color = (0, 220, 80) if ok else (0, 60, 220)
    cv2.putText(img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def _draw_fail_overlay(img: np.ndarray, notes: list[str]) -> np.ndarray:
    debug = img.copy()
    for i, note in enumerate(notes):
        cv2.putText(debug, note, (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    return debug
