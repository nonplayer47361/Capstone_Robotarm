"""
eval/session.py — 실험 세션 데이터 모델 + CSV 로깅

Sample   : 단일 측정값 (true/pred 좌표, 오차, YOLO conf, A4 repro 오차)
EvalSession : 세션 전체 관리 (추가, CSV append, 카운트)
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── 단일 샘플 ──────────────────────────────────────────────────────────────────
@dataclass
class Sample:
    timestamp:    str
    object_type:  str            # 'blueberry' | 'strawberry'
    a4_method:    str            # 'aruco' | 'composite' 등
    pt_num:       int            # 테스트 포인트 번호 (1~30)
    true_x:       float          # 실제 A4 좌표 X (mm)
    true_y:       float          # 실제 A4 좌표 Y (mm)
    pred_x:       Optional[float]  # 예측 A4 좌표 X (mm) — 탐지 실패 시 None
    pred_y:       Optional[float]
    error_x:      Optional[float]  # pred_x - true_x (mm, 부호 있음)
    error_y:      Optional[float]  # pred_y - true_y (mm, 부호 있음)
    error_dist:   Optional[float]  # sqrt(error_x^2 + error_y^2) (mm)
    yolo_conf:    Optional[float]  # YOLO 탐지 신뢰도
    a4_ok:        bool             # A4 검출 성공 여부
    a4_repro_err: Optional[float]  # A4 재투영 오차 (mm)
    yolo_ok:      bool             # YOLO 탐지 성공 여부 (어떤 클래스든 박스 검출됨)
    class_ok:     bool             # 탐지된 클래스가 true_class 와 일치
    true_class:   str              # 실제 올려놓은 객체 클래스 ('blueberry' | 'strawberry')
    pred_class:   Optional[str]    # YOLO 예측 클래스 (탐지 실패 시 None)


def _compute_class_ok(true_class: str, pred_class: Optional[str]) -> bool:
    """두 클래스가 일치하는지 확인. 클래스 정보 없으면 True (단일 모드 하위 호환)."""
    return (pred_class == true_class) if (true_class and pred_class) else True


# ── 실험 세션 ──────────────────────────────────────────────────────────────────
class EvalSession:
    """
    단일 실험 세션 (객체 종류 1개 + A4 방법 1개 + 카메라 1대).

    Parameters
    ----------
    object_type : 'blueberry' | 'strawberry' | 임의 문자열
    a4_method   : 'aruco' | 'composite' | ...
    log_dir     : CSV 로그 저장 디렉터리
    """

    def __init__(
        self,
        object_type: str,
        a4_method:   str,
        log_dir:     Path,
    ) -> None:
        self.object_type = object_type
        self.a4_method   = a4_method
        self.log_dir     = Path(log_dir)
        self.samples: list[Sample] = []

        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"eval_{object_type}_{a4_method}_{ts}.csv"
        self._header_written = False

    # ── 샘플 추가 ──────────────────────────────────────────────────────────────
    def add(
        self,
        pt_num:       int,
        true_x:       float,
        true_y:       float,
        pred_x:       Optional[float],
        pred_y:       Optional[float],
        yolo_conf:    Optional[float],
        a4_ok:        bool,
        a4_repro_err: Optional[float],
        yolo_ok:      bool,
        true_class:   str           = "",
        pred_class:   Optional[str] = None,
    ) -> Sample:
        """샘플 1개를 기록하고 CSV에 즉시 append.

        class_ok: 탐지된 클래스가 true_class 와 일치하는지.
        true_class/pred_class 중 하나라도 없으면 True로 처리 (단일 모드 하위 호환).
        """
        ts = datetime.now().isoformat(timespec="milliseconds")

        if pred_x is not None and pred_y is not None:
            error_x    = pred_x - true_x
            error_y    = pred_y - true_y
            error_dist = math.hypot(error_x, error_y)
        else:
            error_x = error_y = error_dist = None

        tc = true_class or self.object_type
        class_ok = _compute_class_ok(tc, pred_class)

        s = Sample(
            timestamp=ts,
            object_type=self.object_type,
            a4_method=self.a4_method,
            pt_num=pt_num,
            true_x=true_x,
            true_y=true_y,
            pred_x=pred_x,
            pred_y=pred_y,
            error_x=error_x,
            error_y=error_y,
            error_dist=error_dist,
            yolo_conf=yolo_conf,
            a4_ok=a4_ok,
            a4_repro_err=a4_repro_err,
            yolo_ok=yolo_ok,
            class_ok=class_ok,
            true_class=tc,
            pred_class=pred_class,
        )
        self.samples.append(s)
        self._append_csv(s)
        return s

    # ── 카운트 헬퍼 ────────────────────────────────────────────────────────────
    def count_for_pt(self, pt_num: int) -> int:
        """해당 포인트의 전체 캡처 수 (성공+실패)."""
        return sum(1 for s in self.samples if s.pt_num == pt_num)

    def success_for_pt(self, pt_num: int) -> int:
        """해당 포인트의 성공 캡처 수 (A4 + YOLO + class 모두 OK)."""
        return sum(
            1 for s in self.samples
            if s.pt_num == pt_num and s.a4_ok and s.yolo_ok and s.class_ok
        )

    def reset_pt(self, pt_num: int) -> None:
        """해당 포인트의 샘플을 메모리에서 제거하고 CSV에 RESET 마커를 기록합니다."""
        self.samples = [s for s in self.samples if s.pt_num != pt_num]
        self.write_reset_marker(pt_num)

    @property
    def n_success(self) -> int:
        return sum(
            1 for s in self.samples
            if s.a4_ok and s.yolo_ok and s.class_ok
        )

    @property
    def n_total(self) -> int:
        return len(self.samples)

    # ── CSV I/O ────────────────────────────────────────────────────────────────
    def _append_csv(self, s: Sample) -> None:
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(s).keys()))
            if not self._header_written:
                w.writeheader()
                self._header_written = True
            w.writerow({
                k: ("" if v is None else v)
                for k, v in asdict(s).items()
            })

    def write_reset_marker(self, pt_num: int) -> None:
        """R 키 리셋 이벤트를 CSV에 마커로 기록.

        load_csv() 는 이 마커 이전에 기록된 해당 pt_num 샘플을 자동으로 제외한다.
        마커 행은 timestamp 컬럼에 'RESET:PT{pt_num}' 을 기록하고 나머지는 비워둔다.
        """
        fieldnames = list(Sample.__dataclass_fields__.keys())
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written:
                w.writeheader()
                self._header_written = True
            row = {k: "" for k in fieldnames}
            row["timestamp"] = f"RESET:PT{pt_num}"
            w.writerow(row)

    @classmethod
    def load_csv(cls, path: Path) -> list[Sample]:
        """CSV 파일에서 Sample 리스트 복원.

        RESET 마커가 있는 경우, 마커 이전에 기록된 동일 pt_num 샘플을 제외한다.
        """
        raw_rows: list[tuple[int, dict]] = []       # (line_index, row_dict)
        latest_reset_idx: dict[int, int] = {}        # pt_num -> latest reset line_index

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                ts = row.get("timestamp", "")
                if ts.startswith("RESET:PT"):
                    try:
                        pt = int(ts.split("RESET:PT")[1].rstrip(","))
                        latest_reset_idx[pt] = i
                    except ValueError:
                        pass
                    continue
                raw_rows.append((i, row))

        samples = []
        for i, row in raw_rows:
            try:
                pt_num = int(row.get("pt_num", "0") or 0)
            except ValueError:
                continue

            reset_idx = latest_reset_idx.get(pt_num)
            if reset_idx is not None and i < reset_idx:
                continue

            def _f(k):
                v = row.get(k, "")
                return float(v) if v != "" else None

            def _b(k):
                return row.get(k, "").lower() == "true"

            tc = row.get("true_class", "") or row.get("object_type", "")
            pc = row.get("pred_class", "") or None
            ck = _b("class_ok") if "class_ok" in row else _compute_class_ok(tc, pc)

            try:
                samples.append(Sample(
                    timestamp    = row["timestamp"],
                    object_type  = row["object_type"],
                    a4_method    = row["a4_method"],
                    pt_num       = pt_num,
                    true_x       = float(row["true_x"]),
                    true_y       = float(row["true_y"]),
                    pred_x       = _f("pred_x"),
                    pred_y       = _f("pred_y"),
                    error_x      = _f("error_x"),
                    error_y      = _f("error_y"),
                    error_dist   = _f("error_dist"),
                    yolo_conf    = _f("yolo_conf"),
                    a4_ok        = _b("a4_ok"),
                    a4_repro_err = _f("a4_repro_err"),
                    yolo_ok      = _b("yolo_ok"),
                    class_ok     = ck,
                    true_class   = tc,
                    pred_class   = pc,
                ))
            except (KeyError, ValueError):
                continue  # 불완전한 행 무시
        return samples
