"""
eval/report.py — 실험 결과 통계 + 리포트 생성

compute_report(samples)  → 통계 dict
print_report(report)     → 콘솔 출력
save_report_json(report) → JSON 저장
bias_corrections(report) → 배포용 보정값 계산
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

from .session import Sample


# ── 통계 유틸 ──────────────────────────────────────────────────────────────────
def _p90(lst: list[float]) -> float:
    if not lst:
        return float("nan")
    s   = sorted(lst)
    idx = int(len(s) * 0.9)
    return s[min(idx, len(s) - 1)]


def _pct(lst: list[float], thresh: float) -> float:
    if not lst:
        return 0.0
    return sum(1 for e in lst if e <= thresh) / len(lst) * 100.0


def _safe_mean(lst):
    return mean(lst) if lst else float("nan")

def _safe_median(lst):
    return median(lst) if lst else float("nan")

def _safe_stdev(lst):
    return stdev(lst) if len(lst) > 1 else float("nan")

def _safe_min(lst):
    return min(lst) if lst else float("nan")

def _safe_max(lst):
    return max(lst) if lst else float("nan")


# ── 메인 리포트 계산 ───────────────────────────────────────────────────────────
def compute_report(samples: list[Sample]) -> dict:
    """
    Sample 리스트 → 평가 통계 dict.

    반환 키:
      object_type, a4_method, n_total,
      a4_success_rate_pct, detection_rate_pct, a4_repro_mean_mm,
      coord: {mean/median/p90/max/std/bias_x/bias_y/within_5/10/15mm_pct},
      per_position: [{pt_num, true_x, true_y, n, mean_mm, bias_x, bias_y}],
      bias_correction: {x: float, y: float}  (배포용 보정값)
    """
    n = len(samples)
    if n == 0:
        return {"error": "샘플 없음"}

    a4_ok = [s for s in samples if s.a4_ok]
    # 좌표 오차 통계는 A4 좌표 변환, YOLO 탐지, 클래스 정답이 모두 통과하고
    # 실제 좌표 오차가 계산된 샘플만 포함한다.
    success = [
        s for s in samples
        if s.a4_ok and s.yolo_ok and s.class_ok and s.error_dist is not None
    ]

    errors  = [s.error_dist for s in success if s.error_dist is not None]
    ex_list = [s.error_x    for s in success if s.error_x    is not None]
    ey_list = [s.error_y    for s in success if s.error_y    is not None]
    repro   = [s.a4_repro_err for s in a4_ok if s.a4_repro_err is not None]
    tilts   = [s.tilt_score for s in samples if s.tilt_score is not None]
    conditions = sorted({s.condition or "unspecified" for s in samples})

    # ── 포인트별 통계 ──────────────────────────────────────────────────────────
    pt_nums   = sorted({s.pt_num for s in samples})
    per_pos   = []
    all_biases_x, all_biases_y = [], []

    for pt in pt_nums:
        pt_s  = [s for s in success if s.pt_num == pt]
        if not pt_s:
            continue
        errs  = [s.error_dist for s in pt_s if s.error_dist is not None]
        bx    = _safe_mean([s.error_x for s in pt_s if s.error_x is not None])
        by    = _safe_mean([s.error_y for s in pt_s if s.error_y is not None])
        all_biases_x.append(bx)
        all_biases_y.append(by)
        per_pos.append({
            "pt_num"  : pt,
            "true_x"  : pt_s[0].true_x,
            "true_y"  : pt_s[0].true_y,
            "n"       : len(pt_s),
            "mean_mm" : _safe_mean(errs),
            "max_mm"  : max(errs) if errs else float("nan"),
            "bias_x"  : round(bx, 3),
            "bias_y"  : round(by, 3),
        })

    # ── 배포용 보정값 (전체 평균 bias를 제거) ─────────────────────────────────
    # 적용 방법: corrected_x = pred_x - bias_x_mm
    bias_x = _safe_mean(ex_list)
    bias_y = _safe_mean(ey_list)

    # bias 보정 후 오차 (참고용)
    corrected = []
    for s in success:
        if s.error_x is not None and s.error_y is not None:
            cx = s.error_x - bias_x
            cy = s.error_y - bias_y
            corrected.append(math.hypot(cx, cy))

    # ── 분류 정확도 (true_class / pred_class 가 기록된 경우) ───────────────────
    # 분류 정확도는 좌표 성공 샘플이 아니라 YOLO가 박스를 낸 모든 샘플을 기준으로 계산한다.
    cls_samples = [s for s in samples if s.yolo_ok and s.true_class and s.pred_class]
    if cls_samples:
        cls_correct = sum(1 for s in cls_samples if s.pred_class == s.true_class)
        cls_acc     = cls_correct / len(cls_samples) * 100

        classes = sorted({s.true_class for s in cls_samples})
        per_class_cls = {}
        for cls in classes:
            cls_s   = [s for s in cls_samples if s.true_class == cls]
            correct = sum(1 for s in cls_s if s.pred_class == cls)
            coord_s = [
                s for s in cls_s
                if s.a4_ok and s.class_ok and s.error_dist is not None
            ]
            per_class_cls[cls] = {
                "n"          : len(cls_s),
                "correct"    : correct,
                "acc_pct"    : round(correct / len(cls_s) * 100, 1),
                "mean_mm"    : round(_safe_mean([s.error_dist for s in coord_s]), 3),
            }
        classification = {
            "n"          : len(cls_samples),
            "correct"    : cls_correct,
            "acc_pct"    : round(cls_acc, 1),
            "per_class"  : per_class_cls,
        }
    else:
        classification = None

    # ── 조건별 요약: 수평/기울기 실험을 한 CSV에 섞어도 비교 가능 ──────────────
    per_condition = []
    for cond in conditions:
        cond_s = [s for s in samples if (s.condition or "unspecified") == cond]
        cond_success = [
            s for s in cond_s
            if s.a4_ok and s.yolo_ok and s.class_ok and s.error_dist is not None
        ]
        cond_errors = [s.error_dist for s in cond_success if s.error_dist is not None]
        cond_tilts = [s.tilt_score for s in cond_s if s.tilt_score is not None]
        per_condition.append({
            "condition"       : cond,
            "n"               : len(cond_s),
            "n_coord_ok"      : len(cond_success),
            "tilt_score_mean" : round(_safe_mean(cond_tilts), 4),
            "tilt_score_min"  : round(_safe_min(cond_tilts), 4),
            "tilt_score_max"  : round(_safe_max(cond_tilts), 4),
            "coord_mean_mm"   : round(_safe_mean(cond_errors), 3),
            "coord_p90_mm"    : round(_p90(cond_errors), 3),
        })

    return {
        "object_type"         : samples[0].object_type,
        "a4_method"           : samples[0].a4_method,
        "condition"           : conditions[0] if len(conditions) == 1 else "mixed",
        "conditions"          : conditions,
        "n_total"             : n,
        "n_a4_ok"             : len(a4_ok),
        "n_yolo_ok"           : sum(1 for s in samples if s.yolo_ok),
        "n_class_ok"          : sum(1 for s in samples if s.yolo_ok and s.class_ok),
        "n_coord_ok"          : len(success),   # a4_ok + yolo_ok + class_ok
        "a4_success_rate_pct" : len(a4_ok)   / n * 100,
        "detection_rate_pct"  : len(success) / n * 100,
        "a4_repro_mean_mm"    : round(_safe_mean(repro), 3),
        "tilt_score": {
            "n"    : len(tilts),
            "mean" : round(_safe_mean(tilts), 4),
            "min"  : round(_safe_min(tilts), 4),
            "max"  : round(_safe_max(tilts), 4),
            "note" : "1.0에 가까울수록 수직 촬영에 가깝고, 낮을수록 원근 왜곡이 큰 프레임",
        },

        "coord": {
            "n"              : len(errors),
            "mean_mm"        : round(_safe_mean(errors),   3),
            "median_mm"      : round(_safe_median(errors), 3),
            "p90_mm"         : round(_p90(errors),         3),
            "max_mm"         : round(max(errors) if errors else float("nan"), 3),
            "std_mm"         : round(_safe_stdev(errors),  3),
            "bias_x_mm"      : round(bias_x, 3),
            "bias_y_mm"      : round(bias_y, 3),
            "within_5mm_pct" : round(_pct(errors, 5.0),  1),
            "within_10mm_pct": round(_pct(errors, 10.0), 1),
            "within_15mm_pct": round(_pct(errors, 15.0), 1),
            # bias 보정 후
            "corrected_mean_mm"  : round(_safe_mean(corrected),   3),
            "corrected_p90_mm"   : round(_p90(corrected),         3),
            "corrected_5mm_pct"  : round(_pct(corrected, 5.0),  1),
            "corrected_10mm_pct" : round(_pct(corrected, 10.0), 1),
        },

        "bias_correction": {
            "x_mm": round(-bias_x, 3),   # pred_x + x_mm 로 보정
            "y_mm": round(-bias_y, 3),
            "note": "배포 시 pred_x += bias_correction.x_mm 로 보정"
        },

        "per_position"  : per_pos,
        "per_condition" : per_condition,
        "classification": classification,
    }


# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────
def print_report(report: dict) -> None:
    if "error" in report:
        print(f"[report] ERROR: {report['error']}")
        return

    c = report["coord"]
    bc = report["bias_correction"]

    cls  = report.get("classification")
    mode = "MIXED" if report["object_type"] == "mixed" else report["object_type"]
    tilt = report.get("tilt_score") or {}
    tilt_line = "  기울기 지표      : 기록 없음"
    if tilt.get("n", 0):
        tilt_line = (
            f"  기울기 지표      : mean={tilt['mean']:.4f}  "
            f"min={tilt['min']:.4f}  max={tilt['max']:.4f}  "
            "(1.0에 가까울수록 수직)"
        )

    lines = [
        "",
        "=" * 62,
        f"  좌표 오차 리포트",
        f"  객체: {mode}   A4 방법: {report['a4_method']}   조건: {report.get('condition', 'unspecified')}",
        "=" * 62,
        f"  총 캡처         : {report['n_total']}",
        f"  A4 검출 성공    : {report['n_a4_ok']}  ({report['a4_success_rate_pct']:.1f}%)",
        f"  YOLO 탐지 성공  : {report['n_yolo_ok']}",
        f"  클래스 정답     : {report['n_class_ok']}",
        f"  좌표 변환 성공  : {report['n_coord_ok']}  ({report['detection_rate_pct']:.1f}%)  [A4+YOLO+class 모두 OK]",
        f"  A4 repro 오차   : {report['a4_repro_mean_mm']:.3f} mm (평균)",
        tilt_line,
        "-" * 62,
        f"  좌표 오차 통계 (n={c['n']})",
        f"    평균    : {c['mean_mm']:>7.2f} mm",
        f"    중앙값  : {c['median_mm']:>7.2f} mm",
        f"    p90     : {c['p90_mm']:>7.2f} mm",
        f"    최대    : {c['max_mm']:>7.2f} mm",
        f"    표준편차: {c['std_mm']:>7.2f} mm",
        "-" * 62,
        f"  X 방향 bias: {c['bias_x_mm']:+.2f} mm   Y 방향 bias: {c['bias_y_mm']:+.2f} mm",
        "-" * 62,
        f"   5mm 이내: {c['within_5mm_pct']:>5.1f}%",
        f"  10mm 이내: {c['within_10mm_pct']:>5.1f}%",
        f"  15mm 이내: {c['within_15mm_pct']:>5.1f}%",
        "-" * 62,
        f"  bias 보정 후 평균: {c['corrected_mean_mm']:.2f} mm   "
        f"p90: {c['corrected_p90_mm']:.2f} mm",
        f"  bias 보정 후 5mm: {c['corrected_5mm_pct']:.1f}%   "
        f"10mm: {c['corrected_10mm_pct']:.1f}%",
        "-" * 62,
        f"  배포용 보정값: x += {bc['x_mm']:+.2f} mm   y += {bc['y_mm']:+.2f} mm",
        "=" * 62,
    ]
    print("\n".join(lines))

    # 분류 정확도 (mixed 모드)
    if cls:
        print(f"\n  클래스 분류 정확도 (n={cls['n']})")
        print(f"  전체: {cls['correct']}/{cls['n']}  ({cls['acc_pct']:.1f}%)")
        print("  " + "-" * 44)
        print(f"  {'클래스':<14} {'n':>4}  {'정답':>4}  {'정확도':>7}  {'좌표오차(평균)':>14}")
        print("  " + "-" * 44)
        for cls_name, stat in cls["per_class"].items():
            print(f"  {cls_name:<14} {stat['n']:>4}  {stat['correct']:>4}  "
                  f"{stat['acc_pct']:>6.1f}%  {stat['mean_mm']:>12.2f} mm")

    if len(report.get("per_condition", [])) >= 1:
        print(f"\n  조건별 요약")
        print(f"  {'condition':<14} {'n':>4} {'ok':>4} {'tilt_mean':>10} {'mean':>8} {'p90':>8}")
        print("  " + "-" * 58)
        for p in report.get("per_condition", []):
            print(f"  {p['condition']:<14} {p['n']:>4} {p['n_coord_ok']:>4} "
                  f"{p['tilt_score_mean']:>10.4f} {p['coord_mean_mm']:>8.2f} {p['coord_p90_mm']:>8.2f}")

    # 포인트별 요약
    print(f"\n  포인트별 오차 (mm)")
    print(f"  {'pt':>3} {'true':>12}  {'n':>3}  {'mean':>6}  {'max':>6}  bias_x  bias_y")
    print("  " + "-" * 55)
    for p in report.get("per_position", []):
        pos = f"({p['true_x']:.0f},{p['true_y']:.0f})"
        print(f"  {p['pt_num']:>3} {pos:>12}  {p['n']:>3}  "
              f"{p['mean_mm']:>6.2f}  {p['max_mm']:>6.2f}  "
              f"{p['bias_x']:>+6.2f}  {p['bias_y']:>+6.2f}")


# ── JSON 저장 ──────────────────────────────────────────────────────────────────
def save_report_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f"[report] JSON 저장: {path}")
