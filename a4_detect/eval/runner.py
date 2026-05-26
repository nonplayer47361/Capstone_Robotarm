"""
eval/runner.py — 좌표 오차 실험 인터랙티브 러너

사용 흐름:
  자동 진행 모드 (기본)
    1. sheet_eval_30pt.png 출력 후 카메라 아래 고정
    2. 각 테스트 포인트에 객체를 올려놓기
    3. A4 검출 + YOLO 탐지 확인 후 [Space] 캡처
    4. repeats 만큼 캡처되면 자동으로 다음 포인트로 이동
    5. [Q] 종료 시 CSV 저장 + 리포트 출력

  수동 진행 모드 (manual_advance=True, 1점 시트 / 용지 교체 실험)
    1. 화면에 표시된 시트 번호의 용지를 카메라 아래 배치
    2. 40mm 원 위에 캡 올려놓기
    3. [Space] 캡처 (여러 번 가능)
    4. HUD 에 완료 메시지 뜨면 다음 용지로 교체
    5. [N] 다음 포인트, [P] 이전 포인트, [Q] 종료

키 바인딩:
  [Space]  — 현재 프레임 캡처 (A4 + YOLO 상관없이 상태 기록)
  [N]      — 다음 포인트로 이동 (수동 모드에서는 용지 교체 후 누름)
  [P]      — 이전 포인트로 돌아가기
  [Q]      — 종료 + 리포트 생성
  [R]      — 현재 포인트 캡처 초기화 (재실험)
  [S]      — 스냅샷 저장
  [1]      — (mixed 모드) 클래스1 객체로 지정 (예: pill_cap)
  [2]      — (mixed 모드) 클래스2 객체로 지정 (예: coin / bottle_cap)

mixed 모드에서 [1]/[2] 가 매핑되는 실제 클래스 이름은 --object-type 인자가 아닌
YOLO 모델이 반환하는 클래스 이름에 따라 달라집니다.
현재 구현은 blueberry(1) / strawberry(2) 라벨을 기본값으로 사용하며,
pill_cap / coin / bottle_cap 등 새 객체 실험 시에는 mixed 모드 없이
--object-type cap 처럼 단일 타입으로 지정하여 사용하세요.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .session import EvalSession
from .report  import compute_report, print_report, save_report_json

# plane_coord 는 상위 패키지(a4_detect/)에 있으므로 sys.path 보장 후 임포트
_A4_DIR = Path(__file__).resolve().parent.parent
if str(_A4_DIR) not in sys.path:
    sys.path.insert(0, str(_A4_DIR))
from plane_coord import A4_W_MM, A4_H_MM                    # noqa: E402
from plane_coord.camera_calib import CameraCalib, maybe_undistort  # noqa: E402

MINI_SCALE = 2   # 미니맵: 1mm = 2px

# ── 좌표 추정 품질 기본값 ────────────────────────────────────────────────────
# H 품질 게이트: repro_err 가 이 값보다 크면 해당 프레임의 H를 폐기
DEFAULT_REPRO_ERR_THRESH_MM = 3.0


# ═════════════════════════════════════════════════════════════════════════════
# 좌표 추정 헬퍼
# ═════════════════════════════════════════════════════════════════════════════

def _pick_box_on_a4(boxes, a4_result) -> object:
    """
    A4 영역 안에 중심이 있는 박스 중 가장 높은 confidence 를 반환합니다.
    A4 내 박스가 없으면 화면 전체에서 가장 높은 confidence 를 반환합니다.

    이 함수는 A4 밖의 동일 객체(예: 테이블 위 여러 뚜껑 중 용지 위 것만)를
    선택하기 위해 사용됩니다.
    """
    if not a4_result.ok:
        return max(boxes, key=lambda b: float(b.conf[0]))

    candidates = []
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0]
        cx, cy = float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
        mx, my = a4_result.px_to_mm(cx, cy)
        if 0.0 <= mx <= A4_W_MM and 0.0 <= my <= A4_H_MM:
            candidates.append(b)

    pool = candidates if candidates else list(boxes)
    return max(pool, key=lambda b: float(b.conf[0]))


def _validate_coord(pred_x: float, pred_y: float) -> tuple[float | None, float | None]:
    """
    예측 좌표가 A4 용지 범위 안에 있으면 그대로 반환, 범위 밖이면 (None, None).

    호모그래피가 순간적으로 나쁠 때 비정상 좌표가 샘플에 섞이는 것을 방지합니다.
    """
    if 0.0 <= pred_x <= A4_W_MM and 0.0 <= pred_y <= A4_H_MM:
        return pred_x, pred_y
    return None, None


# ═════════════════════════════════════════════════════════════════════════════
# 메인 실행 함수
# ═════════════════════════════════════════════════════════════════════════════

def run_eval(
    object_type:          str,
    model_path:           str,
    camera_id:            int,
    plane_method:         str,
    composite_mode:       str,
    test_pts:             list[tuple[int, float, float]],
    repeats:              int,
    conf_thresh:          float,
    log_dir:              Path,
    manual_advance:       bool              = False,
    calib:                CameraCalib | None = None,
    aruco_marker_size_mm: float             = 20.0,
    repro_err_thresh_mm:  float             = DEFAULT_REPRO_ERR_THRESH_MM,
) -> None:
    """
    Parameters
    ----------
    object_type          : 측정 객체 종류 ('cap'|'coin'|'blueberry'|'mixed' 등)
    model_path           : YOLO .pt 경로
    camera_id            : 카메라 ID
    plane_method         : 'aruco' | 'composite' | ...
    composite_mode       : 'priority_fallback' | 'best_of_all' | 'vote_H'
    test_pts             : [(번호, x_mm, y_mm), ...]
    repeats              : 포인트당 캡처 횟수 (기본 3)
    conf_thresh          : YOLO conf 임계값
    log_dir              : CSV 로그 저장 디렉터리
    manual_advance       : True → 캡처 완료 후 자동 넘김 없음, [N] 키로 수동 진행
    calib                : CameraCalib — 렌즈 왜곡 보정 파라미터. None 이면 보정 안 함
    aruco_marker_size_mm : 출력물 ArUco 마커 크기 (mm). gen.py 와 반드시 일치
    repro_err_thresh_mm  : H 품질 게이트 — repro_err 가 이 값 초과 시 해당 H 폐기
                           (기본: 3.0mm)
    """
    # ── 모델 초기화 ────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics 필요: pip install ultralytics")

    # sys.path 설정은 모듈 레벨에서 완료됨
    from plane_coord.composite import CompositeDetector
    from plane_coord.aruco import ArucoDetector
    from plane_coord import METHODS

    if plane_method == "composite":
        plane_det = CompositeDetector(mode=composite_mode)
    elif plane_method == "aruco":
        plane_det = ArucoDetector(marker_size_mm=aruco_marker_size_mm)
    else:
        plane_det = METHODS[plane_method]()

    yolo_model = YOLO(model_path)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {camera_id} 열기 실패")

    if calib is not None:
        print(f"[eval] 렌즈 왜곡 보정 활성  ({calib.rms_px:.4f}px rms)")
    if repro_err_thresh_mm < float("inf"):
        print(f"[eval] H 품질 게이트: repro_err > {repro_err_thresh_mm:.1f}mm → H 폐기")

    is_mixed = (object_type == "mixed")
    # mixed 모드: 사용자가 B/S 키로 현재 올려놓은 객체 클래스를 지정
    current_class = "blueberry"   # mixed 초기값; 단일 모드에서는 object_type으로 덮어씌움
    if not is_mixed:
        current_class = object_type

    session  = EvalSession(object_type, plane_method, log_dir)
    pt_idx   = 0
    WIN      = f"Coord Eval — {object_type} / {plane_method}  [Space=캡처  N=다음  Q=종료]"
    snap_idx = 0

    print(f"\n[eval] 객체: {object_type}  방법: {plane_method}  모델: {model_path}")
    if is_mixed:
        print("[eval] MIXED 모드: 1=클래스1 / 2=클래스2 키로 현재 객체 지정 후 Space 캡처")
    if manual_advance:
        print("[eval] 수동 진행 모드: 캡처 완료 후 용지 교체 → [N] 다음 / [P] 이전")
    print(f"[eval] {len(test_pts)}개 포인트 × {repeats}회 반복 = {len(test_pts)*repeats}샘플 목표")
    print(f"[eval] 로그: {session.log_path}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = maybe_undistort(frame, calib)

        # 현재 테스트 포인트
        if pt_idx >= len(test_pts):
            # 모든 포인트 완료
            _draw_done(frame, session)
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 13):  # q 또는 Enter
                break
            continue

        pt_num, true_x, true_y = test_pts[pt_idx]

        # ── A4 평면 탐지 ───────────────────────────────────────────────────────
        a4_result = plane_det.timed_detect(frame)

        # H 품질 게이트: repro_err 가 임계값 초과하면 H 폐기
        if (a4_result.ok
                and a4_result.repro_err_mm < 1e6           # inf 제외
                and a4_result.repro_err_mm > repro_err_thresh_mm):
            a4_result.H    = None
            a4_result.note = (
                f"H 폐기 — repro_err {a4_result.repro_err_mm:.2f}mm "
                f"> 게이트 {repro_err_thresh_mm:.1f}mm"
            )

        # ── YOLO 탐지 ──────────────────────────────────────────────────────────
        yolo_box: Optional[object] = None
        cx_px = cy_px = None
        pred_x = pred_y = None
        yolo_conf_val = None
        pred_class_val: Optional[str] = None

        yolo_res = yolo_model.predict(frame, conf=conf_thresh, verbose=False)[0]
        boxes = yolo_res.boxes
        if boxes:
            # A4 영역 내 박스 우선 선택
            yolo_box = _pick_box_on_a4(boxes, a4_result)
            x1, y1, x2, y2 = (int(v) for v in yolo_box.xyxy[0])
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0
            yolo_conf_val  = float(yolo_box.conf[0])
            pred_class_val = yolo_res.names[int(yolo_box.cls[0])]
            if a4_result.ok:
                raw_x, raw_y = a4_result.px_to_mm(cx_px, cy_px)
                pred_x, pred_y = _validate_coord(raw_x, raw_y)

        # ── 시각화 ──────────────────────────────────────────────────────────────
        # debug_img 는 탐지기가 매 프레임 frame.copy() 로 새로 생성하므로 추가 복사 불필요
        vis = a4_result.debug_img if a4_result.debug_img is not None else frame.copy()

        # YOLO 박스
        if yolo_box is not None:
            cls_match = (pred_class_val == current_class) if pred_class_val else False
            if not a4_result.ok:
                box_color = (0, 180, 220)
            elif cls_match:
                box_color = (0, 220, 30)
            else:
                box_color = (0, 40, 220)
            cls_label = (
                f"{pred_class_val}  {'OK' if cls_match else 'WRONG'}"
                if pred_class_val else ""
            )
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
            cv2.drawMarker(vis, (int(cx_px), int(cy_px)),
                           (0, 0, 230), cv2.MARKER_CROSS, 22, 2)
            if pred_x is not None:
                err = np.hypot(pred_x - true_x, pred_y - true_y)
                cv2.putText(vis,
                            f"{cls_label}  ({pred_x:.1f},{pred_y:.1f})  err={err:.1f}mm",
                            (x1, max(y1 - 10, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            else:
                cv2.putText(vis,
                            f"{cls_label}  A4 FAIL - no coord",
                            (x1, max(y1 - 10, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # 현재 테스트 포인트를 카메라 화면에 역투영
        if a4_result.ok:
            px_x, px_y = a4_result.mm_to_px(true_x, true_y)
            h, w = vis.shape[:2]
            if 0 <= px_x < w and 0 <= px_y < h:
                cv2.drawMarker(vis, (int(px_x), int(px_y)),
                               (0, 120, 255), cv2.MARKER_CROSS, 30, 2)
                cv2.putText(vis, f"PT{pt_num} ({true_x:.0f},{true_y:.0f})",
                            (int(px_x) + 8, int(px_y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 100, 220), 1)

        # 현재 포인트 완료 여부
        pt_done = session.count_for_pt(pt_num) >= repeats

        # 정보 바 (상단)
        _draw_info_bar(
            vis, pt_num, true_x, true_y, pt_idx, len(test_pts),
            session.count_for_pt(pt_num), session.success_for_pt(pt_num), repeats,
            a4_result.ok, yolo_box is not None,
            session.n_total, session.n_success,
            current_class if is_mixed else None,
            pred_class_val if is_mixed else None,
            manual_advance=manual_advance,
            pt_done=pt_done,
        )

        # 미니맵
        mini = _make_minimap(
            a4_result, pred_x, pred_y,
            true_x, true_y,
            test_pts, session, vis.shape[0],
        )
        combined = np.hstack([vis, mini])
        cv2.imshow(WIN, combined)

        # ── 키 처리 ─────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):   # 캡처
            s = session.add(
                pt_num       = pt_num,
                true_x       = true_x,
                true_y       = true_y,
                pred_x       = pred_x,
                pred_y       = pred_y,
                yolo_conf    = yolo_conf_val,
                a4_ok        = a4_result.ok,
                a4_repro_err = a4_result.repro_err_mm if a4_result.ok else None,
                tilt_score   = a4_result.tilt_score,
                yolo_ok      = yolo_box is not None,
                true_class   = current_class,
                pred_class   = pred_class_val,
            )
            _print_capture(s, session.count_for_pt(pt_num), repeats)

            # 자동 진행 모드에서만 auto-advance
            if not manual_advance and session.count_for_pt(pt_num) >= repeats:
                pt_idx = min(pt_idx + 1, len(test_pts))

        elif key == ord("n"):   # 다음
            pt_idx = min(pt_idx + 1, len(test_pts))

        elif key == ord("p"):   # 이전
            pt_idx = max(pt_idx - 1, 0)

        elif key == ord("r"):   # 현재 포인트 리셋
            session.reset_pt(pt_num)
            print(f"[eval] PT{pt_num} 초기화  (CSV에 RESET 마커 기록)")

        elif key == ord("1") and is_mixed:   # 클래스 선택: blueberry
            current_class = "blueberry"
            print(f"[eval] 현재 객체 → blueberry")

        elif key == ord("2") and is_mixed:   # 클래스 선택: strawberry
            current_class = "strawberry"
            print(f"[eval] 현재 객체 → strawberry")

        elif key == ord("s"):   # 스냅샷 (모든 모드에서 동작)
            fname = log_dir / f"snap_{snap_idx:04d}.jpg"
            cv2.imwrite(str(fname), combined)
            print(f"[snap] {fname}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # ── 종료: 리포트 생성 ──────────────────────────────────────────────────────
    if session.samples:
        report = compute_report(session.samples)
        print_report(report)
        rp = session.log_path.with_suffix(".json")
        save_report_json(report, rp)
        print(f"\n[eval] CSV: {session.log_path}")
        print(f"[eval] JSON: {rp}")
    else:
        print("[eval] 샘플 없음 — 리포트 생략")


# ═════════════════════════════════════════════════════════════════════════════
# UI 헬퍼
# ═════════════════════════════════════════════════════════════════════════════

def _draw_info_bar(
    vis, pt_num, true_x, true_y, pt_idx, n_pts,
    n_captured, n_success_pt, repeats,
    a4_ok, yolo_ok,
    n_total, n_ok_total,
    current_class=None,    # mixed 모드 전용
    pred_class=None,       # mixed 모드 전용
    manual_advance=False,
    pt_done=False,
):
    """상단 검정 정보 바."""
    h, w = vis.shape[:2]
    bar_h = 80
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)

    a4_c = (0, 220, 80) if a4_ok   else (0, 60, 200)
    yo_c = (0, 220, 80) if yolo_ok else (0, 60, 200)

    # ── 진행 바 (상단 2px) ────────────────────────────────────────────
    bar_fill = int(w * pt_idx / max(n_pts, 1))
    bar[:2, :bar_fill] = (60, 200, 80)

    # ── Row 1: 포인트 정보 ────────────────────────────────────────────
    if manual_advance:
        sheet_text = f"Sheet {pt_idx + 1:03d}/{n_pts:03d}  PT{pt_num:02d}  ({true_x:.0f},{true_y:.0f})mm"
    else:
        sheet_text = f"PT {pt_num:02d}/{n_pts}  ({true_x:.0f},{true_y:.0f})mm"
    cv2.putText(bar,
                f"{sheet_text}  captures: {n_captured}/{repeats}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1)

    # ── Row 2: A4/YOLO 상태 ───────────────────────────────────────────
    cv2.putText(bar, f"A4: {'OK' if a4_ok else 'FAIL'}",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, a4_c, 1)
    cv2.putText(bar, f"YOLO: {'found' if yolo_ok else 'none'}",
                (120, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, yo_c, 1)

    if manual_advance and pt_done:
        # 수동 모드: 현재 포인트 완료 → 용지 교체 안내
        cv2.putText(bar,
                    "DONE  — 용지 교체 후  [N] 다음  /  [P] 이전  /  [R] 재실험",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 200), 2)
    elif current_class is not None:
        # mixed 모드: 현재 지정 클래스 + 예측 결과
        cls_c = (0, 200, 255)
        cv2.putText(bar, f"[1/2] now: {current_class}",
                    (250, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cls_c, 1)
        if pred_class:
            match = pred_class == current_class
            pc_c  = (0, 220, 80) if match else (0, 40, 220)
            cv2.putText(bar, f"pred: {pred_class} ({'OK' if match else 'WRONG'})",
                        (250, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, pc_c, 1)
        hint = "[1=cls1  2=cls2  Space=cap  N=next  Q=quit]"
        cv2.putText(bar, f"total {n_total} / ok {n_ok_total}  {hint}",
                    (480, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    else:
        hint = "[Space=cap  N=next  P=prev  R=reset  Q=quit]"
        if manual_advance:
            hint = "[Space=cap  N=다음용지  P=이전용지  R=재실험  Q=종료]"
        cv2.putText(bar,
                    f"total {n_total} / ok {n_ok_total}  {hint}",
                    (260, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

    vis[:bar_h] = bar


def _draw_done(vis, session: EvalSession):
    cv2.putText(vis, "All points captured!",
                (40, vis.shape[0] // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 80), 3)
    cv2.putText(vis, f"Total: {session.n_total}  Success: {session.n_success}",
                (40, vis.shape[0] // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(vis, "Press Q or Enter to generate report",
                (40, vis.shape[0] // 2 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 100), 1)


def _print_capture(s, count, repeats):
    ok  = "OK" if (s.a4_ok and s.yolo_ok and s.class_ok) else "FAIL"
    err = f"err={s.error_dist:.2f}mm" if s.error_dist is not None else "no-coord"
    cls_info = ""
    if s.true_class and s.pred_class:
        match    = s.pred_class == s.true_class
        cls_info = f"  cls: {s.true_class}->{s.pred_class} ({'OK' if match else 'WRONG'})"
    print(f"  [cap] PT{s.pt_num:02d} ({s.true_x:.0f},{s.true_y:.0f}) "
          f"#{count}/{repeats}  {ok}  {err}{cls_info}")


def _make_minimap(
    a4_result,
    pred_x: Optional[float],
    pred_y: Optional[float],
    true_x: float,
    true_y: float,
    test_pts: list,
    session: EvalSession,
    target_h: int,
) -> np.ndarray:
    """A4 미니맵: 전체 테스트 포인트 + 현재 타겟 + 예측 위치."""
    mw = int(A4_W_MM * MINI_SCALE)
    mh = int(A4_H_MM * MINI_SCALE)
    mini = np.full((mh, mw, 3), 245, dtype=np.uint8)

    # 50mm 격자
    for xm in range(0, 211, 50):
        cv2.line(mini, (xm*MINI_SCALE, 0), (xm*MINI_SCALE, mh), (185,185,185), 1)
    for ym in range(0, 298, 50):
        cv2.line(mini, (0, ym*MINI_SCALE), (mw, ym*MINI_SCALE), (185,185,185), 1)
    cv2.rectangle(mini, (0,0), (mw-1, mh-1), (0,0,0), 2)

    # 전체 테스트 포인트 (완료 여부 색 구분)
    for pt_num, xm, ym in test_pts:
        mx, my = int(xm*MINI_SCALE), int(ym*MINI_SCALE)
        done = session.count_for_pt(pt_num) > 0
        ok   = session.success_for_pt(pt_num) > 0
        c = (0, 180, 0) if ok else ((80, 80, 200) if done else (180, 180, 180))
        cv2.circle(mini, (mx, my), 4, c, -1)

    # 현재 테스트 포인트 (주황 십자)
    tx, ty = int(true_x*MINI_SCALE), int(true_y*MINI_SCALE)
    cv2.drawMarker(mini, (tx, ty), (0, 100, 255), cv2.MARKER_CROSS, 18, 2)

    # 예측 위치 (파란 원)
    if pred_x is not None and pred_y is not None:
        px = int(np.clip(pred_x, 0, A4_W_MM) * MINI_SCALE)
        py = int(np.clip(pred_y, 0, A4_H_MM) * MINI_SCALE)
        cv2.circle(mini, (px, py), 6, (200, 0, 0), -1)
        cv2.circle(mini, (px, py), 7, (255,255,255), 1)
        # 오차 선
        cv2.line(mini, (tx, ty), (px, py), (0, 0, 180), 1)
        err = np.hypot(pred_x - true_x, pred_y - true_y)
        cv2.putText(mini, f"{err:.1f}mm",
                    (min(px,tx) + 2, min(py,ty) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 160), 1)

    # 높이 맞춤
    if target_h > 0 and mh != target_h:
        scale = target_h / mh
        mini = cv2.resize(mini, (int(mw*scale), target_h))
    return mini
