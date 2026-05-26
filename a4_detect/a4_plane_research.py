#!/usr/bin/env python3
"""
a4_plane_research.py — A4 평면좌표계 탐지 방법 연구 도구

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 0: 시트 생성
  python a4_plane_research.py --gen-sheets
  python a4_plane_research.py --gen-sheets --only eval   # 30점 평가 시트만
  python a4_plane_research.py --gen-sheets --one-point --only edge
  python a4_plane_research.py --gen-sheets --one-point --only composite --combo comp_B_aruco_color
  python a4_plane_research.py --gen-sheets --calib-variants

Step 1: 단일 방법 실시간 테스트
  python a4_plane_research.py --live --method aruco
  python a4_plane_research.py --live --method composite --composite-mode best_of_all

Step 2: 모든 방법 동시 비교 (멀티패널)
  python a4_plane_research.py --compare
  python a4_plane_research.py --compare --methods aruco color_dot edge

Step 3: 저장된 이미지로 벤치마크
  python a4_plane_research.py --benchmark --images ./test_images/

Step 4: 모델 포함 실시간 검증 (A4 + YOLO 탐지)
  python a4_plane_research.py --validate --method aruco --model best.pt
  python a4_plane_research.py --validate --method composite --model best.pt \\
         --check 52.5,74 157.5,74 52.5,223 157.5,223

Step 5: 선행 테스트
  # 5-1. A4 평면 좌표계 검출 방식 전체 비교
  python a4_plane_research.py --precheck --precheck-target a4 --all-methods \\
         --condition level --calib calib_camera0.json

  # 5-2. YOLO 객체탐지만 단독 테스트
  python a4_plane_research.py --precheck --precheck-target object --model best.pt \\
         --object-type pill_cap --condition level --calib calib_camera0.json

  # 5-3. A4 전체 비교 후 YOLO 객체 단독 테스트를 순서대로 실행
  python a4_plane_research.py --precheck --precheck-target suite --model best.pt

  # 5-4. 선택한 A4 방식 + YOLO 통합 상태 확인
  python a4_plane_research.py --precheck --precheck-target both --method aruco --model best.pt \\
         --object-type pill_cap --condition level --calib calib_camera0.json

Step 6: 좌표 오차 측정 실험 (핵심 지표: mm 오차)
  python a4_plane_research.py --eval --method aruco --model best.pt \\
         --object-type pill_cap --one-point --manual --repeats 5 \\
         --condition level --calib calib_camera0.json
  # 같은 pill_cap 모델로 동전/페트병뚜껑/돌멩이 일반화 확인:
  python a4_plane_research.py --eval --method aruco --model best.pt \\
         --object-type coin --expected-class pill_cap --one-point --manual \\
         --condition level --calib calib_camera0.json
  mixed 모드 키: 1=blueberry  2=strawberry  Space=캡처  S=스냅샷

Step 7: 기존 CSV 로그에서 리포트 재생성
  python a4_plane_research.py --report --csv eval_logs/eval_blueberry_aruco_YYYYMMDD.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
방법 목록:
  edge         — A4 외곽선(Canny 에지) 기반
  color_dot    — 색상 마커(빨/초/파/노 원) 기반
  aruco        — ArUco 마커 기반 (DICT_4X4_50)
  grid         — 격자 라인 기반 (Hough 변환)
  composite    — 위 방법 복합 (mode 선택 가능)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _configure_utf8_stdio() -> None:
    """Windows 기본 CP949 콘솔에서도 한글 help 출력이 죽지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


_configure_utf8_stdio()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plane_coord import METHODS, DetectResult, A4_W_MM, A4_H_MM
from plane_coord.camera_calib import maybe_undistort
from plane_coord.composite import CompositeDetector

MINI_SCALE = 2   # 미니맵 스케일: 1mm = 2px

PRECHECK_LOG_DIR = HERE / "precheck_logs"


def _save_precheck_image(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return False
    try:
        path.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True


def _save_precheck_log(tag: str, data: dict, images: list[tuple[str, np.ndarray]] | None = None) -> Path:
    """precheck 결과를 precheck_logs/<tag>_<timestamp>.json 에 저장."""
    PRECHECK_LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PRECHECK_LOG_DIR / f"{tag}_{ts}.json"
    saved_images = []
    if images:
        img_dir = PRECHECK_LOG_DIR / "images"
        seen = set()
        for idx, (label, image) in enumerate(images, start=1):
            if image is None or id(image) in seen:
                continue
            seen.add(id(image))
            safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)[:32] or f"sample{idx}"
            img_path = img_dir / f"{tag}_{ts}_{idx:02d}_{safe_label}.jpg"
            if _save_precheck_image(img_path, image):
                saved_images.append(str(img_path.relative_to(HERE)))
    data["saved_at"] = ts
    if saved_images:
        data["sample_images"] = saved_images
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[precheck] 결과 저장 → {path}")
    for img_path in saved_images:
        print(f"[precheck] sample image -> {HERE / img_path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Mode: --live  단일 방법 실시간 테스트
# ═════════════════════════════════════════════════════════════════════════════

def run_live(
    method_name: str,
    camera_id: int,
    composite_mode: str,
    check_pts: list[tuple[float, float]],
    aruco_marker_size_mm: float = 20.0,
    calib=None,
) -> None:
    """단일 탐지 방법을 실시간으로 테스트."""
    if method_name == "composite":
        detector = CompositeDetector(mode=composite_mode)
    else:
        detector = _make_detector(method_name, aruco_marker_size_mm)

    cap = _open_camera(camera_id)
    print(f"\n[live] 방법: {method_name}  카메라: {camera_id}")
    if calib is not None:
        print(f"[live] 렌즈 왜곡 보정 활성 (rms={calib.rms_px:.4f}px)")
    if check_pts:
        print(f"[live] 검증 포인트: {check_pts}")
    print("[live] 키: q=종료  s=스냅샷  r=결과 초기화\n")

    WIN = f"A4 Research — {method_name}  [q=종료  s=스냅]"
    snap_idx = 0
    fps_buf: deque[float] = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = maybe_undistort(frame, calib)
        result = detector.timed_detect(frame)
        fps_buf.append(result.elapsed_ms)
        avg_ms = sum(fps_buf) / len(fps_buf)

        vis = result.debug_img.copy() if result.debug_img is not None else frame.copy()

        # 상태 오버레이
        ok_color = (0, 220, 80) if result.ok else (0, 60, 220)
        status   = f"OK  repro={result.repro_err_mm:.2f}mm" if result.ok else f"FAIL: {result.note}"
        cv2.putText(vis, f"[{method_name}] {status}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, ok_color, 2)
        cv2.putText(vis, f"{result.elapsed_ms:.1f}ms  avg={avg_ms:.1f}ms",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        # 검증 포인트 오차 표시
        if result.ok and check_pts:
            _overlay_check_errors(vis, result, check_pts)

        mini     = _make_minimap(result, check_pts, vis.shape[0])
        combined = np.hstack([vis, mini])

        cv2.imshow(WIN, combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = HERE / f"snap_{method_name}_{snap_idx:04d}.jpg"
            cv2.imwrite(str(fname), combined)
            print(f"[snap] {fname}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
# Mode: --compare  모든 방법 동시 비교
# ═════════════════════════════════════════════════════════════════════════════

def run_compare(
    methods: list[str],
    camera_id: int,
    aruco_marker_size_mm: float = 20.0,
) -> None:
    """여러 탐지 방법을 멀티패널로 동시 비교."""
    detectors = [(m, _make_detector(m, aruco_marker_size_mm)) for m in methods if m in METHODS]
    if not detectors:
        raise SystemExit("유효한 방법이 없습니다")

    cap = _open_camera(camera_id)
    print(f"\n[compare] 비교 방법: {[m for m,_ in detectors]}")
    print("[compare] 키: q=종료  s=스냅\n")

    WIN   = "A4 Research — Method Comparison  [q=종료]"
    snap_idx = 0
    PANEL_W = 400   # 패널 너비 (px)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        panels = []
        for m_name, det in detectors:
            result   = det.timed_detect(frame)
            vis      = result.debug_img.copy() if result.debug_img is not None else frame.copy()

            # 패널 리사이즈
            scale = PANEL_W / vis.shape[1]
            vis   = cv2.resize(vis, (PANEL_W, int(vis.shape[0] * scale)))

            # 헤더 바
            header = np.zeros((52, PANEL_W, 3), dtype=np.uint8)
            ok_c   = (0, 220, 80) if result.ok else (0, 60, 200)
            cv2.putText(header, m_name, (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, ok_c, 1)
            status = (f"OK  {result.elapsed_ms:.0f}ms  repro={result.repro_err_mm:.1f}mm"
                      if result.ok else f"FAIL {result.elapsed_ms:.0f}ms")
            cv2.putText(header, status, (6, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, ok_c, 1)

            panels.append(np.vstack([header, vis]))

        combined = _tile_panels(panels)
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = HERE / f"compare_{snap_idx:04d}.jpg"
            cv2.imwrite(str(fname), combined)
            print(f"[snap] {fname}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
# Mode: --benchmark  이미지 파일 벤치마크
# ═════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    image_dir: str,
    methods: list[str],
    aruco_marker_size_mm: float = 20.0,
) -> None:
    """이미지 파일 디렉터리에서 각 방법의 성능을 측정."""
    img_dir = Path(image_dir)
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)

    if not images:
        raise SystemExit(f"이미지 없음: {img_dir}")

    valid_methods = [m for m in methods if m in METHODS]
    if not valid_methods:
        raise SystemExit(f"유효한 방법 없음: {methods}")

    detectors = [(m, _make_detector(m, aruco_marker_size_mm)) for m in valid_methods]
    stats: dict[str, dict] = {
        m: {"ok": 0, "fail": 0, "ms_list": [], "repro_list": []}
        for m in valid_methods
    }

    # 헤더
    col_w = 26
    header = "이미지".ljust(25) + "".join(m.ljust(col_w) for m in valid_methods)
    print(f"\n[benchmark] {len(images)}장 × {len(valid_methods)}방법\n")
    print(header)
    print("-" * len(header))

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        row = [img_path.name[:24].ljust(25)]
        for m_name, det in detectors:
            r   = det.timed_detect(frame)
            s   = stats[m_name]
            s["ms_list"].append(r.elapsed_ms)
            if r.ok:
                s["ok"] += 1
                s["repro_list"].append(r.repro_err_mm)
                row.append(f"OK {r.elapsed_ms:5.1f}ms err={r.repro_err_mm:5.2f}mm".ljust(col_w))
            else:
                s["fail"] += 1
                row.append(f"FAIL {r.elapsed_ms:4.1f}ms  {r.note[:10]}".ljust(col_w))
        print("".join(row))

    # 요약
    n = len(images)
    print("\n" + "=" * 80)
    print(f"{'방법':<15} {'성공률':>8} {'평균ms':>8} {'평균repro':>10} {'min/max repro':>16}")
    print("-" * 80)
    for m in valid_methods:
        s    = stats[m]
        ms   = s["ms_list"]
        rep  = s["repro_list"]
        rate = f"{s['ok']}/{n}"
        avg_ms   = sum(ms) / len(ms) if ms else 0
        avg_rep  = sum(rep) / len(rep) if rep else float("nan")
        min_rep  = min(rep) if rep else float("nan")
        max_rep  = max(rep) if rep else float("nan")
        print(f"{m:<15} {rate:>8} {avg_ms:>8.1f} {avg_rep:>10.2f} "
              f"{min_rep:>7.2f}~{max_rep:<7.2f}")
    print("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
# Mode: --validate  A4 좌표계 + YOLO 탐지 검증
# ═════════════════════════════════════════════════════════════════════════════

def run_validate(
    method_name: str,
    model_path: str,
    camera_id: int,
    composite_mode: str,
    check_pts: list[tuple[float, float]],
    conf_thresh: float,
    aruco_marker_size_mm: float = 20.0,
    calib=None,
) -> None:
    """A4 평면좌표계 + YOLO 탐지를 결합한 실시간 검증."""
    try:
        from eval import load_yolo_model
    except ImportError:
        raise SystemExit("ultralytics 필요: pip install ultralytics")

    if method_name == "composite":
        plane_det = CompositeDetector(mode=composite_mode)
    else:
        plane_det = _make_detector(method_name, aruco_marker_size_mm)

    yolo_model = load_yolo_model(model_path)   # validate 는 단일 클래스 모드 없음
    cap        = _open_camera(camera_id)

    print(f"\n[validate] 평면 방법: {method_name}  모델: {model_path}")
    print(f"[validate] 카메라: {camera_id}  conf: {conf_thresh}")
    if calib is not None:
        print(f"[validate] 렌즈 왜곡 보정 활성 (rms={calib.rms_px:.4f}px)")
    if check_pts:
        print(f"[validate] 검증 포인트(mm): {check_pts}")
    print("[validate] 키: q=종료  s=스냅\n")

    WIN      = f"A4 Validate — {method_name} + YOLO  [q=종료]"
    snap_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = maybe_undistort(frame, calib)

        # A4 평면 탐지
        plane_result = plane_det.timed_detect(frame)
        vis = plane_result.debug_img.copy() if plane_result.debug_img is not None else frame.copy()

        a4_x: float | None = None
        a4_y: float | None = None

        if plane_result.ok:
            # YOLO 탐지
            yolo_res = yolo_model.predict(frame, conf=conf_thresh, verbose=False)[0]
            boxes    = yolo_res.boxes

            if boxes:
                box  = max(boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0
                conf  = float(box.conf[0])

                a4_x, a4_y = plane_result.px_to_mm(cx_px, cy_px)

                # 바운딩 박스 + 중심
                cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 220, 30), 2)
                cv2.drawMarker(vis, (int(cx_px), int(cy_px)),
                               (0, 0, 230), cv2.MARKER_CROSS, 24, 2)
                coord_txt = f"A4: ({a4_x:.1f}, {a4_y:.1f}) mm   conf={conf:.2f}"
                cv2.putText(vis, coord_txt, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30, 230, 30), 2)

                if check_pts:
                    dists = [(np.hypot(a4_x - ex, a4_y - ey), (ex, ey))
                             for ex, ey in check_pts]
                    err, near = min(dists)
                    cv2.putText(vis,
                                f"가장 가까운 포인트: ({near[0]:.0f},{near[1]:.0f})mm  오차={err:.1f}mm",
                                (10, vis.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 210, 255), 2)
            else:
                cv2.putText(vis, "YOLO: 탐지 없음", (10, vis.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 220), 2)

        # 평면 상태
        ok_c = (0, 220, 80) if plane_result.ok else (0, 60, 200)
        cv2.putText(vis, f"[{method_name}] {'OK' if plane_result.ok else plane_result.note}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, ok_c, 2)

        mini     = _make_minimap_dot(a4_x, a4_y, check_pts, vis.shape[0])
        combined = np.hstack([vis, mini])

        cv2.imshow(WIN, combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = HERE / f"validate_{snap_idx:04d}.jpg"
            cv2.imwrite(str(fname), combined)
            print(f"[snap] {fname}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
# Mode: --precheck  선행 테스트 (A4 좌표 평면계 + YOLO 탐지 동작 확인)
# ═════════════════════════════════════════════════════════════════════════════

def run_precheck(
    model_path:     str,
    camera_id:      int,
    plane_method:   str,
    composite_mode: str,
    conf_thresh:    float,
    window:         int   = 60,
    aruco_marker_size_mm: float = 20.0,
    calib=None,
    object_type: str = "object",
    condition: str = "unspecified",
) -> None:
    """
    실험 전 선행 테스트.
    - A4 좌표 평면계 계산 가능 여부 (검출 성공률, 재투영 오차)
    - YOLO 객체 탐지 동작 여부 (탐지 성공률, confidence, 클래스)

    --model 생략 시 A4 검출만 확인.
    window 프레임 기준 롤링 통계 + GO / NO-GO 판정.
    키: Q=종료  R=통계 초기화
    """
    from collections import deque, Counter

    # YOLO 로드 (선택)
    yolo_model = None
    if model_path:
        try:
            from eval import load_yolo_model, WORLD_CLASS_MAP
            _wc = WORLD_CLASS_MAP.get(object_type, object_type)
            _is_world = "world" in Path(model_path).stem.lower()
            yolo_model = load_yolo_model(
                model_path, world_classes=[_wc] if _is_world else None
            )
            print(f"[precheck] YOLO 모델: {model_path}")
        except Exception as e:
            print(f"[precheck] YOLO 로드 실패 — {e}")

    if plane_method == "composite":
        detector = CompositeDetector(mode=composite_mode)
    else:
        detector = _make_detector(plane_method, aruco_marker_size_mm)

    cap = _open_camera(camera_id)
    tag  = f"{plane_method}" + (" + YOLO" if yolo_model else "")
    WIN  = f"Pre-Check — {tag}  [Q=종료  R=초기화]"

    # 롤링 버퍼
    a4_buf    = deque(maxlen=window)
    yolo_buf  = deque(maxlen=window)
    repro_buf = deque(maxlen=window)
    conf_buf  = deque(maxlen=window)
    cls_cnt   = Counter()
    frame_n   = 0
    a4_go     = False
    yolo_go   = False
    first_sample = None
    first_ok_sample = None
    last_sample = None

    print(f"[precheck] 카메라: {camera_id}  방법: {plane_method}  조건: {condition}  window: {window}프레임")
    if yolo_model is not None:
        print(f"[precheck] 물체 라벨: {object_type}")
    if calib is not None:
        print(f"[precheck] 렌즈 왜곡 보정 활성 (rms={calib.rms_px:.4f}px)")
    print("[precheck] Q=종료  R=통계 초기화\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = maybe_undistort(frame, calib)
        frame_n += 1

        # ── A4 탐지 ────────────────────────────────────────────────────────────
        result = detector.timed_detect(frame)
        a4_buf.append(result.ok)
        if result.ok and result.repro_err_mm < 1e6:
            repro_buf.append(result.repro_err_mm)

        vis = result.debug_img.copy() if result.debug_img is not None else frame.copy()

        # ── YOLO 탐지 (A4 탐지와 독립 실행) ──────────────────────────────────
        pred_x = pred_y = None
        pred_cls = yolo_conf_val = None

        if yolo_model:
            yolo_res = yolo_model.predict(frame, conf=conf_thresh, verbose=False)[0]
            boxes = yolo_res.boxes
            if boxes:
                box = max(boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0
                yolo_conf_val = float(box.conf[0])
                pred_cls      = yolo_res.names[int(box.cls[0])]
                cls_cnt[pred_cls] += 1

                # 좌표 변환은 A4 탐지 성공 시에만
                if result.ok:
                    pred_x, pred_y = result.px_to_mm(cx_px, cy_px)

                box_color = (30, 220, 30) if result.ok else (0, 180, 220)
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                cv2.drawMarker(vis, (int(cx_px), int(cy_px)),
                               (0, 0, 230), cv2.MARKER_CROSS, 22, 2)
                lbl = f"{pred_cls}  {yolo_conf_val:.2f}"
                if pred_x is not None:
                    lbl += f"  ({pred_x:.1f},{pred_y:.1f})mm"
                else:
                    lbl += "  (A4 FAIL — 좌표 없음)"
                cv2.putText(vis, lbl, (x1, max(y1 - 10, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, box_color, 2)

            yolo_buf.append(bool(boxes))
            if yolo_conf_val:
                conf_buf.append(yolo_conf_val)

        # ── 통계 계산 ──────────────────────────────────────────────────────────
        a4_rate   = sum(a4_buf)   / len(a4_buf)   * 100 if a4_buf   else 0.0
        yolo_rate = sum(yolo_buf) / len(yolo_buf) * 100 if yolo_buf else 0.0
        avg_repro = sum(repro_buf) / len(repro_buf) if repro_buf else float("nan")
        avg_conf  = sum(conf_buf)  / len(conf_buf)  if conf_buf  else float("nan")

        a4_go   = a4_rate   >= 80.0
        yolo_go = yolo_rate >= 70.0

        # ── 상단 현재 프레임 상태 바 ────────────────────────────────────────────
        _draw_precheck_top(vis, result.ok, result.repro_err_mm,
                           pred_cls, yolo_conf_val, pred_x, pred_y)

        # ── 하단 롤링 통계 패널 ─────────────────────────────────────────────────
        _draw_precheck_stats(vis, frame_n, window,
                             a4_rate, a4_go, avg_repro,
                             yolo_rate, yolo_go, avg_conf,
                             cls_cnt, yolo_model is not None)

        # ── 미니맵 ────────────────────────────────────────────────────────────
        mini = _make_minimap(result, [], vis.shape[0])
        if pred_x is not None and pred_y is not None:
            # 미니맵은 종횡비 유지로 리사이즈되므로 단일 스케일 사용
            s  = mini.shape[0] / A4_H_MM
            mx = int(np.clip(pred_x, 0, A4_W_MM) * s)
            my = int(np.clip(pred_y, 0, A4_H_MM) * s)
            cv2.circle(mini, (mx, my), 7, (0, 0, 210), -1)
            cv2.circle(mini, (mx, my), 8, (255, 255, 255), 1)

        display = np.hstack([vis, mini])
        if first_sample is None:
            first_sample = display.copy()
        if result.ok and first_ok_sample is None:
            first_ok_sample = display.copy()
        last_sample = display.copy()

        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            a4_buf.clear(); yolo_buf.clear()
            repro_buf.clear(); conf_buf.clear()
            cls_cnt.clear(); frame_n = 0
            print("[precheck] 통계 초기화")

    cap.release()
    cv2.destroyAllWindows()

    # ── 최종 요약 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  Pre-Check 최종 결과")
    print("=" * 52)
    if a4_buf:
        a4_f = sum(a4_buf) / len(a4_buf) * 100
        go   = "GO" if a4_f >= 80 else "NO-GO"
        print(f"  A4 검출 성공률 : {a4_f:5.1f}%  [{go}]")
        if repro_buf:
            print(f"  재투영 오차 평균: {sum(repro_buf)/len(repro_buf):.3f} mm")
    if yolo_buf and yolo_model:
        yo_f = sum(yolo_buf) / len(yolo_buf) * 100
        go   = "GO" if yo_f >= 70 else "NO-GO"
        print(f"  YOLO 탐지 성공률: {yo_f:5.1f}%  [{go}]")
        if conf_buf:
            print(f"  confidence 평균: {sum(conf_buf)/len(conf_buf):.3f}")
        if cls_cnt:
            print(f"  탐지 클래스    : {dict(cls_cnt.most_common())}")
    overall = (not yolo_model or yolo_go) and a4_go
    print("-" * 52)
    print(f"  종합 판정: {'[ GO ] → eval 실험 시작 가능' if overall else '[NO-GO] → 세팅 확인 필요'}")
    print("=" * 52)

    a4_f = sum(a4_buf) / len(a4_buf) * 100 if a4_buf else 0.0
    yo_f = sum(yolo_buf) / len(yolo_buf) * 100 if yolo_buf else 0.0
    avg_repro = sum(repro_buf) / len(repro_buf) if repro_buf else float("nan")
    avg_conf = sum(conf_buf) / len(conf_buf) if conf_buf else float("nan")
    _save_precheck_log("both" if yolo_model else f"a4_{plane_method}", {
        "mode": "both" if yolo_model else "a4_single",
        "object_type": object_type,
        "condition": condition,
        "model_path": model_path,
        "camera_id": camera_id,
        "plane_method": plane_method,
        "composite_mode": composite_mode,
        "conf_thresh": conf_thresh,
        "window": window,
        "total_frames": frame_n,
        "calib_enabled": calib is not None,
        "a4_success_rate_pct": round(a4_f, 2),
        "a4_judge": "GO" if a4_f >= 80 else "NO-GO",
        "a4_repro_mean_mm": None if math.isnan(avg_repro) else round(avg_repro, 4),
        "yolo_success_rate_pct": round(yo_f, 2) if yolo_model else None,
        "yolo_judge": ("GO" if yo_f >= 70 else "NO-GO") if yolo_model else None,
        "yolo_conf_mean": None if math.isnan(avg_conf) else round(avg_conf, 4),
        "class_counts": dict(cls_cnt.most_common()),
        "overall_judge": "GO" if overall else "NO-GO",
    }, images=[
        ("first", first_sample),
        ("first_ok", first_ok_sample),
        ("last", last_sample),
    ])


def run_precheck_a4_all(
    camera_id: int,
    composite_mode: str,
    window: int = 60,
    aruco_marker_size_mm: float = 20.0,
    calib=None,
    condition: str = "unspecified",
) -> None:
    """모든 A4 평면 좌표계 검출 방식을 같은 카메라 프레임에서 선행 테스트."""
    from collections import deque

    method_order = ["aruco", "color_dot", "edge", "grid", "composite"]
    detectors = []
    for name in method_order:
        if name == "composite":
            detectors.append((name, CompositeDetector(mode=composite_mode)))
        else:
            detectors.append((name, _make_detector(name, aruco_marker_size_mm)))

    stats = {
        name: {
            "ok": deque(maxlen=window),
            "repro": deque(maxlen=window),
            "ms": deque(maxlen=window),
        }
        for name, _ in detectors
    }

    cap = _open_camera(camera_id)
    WIN = "Pre-Check A4 Methods — all  [R=reset  Q=quit]"
    frame_n = 0

    print(f"[precheck:a4] 카메라: {camera_id}  조건: {condition}  methods: {', '.join(method_order)}")
    if calib is not None:
        print(f"[precheck:a4] 렌즈 왜곡 보정 활성 (rms={calib.rms_px:.4f}px)")
    print("[precheck:a4] A4 시트를 카메라 아래에 두고 각 방식의 성공률/재투영 오차를 비교하세요.")
    print("[precheck:a4] R=통계 초기화  Q=종료\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = maybe_undistort(frame, calib)
        frame_n += 1

        panels = []
        for name, detector in detectors:
            result = detector.timed_detect(frame)
            s = stats[name]
            s["ok"].append(result.ok)
            s["ms"].append(result.elapsed_ms)
            if result.ok and result.repro_err_mm < 1e6:
                s["repro"].append(result.repro_err_mm)

            ok_rate = sum(s["ok"]) / len(s["ok"]) * 100 if s["ok"] else 0.0
            avg_repro = sum(s["repro"]) / len(s["repro"]) if s["repro"] else float("nan")
            avg_ms = sum(s["ms"]) / len(s["ms"]) if s["ms"] else 0.0

            vis = result.debug_img.copy() if result.debug_img is not None else frame.copy()
            vis = _resize_panel(vis, width=500)
            _draw_a4_method_panel(vis, name, result, ok_rate, avg_repro, avg_ms, window)
            panels.append(vis)

        grid = _tile_panels(panels)
        cv2.putText(grid, f"A4 method precheck | frames={frame_n} window={window}",
                    (12, grid.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        cv2.imshow(WIN, grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            for s in stats.values():
                s["ok"].clear()
                s["repro"].clear()
                s["ms"].clear()
            frame_n = 0
            print("[precheck:a4] 통계 초기화")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 72)
    print("  A4 평면 좌표계 방식별 선행 테스트 요약")
    print("=" * 72)
    print(f"{'method':<14} {'success':>9} {'avg_repro(mm)':>15} {'avg_ms':>9} {'judge':>8}")
    print("-" * 72)
    method_results = []
    for name, _ in detectors:
        s = stats[name]
        ok_rate = sum(s["ok"]) / len(s["ok"]) * 100 if s["ok"] else 0.0
        avg_repro = sum(s["repro"]) / len(s["repro"]) if s["repro"] else float("nan")
        avg_ms = sum(s["ms"]) / len(s["ms"]) if s["ms"] else 0.0
        judge = "GO" if ok_rate >= 80.0 else "NO-GO"
        repro_text = f"{avg_repro:>15.3f}" if not math.isnan(avg_repro) else f"{'-':>15}"
        print(f"{name:<14} {ok_rate:>8.1f}% {repro_text} {avg_ms:>9.1f} {judge:>8}")
        method_results.append({
            "method": name,
            "ok_rate_pct": round(ok_rate, 2),
            "avg_repro_mm": None if math.isnan(avg_repro) else round(avg_repro, 4),
            "avg_ms": round(avg_ms, 2),
            "judge": judge,
            "n_frames": len(s["ok"]),
        })
    print("=" * 72)

    _save_precheck_log("a4_all", {
        "mode": "a4_all",
        "condition": condition,
        "camera_id": camera_id,
        "composite_mode": composite_mode,
        "window": window,
        "total_frames": frame_n,
        "calib_enabled": calib is not None,
        "methods": method_results,
    })


def run_precheck_object_only(
    model_path: str,
    camera_id: int,
    conf_thresh: float,
    window: int = 60,
    calib=None,
    object_type: str = "object",
    condition: str = "unspecified",
) -> None:
    """A4 좌표계 없이 YOLO 객체탐지만 단독 선행 테스트."""
    from collections import Counter, deque

    if not model_path:
        raise SystemExit("--precheck-target object 사용 시 --model MODEL.pt 가 필요합니다.")

    try:
        from eval import load_yolo_model, WORLD_CLASS_MAP
    except ImportError:
        raise SystemExit("ultralytics 필요: pip install ultralytics")

    _wc = WORLD_CLASS_MAP.get(object_type, object_type)
    _is_world = "world" in Path(model_path).stem.lower()
    model = load_yolo_model(model_path, world_classes=[_wc] if _is_world else None)
    cap = _open_camera(camera_id)

    det_buf = deque(maxlen=window)
    conf_buf = deque(maxlen=window)
    cls_cnt = Counter()
    frame_n = 0
    snap_idx = 0
    first_sample = None
    first_detected_sample = None
    last_sample = None
    WIN = "Pre-Check Object Detection — YOLO only  [S=snap  R=reset  Q=quit]"

    print(f"[precheck:object] 모델: {model_path}")
    print(f"[precheck:object] 물체 라벨: {object_type}  조건: {condition}")
    print(f"[precheck:object] 카메라: {camera_id}  conf: {conf_thresh}  window: {window}")
    if calib is not None:
        print(f"[precheck:object] 렌즈 왜곡 보정 활성 (rms={calib.rms_px:.4f}px)")
    print("[precheck:object] 객체탐지만 확인합니다. A4 검출/좌표 변환은 수행하지 않습니다.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = maybe_undistort(frame, calib)
        frame_n += 1

        vis = frame.copy()
        yolo_res = model.predict(frame, conf=conf_thresh, verbose=False)[0]
        boxes = yolo_res.boxes
        detected = bool(boxes)
        det_buf.append(detected)

        best_info = None
        if boxes:
            best = max(boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = (int(v) for v in best.xyxy[0])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            conf = float(best.conf[0])
            cls_name = yolo_res.names[int(best.cls[0])]
            conf_buf.append(conf)
            cls_cnt[cls_name] += 1
            best_info = (cls_name, conf, cx, cy)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 220, 30), 2)
            cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 230), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(vis, f"{cls_name} {conf:.2f}  px=({cx:.0f},{cy:.0f})",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30, 220, 30), 2)

        det_rate = sum(det_buf) / len(det_buf) * 100 if det_buf else 0.0
        avg_conf = sum(conf_buf) / len(conf_buf) if conf_buf else float("nan")
        _draw_object_precheck_panel(
            vis, frame_n, window, det_rate, avg_conf, cls_cnt, best_info
        )
        if first_sample is None:
            first_sample = vis.copy()
        if detected and first_detected_sample is None:
            first_detected_sample = vis.copy()
        last_sample = vis.copy()

        cv2.imshow(WIN, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            det_buf.clear()
            conf_buf.clear()
            cls_cnt.clear()
            frame_n = 0
            print("[precheck:object] 통계 초기화")
        if key == ord("s"):
            fname = HERE / f"precheck_object_{snap_idx:04d}.jpg"
            cv2.imwrite(str(fname), vis)
            print(f"[snap] {fname}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 52)
    print("  YOLO 객체탐지 단독 선행 테스트 요약")
    print("=" * 52)
    det_rate = sum(det_buf) / len(det_buf) * 100 if det_buf else 0.0
    avg_conf = sum(conf_buf) / len(conf_buf) if conf_buf else float("nan")
    if det_buf:
        print(f"  탐지 성공률     : {det_rate:5.1f}%  [{'GO' if det_rate >= 70 else 'NO-GO'}]")
    if conf_buf:
        print(f"  confidence 평균: {avg_conf:.3f}")
    if cls_cnt:
        print(f"  탐지 클래스    : {dict(cls_cnt.most_common())}")
    print("=" * 52)

    _save_precheck_log("object", {
        "mode": "object",
        "object_type": object_type,
        "condition": condition,
        "model_path": model_path,
        "camera_id": camera_id,
        "conf_thresh": conf_thresh,
        "window": window,
        "total_frames": frame_n,
        "calib_enabled": calib is not None,
        "det_rate_pct": round(det_rate, 2),
        "judge": "GO" if det_rate >= 70 else "NO-GO",
        "avg_conf": None if math.isnan(avg_conf) else round(avg_conf, 4),
        "class_counts": dict(cls_cnt.most_common()),
    }, images=[
        ("first", first_sample),
        ("first_detected", first_detected_sample),
        ("last", last_sample),
    ])


def _draw_precheck_top(
    vis: np.ndarray,
    a4_ok: bool,
    repro_mm: float,
    pred_cls,
    yolo_conf,
    pred_x,
    pred_y,
) -> None:
    """상단 70px: 현재 프레임 A4 + YOLO 상태."""
    w = vis.shape[1]
    bar = np.zeros((70, w, 3), dtype=np.uint8)

    a4_c = (0, 220, 80) if a4_ok else (0, 60, 200)
    a4_t = f"A4: OK  repro={repro_mm:.2f}mm" if a4_ok else "A4: FAIL — A4 시트를 카메라 앞에"
    cv2.putText(bar, a4_t, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, a4_c, 2)

    if pred_cls is not None:
        yo_t = f"YOLO: {pred_cls}  conf={yolo_conf:.2f}"
        if pred_x is not None:
            yo_t += f"  → ({pred_x:.1f}, {pred_y:.1f}) mm"
        cv2.putText(bar, yo_t, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 220, 30), 1)
    else:
        cv2.putText(bar, "YOLO: 탐지 없음 — 객체를 A4 위에 올려놓으세요",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 100, 100), 1)

    vis[:70] = bar


def _draw_precheck_stats(
    vis: np.ndarray,
    frame_n: int,
    window: int,
    a4_rate: float,
    a4_go: bool,
    avg_repro: float,
    yolo_rate: float,
    yolo_go: bool,
    avg_conf: float,
    cls_cnt,
    has_yolo: bool,
) -> None:
    """하단 88px: 롤링 통계 + GO/NO-GO 바."""
    h, w = vis.shape[:2]
    panel = np.full((88, w, 3), 28, dtype=np.uint8)
    half  = w // 2

    # ── A4 진행 바 ────────────────────────────────────────
    a4_c  = (0, 200, 60) if a4_go else (30, 30, 220)
    bw    = half - 20
    fill  = int(bw * min(a4_rate, 100) / 100)
    cv2.rectangle(panel, (10, 8),  (10 + bw, 32), (70, 70, 70), -1)
    cv2.rectangle(panel, (10, 8),  (10 + fill, 32), a4_c, -1)
    cv2.putText(panel, f"A4 {a4_rate:.0f}%  {'[  GO  ]' if a4_go else '[NO-GO]'}",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    repro_t = f"재투영 오차 평균: {avg_repro:.2f} mm" if not math.isnan(avg_repro) else "재투영 오차: -"
    cv2.putText(panel, repro_t, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

    # ── YOLO 진행 바 ──────────────────────────────────────
    if has_yolo:
        yo_c  = (0, 200, 60) if yolo_go else (30, 30, 220)
        ox    = half + 10
        fill2 = int(bw * min(yolo_rate, 100) / 100)
        cv2.rectangle(panel, (ox, 8),  (ox + bw, 32), (70, 70, 70), -1)
        cv2.rectangle(panel, (ox, 8),  (ox + fill2, 32), yo_c, -1)
        cv2.putText(panel, f"YOLO {yolo_rate:.0f}%  {'[  GO  ]' if yolo_go else '[NO-GO]'}",
                    (ox + 4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        conf_t = f"conf 평균: {avg_conf:.3f}" if not math.isnan(avg_conf) else "conf: -"
        cv2.putText(panel, conf_t, (ox, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)
        if cls_cnt:
            cls_t = "  ".join(f"{k}: {v}" for k, v in cls_cnt.most_common())
            cv2.putText(panel, cls_t, (ox, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 200, 120), 1)

    cv2.putText(panel, f"frames: {frame_n}  window: {window}  [R=초기화  Q=종료]",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)

    vis[h - 88:h] = panel


def _resize_panel(img: np.ndarray, width: int = 500) -> np.ndarray:
    """멀티패널 표시용 폭 기준 리사이즈."""
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / max(w, 1)
    return cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _draw_a4_method_panel(
    vis: np.ndarray,
    method_name: str,
    result: DetectResult,
    ok_rate: float,
    avg_repro: float,
    avg_ms: float,
    window: int,
) -> None:
    """A4 방식 전체 선행 테스트 패널 상단에 통계 오버레이."""
    h, w = vis.shape[:2]
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 76), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)

    ok_color = (0, 220, 80) if result.ok else (0, 60, 220)
    go = ok_rate >= 80.0
    go_color = (0, 220, 80) if go else (0, 60, 220)
    repro_text = f"{avg_repro:.2f}mm" if not math.isnan(avg_repro) else "-"

    cv2.putText(vis, method_name, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, ok_color, 2)
    cv2.putText(vis, f"now: {'OK' if result.ok else 'FAIL'}  {result.elapsed_ms:.1f}ms",
                (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)
    cv2.putText(vis, f"rate {ok_rate:.0f}%  repro {repro_text}  avg {avg_ms:.1f}ms",
                (10, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.42, go_color, 1)

    cv2.putText(vis, f"W{window}", (w - 52, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)


def _draw_object_precheck_panel(
    vis: np.ndarray,
    frame_n: int,
    window: int,
    det_rate: float,
    avg_conf: float,
    cls_cnt,
    best_info,
) -> None:
    """YOLO 객체 단독 선행 테스트용 상태/통계 오버레이."""
    h, w = vis.shape[:2]

    top = vis.copy()
    cv2.rectangle(top, (0, 0), (w, 72), (0, 0, 0), -1)
    cv2.addWeighted(top, 0.70, vis, 0.30, 0, vis)

    if best_info is not None:
        cls_name, conf, cx, cy = best_info
        now_text = f"YOLO: {cls_name} conf={conf:.2f} center_px=({cx:.0f},{cy:.0f})"
        color = (0, 220, 80)
    else:
        now_text = "YOLO: no object detected"
        color = (0, 60, 220)

    cv2.putText(vis, now_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
    conf_text = f"{avg_conf:.3f}" if not math.isnan(avg_conf) else "-"
    cv2.putText(vis, f"frames={frame_n} window={window}  detect_rate={det_rate:.0f}%  avg_conf={conf_text}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

    panel = vis.copy()
    cv2.rectangle(panel, (0, h - 72), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.70, vis, 0.30, 0, vis)
    bw = max(1, w - 28)
    fill = int(bw * min(det_rate, 100.0) / 100.0)
    bar_color = (0, 200, 60) if det_rate >= 70.0 else (30, 30, 220)
    cv2.rectangle(vis, (14, h - 58), (14 + bw, h - 34), (70, 70, 70), -1)
    cv2.rectangle(vis, (14, h - 58), (14 + fill, h - 34), bar_color, -1)
    cv2.putText(vis, f"{'GO' if det_rate >= 70.0 else 'NO-GO'}  [S=snap  R=reset  Q=quit]",
                (18, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    if cls_cnt:
        cls_text = "  ".join(f"{k}:{v}" for k, v in cls_cnt.most_common())
        cv2.putText(vis, cls_text, (14, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 220, 120), 1)


# ═════════════════════════════════════════════════════════════════════════════
# Mode: --report  CSV 로그에서 리포트 재생성
# ═════════════════════════════════════════════════════════════════════════════

def run_report(csv_path: str, save_json: bool = True) -> None:
    """기존 CSV 파일을 읽어 좌표 오차 리포트를 출력한다."""
    from eval.session import EvalSession
    from eval.report  import compute_report, print_report, save_report_json

    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(f"파일 없음: {path}")

    samples = EvalSession.load_csv(path)
    if not samples:
        raise SystemExit(f"샘플 없음: {path}")

    print(f"[report] {len(samples)}개 샘플 로드: {path}")
    report = compute_report(samples)
    print_report(report)

    if save_json:
        save_report_json(report, path.with_suffix(".json"))


# ═════════════════════════════════════════════════════════════════════════════
# 헬퍼
# ═════════════════════════════════════════════════════════════════════════════

def _open_camera(camera_id: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {camera_id}를 열 수 없습니다")
    return cap


def _make_minimap(
    result: DetectResult,
    check_pts: list[tuple[float, float]],
    target_h: int,
) -> np.ndarray:
    """A4 미니맵 이미지 생성."""
    mw = int(A4_W_MM * MINI_SCALE)
    mh = int(A4_H_MM * MINI_SCALE)
    mini = np.full((mh, mw, 3), 245, dtype=np.uint8)

    # 50mm 격자
    for x in range(0, int(A4_W_MM) + 1, 50):
        cv2.line(mini, (x * MINI_SCALE, 0), (x * MINI_SCALE, mh), (185, 185, 185), 1)
        cv2.putText(mini, str(x), (x * MINI_SCALE + 1, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1)
    for y in range(0, int(A4_H_MM) + 1, 50):
        cv2.line(mini, (0, y * MINI_SCALE), (mw, y * MINI_SCALE), (185, 185, 185), 1)
        cv2.putText(mini, str(y), (2, y * MINI_SCALE + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1)
    cv2.rectangle(mini, (0, 0), (mw - 1, mh - 1), (0, 0, 0), 2)

    # 검증 포인트 (주황 십자)
    for cx, cy in check_pts:
        mx, my = int(cx * MINI_SCALE), int(cy * MINI_SCALE)
        cv2.drawMarker(mini, (mx, my), (0, 100, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(mini, f"({cx:.0f},{cy:.0f})", (mx + 4, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 80, 200), 1)

    # 탐지된 대응점 (초록 점)
    if result.ok and result.ref_pts_px is not None and result.ref_pts_mm is not None:
        for px_pt, mm_pt in zip(result.ref_pts_px, result.ref_pts_mm):
            xm = int(np.clip(mm_pt[0], 0, A4_W_MM) * MINI_SCALE)
            ym = int(np.clip(mm_pt[1], 0, A4_H_MM) * MINI_SCALE)
            cv2.circle(mini, (xm, ym), 4, (0, 180, 0), -1)

    # 방법 라벨
    label = result.method_name or "?"
    ok_c  = (0, 180, 60) if result.ok else (0, 0, 200)
    cv2.putText(mini, label, (4, mh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, ok_c, 1)

    # 높이 맞춤
    if target_h > 0 and mh != target_h:
        scale = target_h / mh
        mini  = cv2.resize(mini, (int(mw * scale), target_h),
                           interpolation=cv2.INTER_LINEAR)
    return mini


def _make_minimap_dot(
    a4_x: float | None,
    a4_y: float | None,
    check_pts: list[tuple[float, float]],
    target_h: int,
) -> np.ndarray:
    """validate 모드용 미니맵 (탐지 점 1개)."""
    mw = int(A4_W_MM * MINI_SCALE)
    mh = int(A4_H_MM * MINI_SCALE)
    mini = np.full((mh, mw, 3), 245, dtype=np.uint8)

    for x in range(0, int(A4_W_MM) + 1, 50):
        cv2.line(mini, (x * MINI_SCALE, 0), (x * MINI_SCALE, mh), (185, 185, 185), 1)
    for y in range(0, int(A4_H_MM) + 1, 50):
        cv2.line(mini, (0, y * MINI_SCALE), (mw, y * MINI_SCALE), (185, 185, 185), 1)
    cv2.rectangle(mini, (0, 0), (mw - 1, mh - 1), (0, 0, 0), 2)

    for cx, cy in check_pts:
        mx, my = int(cx * MINI_SCALE), int(cy * MINI_SCALE)
        cv2.drawMarker(mini, (mx, my), (0, 100, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(mini, f"({cx:.0f},{cy:.0f})", (mx + 4, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 80, 200), 1)

    if a4_x is not None and a4_y is not None:
        mx = int(np.clip(a4_x, 0, A4_W_MM) * MINI_SCALE)
        my = int(np.clip(a4_y, 0, A4_H_MM) * MINI_SCALE)
        cv2.circle(mini, (mx, my), 7, (0, 0, 210), -1)
        cv2.circle(mini, (mx, my), 8, (255, 255, 255), 1)
        cv2.putText(mini, f"({a4_x:.1f},{a4_y:.1f})", (mx + 9, my + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 180), 1)

    if target_h > 0 and mh != target_h:
        scale = target_h / mh
        mini  = cv2.resize(mini, (int(mw * scale), target_h))
    return mini


def _overlay_check_errors(
    vis: np.ndarray,
    result: DetectResult,
    check_pts: list[tuple[float, float]],
) -> None:
    """검증 포인트를 역변환하여 카메라 화면에 오버레이."""
    h, w = vis.shape[:2]
    for cx_mm, cy_mm in check_pts:
        px_x, px_y = result.mm_to_px(cx_mm, cy_mm)
        if 0 <= px_x < w and 0 <= px_y < h:
            cv2.drawMarker(vis, (int(px_x), int(px_y)),
                           (0, 120, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(vis, f"({cx_mm:.0f},{cy_mm:.0f})",
                        (int(px_x) + 8, int(px_y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 220), 1)


def _tile_panels(panels: list[np.ndarray]) -> np.ndarray:
    """패널 리스트를 가로로 타일 배치."""
    if not panels:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    n    = len(panels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    h    = max(p.shape[0] for p in panels)
    w    = max(p.shape[1] for p in panels)
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        ph, pw = p.shape[:2]
        grid[r*h : r*h+ph, c*w : c*w+pw] = p
    return grid


def _parse_check_pts(raw: list[str]) -> list[tuple[float, float]]:
    pts = []
    for s in raw:
        parts = s.split(",")
        if len(parts) != 2:
            raise SystemExit(f"잘못된 포인트 형식: '{s}'  예) 52.5,74")
        pts.append((float(parts[0]), float(parts[1])))
    return pts


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _make_detector(method_name: str, aruco_marker_size_mm: float = 20.0):
    """탐지기 인스턴스 생성. aruco 일 때만 marker_size_mm 을 적용."""
    if method_name == "aruco":
        from plane_coord.aruco import ArucoDetector
        return ArucoDetector(marker_size_mm=aruco_marker_size_mm)
    return METHODS[method_name]()


def main() -> None:
    p = argparse.ArgumentParser(
        description="A4 평면좌표계 탐지 방법 연구 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 실행 모드
    mode_g = p.add_mutually_exclusive_group(required=True)
    mode_g.add_argument("--gen-sheets", action="store_true",
                        help="모든 방법용 테스트 시트 생성")
    mode_g.add_argument("--live",       action="store_true",
                        help="단일 방법 실시간 테스트")
    mode_g.add_argument("--compare",    action="store_true",
                        help="여러 방법 멀티패널 동시 비교")
    mode_g.add_argument("--benchmark",  action="store_true",
                        help="이미지 파일 디렉터리 벤치마크")
    mode_g.add_argument("--validate",   action="store_true",
                        help="A4 좌표계 + YOLO 탐지 실시간 검증")
    mode_g.add_argument("--precheck",   action="store_true",
                        help="선행 테스트: A4 좌표 평면계 + YOLO 탐지 동작 확인")
    mode_g.add_argument("--eval",       action="store_true",
                        help="좌표 오차 측정 실험 (핵심 지표: mm 오차)")
    mode_g.add_argument("--report",     action="store_true",
                        help="기존 CSV 로그에서 오차 리포트 재생성")

    # 공통 옵션
    p.add_argument("--method", default="aruco",
                   choices=list(METHODS.keys()),
                   help="탐지 방법 (기본: aruco)")
    p.add_argument("--methods", nargs="+",
                   default=["edge", "color_dot", "aruco", "grid"],
                   help="--compare / --benchmark 용 방법 목록")
    p.add_argument("--camera", type=int, default=0,
                   help="카메라 장치 ID (기본: 0)")
    p.add_argument("--model",  default="",
                   help="YOLO 모델 경로 (.pt) — --precheck / --validate / --eval")
    p.add_argument("--conf",   type=float, default=0.30,
                   help="YOLO 신뢰도 임계값 (기본: 0.30)")
    p.add_argument("--check", nargs="*", default=[], metavar="X,Y",
                   help="검증 포인트(mm) 예: --check 52.5,74 157.5,74")
    p.add_argument("--composite-mode", default="priority_fallback",
                   choices=["priority_fallback", "best_of_all", "vote_H"],
                   help="composite 탐지기 모드 (기본: priority_fallback)")
    p.add_argument("--images", default="./test_images",
                   help="--benchmark 용 이미지 디렉터리")
    p.add_argument("--out-dir", default=None,
                   help="--gen-sheets 출력 디렉터리 (기본: sheets/output)")
    p.add_argument("--only", default=None,
                   choices=["edge", "color_dot", "aruco",
                            "grid", "composite", "eval", "eval_one_point",
                            "calib_checkerboard"],
                   help="--gen-sheets 에서 특정 시트만 생성 "
                        "(calib_checkerboard=카메라 왜곡 보정용 체커보드 PDF)")
    p.add_argument("--one-point", action="store_true",
                   help="--gen-sheets: 좌표 실험용 한 장당 표시점 1개 시트 묶음 생성. "
                        "--eval: QUICK_TEST_PTS(5점) 사용 (1점 시트 출력물 전용)")
    p.add_argument("--combo", default="all",
                   choices=["all", "comp_A_aruco", "comp_B_aruco_color",
                            "comp_C_aruco_grid", "comp_D_full"],
                   help="--one-point --only composite 에서 생성할 복합 조합")
    p.add_argument("--calib-variants", action="store_true",
                   help="--gen-sheets: ArUco/색상점 크기와 위치 변형 테스트 시트 생성")
    p.add_argument("--calib-sheet", action="store_true",
                   help="--gen-sheets: 카메라 렌즈 왜곡 보정용 체커보드 PDF 추가 생성")

    # --eval 전용
    p.add_argument("--manual", action="store_true",
                   help=("수동 진행 모드 (1점 시트 / 용지 교체 실험). "
                         "캡처 완료 후 자동으로 다음 포인트로 넘어가지 않고 "
                         "[N] 키 입력 시에만 진행."))
    p.add_argument("--object-type", default="cap",
                   metavar="TYPE",
                   help=("측정 객체 종류 (기본: cap). "
                         "예: cap | coin | bottle_cap | blueberry | strawberry | mixed. "
                         "mixed = 두 클래스 혼합 배치, 1/2 키로 현재 객체 지정"))
    p.add_argument("--expected-class", default="",
                   metavar="CLASS",
                   help=("YOLO가 반환해야 하는 클래스명. 생략하면 object-type과 동일하게 사용. "
                         "YOLO-World(yoloworld_s.pt) 사용 시 set_classes() 에도 적용됨. "
                         "예: 동전을 OI7 모델로 테스트할 때 --expected-class Coin, "
                         "돌멩이를 YOLO-World로 테스트할 때 --expected-class rock"))
    p.add_argument("--repeats", type=int, default=3,
                   help="포인트당 캡처 반복 횟수 (기본: 3)")
    p.add_argument("--log-dir", default="./eval_logs",
                   metavar="DIR",
                   help="CSV/JSON 로그 저장 디렉터리 (기본: ./eval_logs)")
    p.add_argument("--condition", default="unspecified",
                   metavar="LABEL",
                   help=("실험 조건 라벨. 예: level, tilt_low, tilt_mid, tilt_high. "
                         "CSV/JSON 리포트와 파일명에 저장됩니다."))
    p.add_argument("--repro-thresh", type=float, default=3.0,
                   metavar="MM",
                   help=("H 품질 게이트 — repro_err 가 이 값(mm)을 넘는 프레임의 H를 폐기. "
                         "0이면 게이트 비활성 (기본: 3.0)"))

    # 카메라 렌즈 왜곡 보정 (calibrate_camera.py 로 생성)
    p.add_argument("--calib", default=None,
                   metavar="FILE",
                   help=("카메라 캘리브레이션 JSON 파일 경로. "
                         "calibrate_camera.py 로 생성한 calib_camera<N>.json. "
                         "지정하면 매 프레임 왜곡 보정 후 A4 검출을 수행합니다."))

    # ArUco 마커 크기 (aruco.py marker_size_mm 과 반드시 일치)
    p.add_argument("--aruco-marker-size", type=float, default=20.0,
                   metavar="MM",
                   help=("출력물에 프린트된 ArUco 마커 한 변 길이 mm. "
                         "gen.py 의 ARUCO_SIZE_VARIANTS_MM=[16,20,24] 중 하나. "
                         "(기본: 20.0)"))

    # --precheck 전용
    p.add_argument("--precheck-target", default="auto",
                   choices=["auto", "a4", "object", "both", "suite"],
                   help=("선행 테스트 대상: auto=옵션에 따라 자동, "
                         "a4=A4 평면만, object=YOLO 객체만, "
                         "both=선택 A4 방식+YOLO, suite=A4 전체 후 YOLO 객체"))
    p.add_argument("--all-methods", action="store_true",
                   help="--precheck-target a4 에서 모든 A4 검출 방식을 동시에 비교")
    p.add_argument("--window", type=int, default=60,
                   help="--precheck 롤링 통계 window 프레임 수 (기본: 60)")

    # --report 전용
    p.add_argument("--csv", default="",
                   metavar="PATH",
                   help="--report 에서 읽을 CSV 파일 경로")
    p.add_argument("--no-json", action="store_true",
                   help="--report 시 JSON 저장 건너뜀")

    args = p.parse_args()

    # ── 카메라 캘리브레이션 로딩 ───────────────────────────────────────────────
    _calib = None
    if args.calib:
        from plane_coord.camera_calib import CameraCalib
        _calib = CameraCalib.load(args.calib)
        print(f"[calib] {_calib.summary()}")

    if args.gen_sheets:
        from sheets.gen import gen_all_sheets
        out = Path(args.out_dir) if args.out_dir else None
        only = None if args.only == "eval_one_point" else args.only
        gen_all_sheets(
            out_dir=out,
            only=only,
            one_point=args.one_point or args.only == "eval_one_point",
            combo_key=args.combo,
            calib_variants=args.calib_variants,
            calib_sheet=args.calib_sheet,
        )

    elif args.live:
        check_pts = _parse_check_pts(args.check)
        run_live(args.method, args.camera, args.composite_mode, check_pts,
                 aruco_marker_size_mm=args.aruco_marker_size, calib=_calib)

    elif args.compare:
        run_compare(args.methods, args.camera,
                    aruco_marker_size_mm=args.aruco_marker_size)

    elif args.benchmark:
        run_benchmark(args.images, args.methods,
                      aruco_marker_size_mm=args.aruco_marker_size)

    elif args.precheck:
        target = args.precheck_target
        if target == "auto":
            if args.all_methods:
                target = "a4"
            elif args.model:
                target = "both"
            else:
                target = "a4"

        if target == "suite":
            run_precheck_a4_all(args.camera, args.composite_mode, args.window,
                                aruco_marker_size_mm=args.aruco_marker_size, calib=_calib,
                                condition=args.condition)
            if args.model:
                run_precheck_object_only(args.model, args.camera, args.conf, args.window,
                                         calib=_calib, object_type=args.object_type,
                                         condition=args.condition)
            else:
                print("[precheck:suite] --model 이 없어 YOLO 객체 단독 테스트는 건너뜁니다.")

        elif target == "object":
            if not args.model:
                p.error("--precheck-target object 사용 시 --model MODEL.pt 필수")
            run_precheck_object_only(args.model, args.camera, args.conf, args.window,
                                     calib=_calib, object_type=args.object_type,
                                     condition=args.condition)

        elif target == "a4":
            if args.all_methods:
                run_precheck_a4_all(args.camera, args.composite_mode, args.window,
                                    aruco_marker_size_mm=args.aruco_marker_size, calib=_calib,
                                    condition=args.condition)
            else:
                run_precheck("", args.camera, args.method,
                             args.composite_mode, args.conf, args.window,
                             aruco_marker_size_mm=args.aruco_marker_size, calib=_calib,
                             object_type=args.object_type, condition=args.condition)

        elif target == "both":
            if args.all_methods:
                p.error("--precheck-target both 에서는 --all-methods 대신 --method 로 좌표 방식을 하나 선택하세요")
            if not args.model:
                p.error("--precheck-target both 사용 시 --model MODEL.pt 필수")
            run_precheck(args.model, args.camera, args.method,
                         args.composite_mode, args.conf, args.window,
                         aruco_marker_size_mm=args.aruco_marker_size, calib=_calib,
                         object_type=args.object_type, condition=args.condition)

    elif args.validate:
        if not args.model:
            p.error("--validate 사용 시 --model MODEL.pt 필수")
        check_pts = _parse_check_pts(args.check)
        run_validate(args.method, args.model, args.camera,
                     args.composite_mode, check_pts, args.conf,
                     aruco_marker_size_mm=args.aruco_marker_size, calib=_calib)

    elif args.eval:
        if not args.model:
            p.error("--eval 사용 시 --model MODEL.pt 필수")
        from eval.runner import run_eval
        if args.one_point:
            # 1점 시트(one-point 모드 출력물) 전용 — QUICK_TEST_PTS(5점) 사용
            # 좌표: ①중앙(105,148.5) ②좌상(60,65) ③우상(150,65) ④좌하(60,232) ⑤우하(150,232)
            from sheets.gen import QUICK_TEST_PTS
            _test_pts = QUICK_TEST_PTS
            print(f"[eval] --one-point 모드: QUICK_TEST_PTS {len(_test_pts)}점 사용")
        else:
            from eval import EVAL_TEST_PTS
            _test_pts = EVAL_TEST_PTS
            print(f"[eval] 표준 모드: EVAL_TEST_PTS {len(_test_pts)}점 사용")
        run_eval(
            object_type          = args.object_type,
            model_path           = args.model,
            camera_id            = args.camera,
            plane_method         = args.method,
            composite_mode       = args.composite_mode,
            test_pts             = _test_pts,
            repeats              = args.repeats,
            conf_thresh          = args.conf,
            log_dir              = Path(args.log_dir),
            condition            = args.condition,
            expected_class       = args.expected_class or None,
            manual_advance       = args.manual,
            calib                = _calib,
            aruco_marker_size_mm = args.aruco_marker_size,
            repro_err_thresh_mm  = args.repro_thresh,
        )

    elif args.report:
        if not args.csv:
            p.error("--report 사용 시 --csv CSV_PATH 필수")
        run_report(args.csv, save_json=not args.no_json)


if __name__ == "__main__":
    main()
