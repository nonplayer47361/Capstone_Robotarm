#!/usr/bin/env python3
"""
calibrate_camera.py — 체커보드 기반 카메라 캘리브레이션 도구

체커보드 인쇄물을 카메라 앞에서 다양한 각도로 촬영해
렌즈 왜곡 파라미터를 계산하고 JSON 파일로 저장합니다.

팀원별로 이 파일을 한 번 실행해 자신의 카메라 캘리브레이션 파일을 만들고,
이후 a4_plane_research.py 에 --calib 옵션으로 전달하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: 체커보드 시트 생성 (아직 없다면)
  python calibrate_camera.py --gen-sheet

Step 2: 라이브 캡처 모드 — 카메라 앞에서 다양한 각도로 체커보드 촬영
  python calibrate_camera.py --capture
  python calibrate_camera.py --capture --camera 1   # 카메라 ID 지정
  python calibrate_camera.py --capture --out-dir calib_images/

  키: [S] 스냅 / [C] 자동 스냅 ON-OFF / [Q] 종료 + 캘리브레이션 실행

Step 3: 저장된 이미지 파일로 캘리브레이션만 실행
  python calibrate_camera.py --calibrate --images calib_images/
  python calibrate_camera.py --calibrate --images calib_images/ --grid 9x6 --square 25

Step 4: 캘리브레이션 결과 확인 + 왜곡 보정 미리보기
  python calibrate_camera.py --preview --calib calib_camera0.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

체커보드 파라미터 기본값:
  --grid 9x6    : 내부 코너 수 (가로 × 세로)  — 흰 칸+검은 칸 경계점
  --square 25   : 체커 한 칸의 실제 크기 (mm)

체커보드 인쇄 주의:
  - 반드시 100% 실제 크기로 출력
  - 구겨지지 않는 딱딱한 판에 붙이거나 클립보드 등에 고정
  - 최소 15~20장 이상, 다양한 기울기와 거리에서 촬영
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from plane_coord.camera_calib import CameraCalib


# ── 기본 파라미터 ─────────────────────────────────────────────────────────────
DEFAULT_GRID   = (9, 6)      # 내부 코너 수 (cols, rows)
DEFAULT_SQUARE = 25.0        # 체커 한 칸 크기 (mm)
DEFAULT_CAMERA = 0
DEFAULT_OUT_DIR = _HERE / "calib_images"
DEFAULT_CALIB_OUT = _HERE / "calib_camera{camera_id}.json"

# 서브픽셀 정밀도 조건
_SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
)
# findChessboardCorners 플래그 (run_capture / run_calibrate 공용)
_CB_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _imread_unicode(path: Path) -> np.ndarray | None:
    """Read an image from paths that may contain non-ASCII characters."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    data = np.frombuffer(raw, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """Write an image to paths that may contain non-ASCII characters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        path.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: 체커보드 시트 생성 — sheets/gen.py 에 위임
# ═════════════════════════════════════════════════════════════════════════════

def gen_checkerboard_sheet(
    grid: tuple[int, int] = DEFAULT_GRID,
    square_mm: float = DEFAULT_SQUARE,
    out_path: Path | None = None,
) -> Path:
    """
    카메라 렌즈 왜곡 보정용 체커보드 A4 PDF 생성.
    실제 구현은 sheets/gen.py:gen_calib_checkerboard_sheet() 에 위임합니다.
    """
    from sheets.gen import gen_calib_checkerboard_sheet
    out_dir = _HERE / "sheets" / "output"
    return gen_calib_checkerboard_sheet(out_dir, grid=grid, square_mm=square_mm, out_path=out_path)


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: 라이브 캡처
# ═════════════════════════════════════════════════════════════════════════════

def run_capture(
    camera_id: int = DEFAULT_CAMERA,
    grid: tuple[int, int] = DEFAULT_GRID,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_images: int = 15,
    auto_interval_s: float = 2.0,
) -> list[Path]:
    """
    라이브 카메라에서 체커보드 이미지를 촬영합니다.

    Keys:
      [S]     — 스냅 (체커보드 검출 성공 시에만)
      [C]     — 자동 스냅 토글  (interval마다 자동 저장)
      [Q]     — 종료 → 캘리브레이션 자동 실행
      [ESC]   — 캘리브레이션 없이 종료
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {camera_id} 열기 실패")

    snap_paths: list[Path] = []
    snap_idx = 0
    auto_snap = False
    last_auto_t = 0.0
    WIN = f"Camera Calib Capture — grid={grid[0]}×{grid[1]}  [S=snap  C=auto  Q=calibrate  ESC=quit]"

    print(f"[capture] 카메라: {camera_id}  저장: {out_dir}")
    print(f"[capture] 목표: {min_images}장 이상  (다양한 각도/거리)")
    print("[capture] S=스냅  C=자동스냅  Q=종료+캘리브레이션  ESC=그냥종료\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, grid,
            _CB_FLAGS
        )

        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, grid, corners, found)
            cv2.putText(vis, "FOUND", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 30), 2)
        else:
            cv2.putText(vis, "NOT FOUND", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 40, 220), 2)

        status = (
            f"snaps: {len(snap_paths)}/{min_images}  "
            f"{'AUTO ' if auto_snap else ''}"
            f"{'✓ READY' if len(snap_paths) >= min_images else ''}"
        )
        cv2.putText(vis, status, (12, vis.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)
        cv2.imshow(WIN, vis)

        key = cv2.waitKey(1) & 0xFF
        now = time.perf_counter()

        do_snap = (
            (key == ord("s") and found) or
            (auto_snap and found and now - last_auto_t >= auto_interval_s)
        )

        if do_snap:
            path = out_dir / f"calib_{snap_idx:04d}.jpg"
            if not _imwrite_unicode(path, frame):
                print(f"[snap][WARN] failed to save: {path}")
                last_auto_t = now
                continue
            snap_paths.append(path)
            snap_idx += 1
            last_auto_t = now
            print(f"[snap] {path}  ({len(snap_paths)}/{min_images})")

        elif key == ord("c"):
            auto_snap = not auto_snap
            print(f"[capture] 자동 스냅: {'ON' if auto_snap else 'OFF'}")

        elif key == ord("q"):
            print(f"[capture] {len(snap_paths)}장 촬영 완료 → 캘리브레이션 시작")
            break

        elif key == 27:  # ESC
            print("[capture] 캘리브레이션 없이 종료")
            snap_paths.clear()
            break

    cap.release()
    cv2.destroyAllWindows()
    return snap_paths


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: 캘리브레이션 실행
# ═════════════════════════════════════════════════════════════════════════════

def run_calibrate(
    image_paths: list[Path],
    grid: tuple[int, int] = DEFAULT_GRID,
    square_mm: float = DEFAULT_SQUARE,
    out_path: Path | None = None,
    camera_id: int = DEFAULT_CAMERA,
) -> CameraCalib | None:
    """
    이미지 리스트에서 캘리브레이션을 실행하고 CameraCalib 를 반환합니다.

    체커보드 코너가 검출된 이미지만 사용합니다.
    """
    if not image_paths:
        print("[calib] 이미지 없음")
        return None

    # 3D 오브젝트 포인트 (체커보드 평면 z=0)
    cols, rows = grid
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_mm

    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None
    used, skipped = 0, 0

    for path in image_paths:
        img = _imread_unicode(path)
        if img is None:
            skipped += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            h, w = gray.shape[:2]
            img_size = (w, h)

        found, corners = cv2.findChessboardCorners(
            gray, grid,
            _CB_FLAGS
        )
        if not found:
            skipped += 1
            continue

        # 서브픽셀 정밀도
        corners_sp = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
        obj_pts.append(objp)
        img_pts.append(corners_sp)
        used += 1

    print(f"[calib] 사용: {used}장 / 스킵: {skipped}장 / 전체: {len(image_paths)}장")

    if used < 10:
        print(f"[calib] 캘리브레이션에 최소 10장 필요 (현재 {used}장). 더 촬영하세요.")
        return None

    if img_size is None:
        return None

    # 캘리브레이션
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None
    )
    print(f"[calib] RMS 재투영 오차: {rms:.4f} px  (목표: < 1.0)")
    if rms > 1.5:
        print("[calib] ⚠️  RMS 가 높습니다. 더 다양한 각도로 다시 촬영을 권장합니다.")

    calib = CameraCalib(
        camera_matrix = K,
        dist_coeffs   = dist,
        image_size    = img_size,
        rms_px        = float(rms),
    )

    if out_path is None:
        out_path = _HERE / f"calib_camera{camera_id}.json"
    calib.save(out_path)
    print(calib.summary())
    return calib


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: 왜곡 보정 미리보기
# ═════════════════════════════════════════════════════════════════════════════

def run_preview(
    calib_path: str,
    camera_id: int = DEFAULT_CAMERA,
) -> None:
    """왜곡 보정 전/후를 나란히 보여줍니다."""
    calib = CameraCalib.load(calib_path)
    print(calib.summary())

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {camera_id} 열기 실패")

    WIN = "Distortion Preview  [Q=quit]  왼쪽=원본  오른쪽=보정됨"
    print("[preview] Q=종료  왼쪽=원본  오른쪽=보정됨")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        undist = calib.undistort(frame)
        # 같은 크기로 맞추기
        h = min(frame.shape[0], undist.shape[0])
        w = min(frame.shape[1], undist.shape[1])
        side = np.hstack([
            frame[:h, :w],
            undist[:h, :w],
        ])
        # 구분선
        cv2.line(side, (w, 0), (w, h), (0, 200, 0), 2)
        cv2.putText(side, "ORIGINAL",   (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
        cv2.putText(side, "UNDISTORTED", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2)
        cv2.putText(side, f"rms={calib.rms_px:.4f}px", (w + 10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow(WIN, side)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_grid(s: str) -> tuple[int, int]:
    try:
        a, b = s.lower().replace("×", "x").split("x")
        return int(a), int(b)
    except Exception:
        raise argparse.ArgumentTypeError(f"grid 형식 오류 (예: 9x6): {s!r}")


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    _configure_utf8_stdio()

    p = argparse.ArgumentParser(
        description="체커보드 기반 카메라 캘리브레이션",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gen-sheet",  action="store_true",
                      help="체커보드 캘리브레이션 시트 생성 (PDF)")
    mode.add_argument("--capture",    action="store_true",
                      help="라이브 캡처 + 캘리브레이션")
    mode.add_argument("--calibrate",  action="store_true",
                      help="저장된 이미지로 캘리브레이션만 실행")
    mode.add_argument("--preview",    action="store_true",
                      help="왜곡 보정 전/후 미리보기")

    p.add_argument("--camera",   type=int,        default=DEFAULT_CAMERA,
                   help=f"카메라 ID (기본: {DEFAULT_CAMERA})")
    p.add_argument("--grid",     type=_parse_grid, default=DEFAULT_GRID,
                   metavar="WxH",
                   help=f"내부 코너 수 (기본: {DEFAULT_GRID[0]}x{DEFAULT_GRID[1]})")
    p.add_argument("--square",   type=float,       default=DEFAULT_SQUARE,
                   metavar="MM",
                   help=f"체커 한 칸 크기 mm (기본: {DEFAULT_SQUARE})")
    p.add_argument("--images",   default=str(DEFAULT_OUT_DIR),
                   metavar="DIR",
                   help="--calibrate 용 이미지 디렉터리")
    p.add_argument("--out-dir",  default=str(DEFAULT_OUT_DIR),
                   metavar="DIR",
                   help="--capture 스냅 저장 디렉터리")
    p.add_argument("--out",      default=None,
                   metavar="FILE",
                   help="캘리브레이션 JSON 출력 경로 (기본: calib_camera<id>.json)")
    p.add_argument("--calib",    default=None,
                   metavar="FILE",
                   help="--preview 용 캘리브레이션 JSON 경로")
    p.add_argument("--min-images", type=int, default=15,
                   help="--capture 최소 스냅 수 (기본: 15)")

    args = p.parse_args()

    out_path = Path(args.out) if args.out else None

    if args.gen_sheet:
        gen_checkerboard_sheet(args.grid, args.square)

    elif args.capture:
        snaps = run_capture(
            camera_id   = args.camera,
            grid        = args.grid,
            out_dir     = Path(args.out_dir),
            min_images  = args.min_images,
        )
        if snaps:
            run_calibrate(snaps, args.grid, args.square, out_path, args.camera)
        else:
            print("[capture] 스냅 없음 — 캘리브레이션 생략")

    elif args.calibrate:
        img_dir = Path(args.images)
        if not img_dir.exists():
            p.error(f"이미지 디렉터리 없음: {img_dir}")
        paths = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in _IMG_EXTS)
        if not paths:
            p.error(f"이미지 없음: {img_dir}")
        run_calibrate(paths, args.grid, args.square, out_path, args.camera)

    elif args.preview:
        if not args.calib:
            p.error("--preview 에는 --calib FILE 이 필요합니다")
        run_preview(args.calib, args.camera)


if __name__ == "__main__":
    main()
