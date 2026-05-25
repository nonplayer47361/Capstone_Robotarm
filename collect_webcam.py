"""
collect_webcam.py  --  웹캠 이미지 수집 도구 v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

웹캠으로 YOLO 학습용 원본 이미지를 수집합니다.
수집된 이미지는 라벨링 전 raw 이미지 폴더에 저장되며,
이후 annotate_simple.py / label_clay_ball.py 로 라벨링합니다.

팀원 충돌 방지:
  --prefix 옵션으로 팀원마다 다른 접두사를 사용하세요.
  예) --prefix alice   → alice_clay_ball_00000.jpg
      --prefix bob     → bob_clay_ball_00000.jpg

사용법:
  python collect_webcam.py --class-name clay_ball
  python collect_webcam.py --class-name clay_ball --prefix alice
  python collect_webcam.py --class-name clay_ball --count 200   # 200장 자동 중단
  python collect_webcam.py --class-name clay_ball --camera 1 --output raw/clay_ball

조작:
  c           : 현재 프레임 저장
  스페이스바  : 연속 캡처 모드 ON/OFF (--auto-interval 초 간격)
  q           : 종료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="웹캠 이미지 수집 도구 v2.0 (YOLO 학습용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--class-name", required=True,
                   help="수집할 클래스 이름 (파일명에 포함됨)")
    p.add_argument("--output", default="raw_images",
                   help="저장 폴더 (기본: ./raw_images)")
    p.add_argument("--prefix", default="",
                   help="팀원 식별자 접두사 (예: alice). 충돌 방지용")
    p.add_argument("--camera", type=int, default=0,
                   help="카메라 인덱스 (기본 0; 외부 카메라는 1 또는 2)")
    p.add_argument("--width", type=int, default=1280,
                   help="캡처 해상도 가로 (기본 1280)")
    p.add_argument("--height", type=int, default=720,
                   help="캡처 해상도 세로 (기본 720)")
    p.add_argument("--auto-interval", type=float, default=0.5,
                   help="연속 캡처 간격 초 (기본 0.5)")
    p.add_argument("--count", type=int, default=0,
                   help="목표 장수. 0=무제한. 달성 시 자동 종료")
    return p.parse_args()


def _make_stem(prefix: str, class_name: str, idx: int) -> str:
    """파일명 줄기 생성. prefix 있으면 앞에 붙임."""
    base = f"{class_name}_{idx:05d}"
    return f"{prefix}_{base}" if prefix else base


def _save_frame(frame: np.ndarray, out_dir: Path, stem: str) -> Path:
    """한글/유니코드 경로 대응 저장 (imencode + tofile)."""
    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    path = out_dir / f"{stem}.jpg"
    enc.tofile(str(path))
    return path


def _count_existing(out_dir: Path, prefix: str, class_name: str) -> int:
    """기존 파일 수 카운트 (파일명 패턴 일치)."""
    if not out_dir.exists():
        return 0
    pattern = f"{prefix}_{class_name}_" if prefix else f"{class_name}_"
    return sum(1 for f in out_dir.iterdir()
               if f.suffix.lower() == ".jpg" and f.stem.startswith(pattern))


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit(f"[오류] 카메라 {args.camera} 열기 실패")

    # 기존 파일 수로 카운터 초기화 → 덮어쓰기 방지
    count      = _count_existing(out_dir, args.prefix, args.class_name)
    auto_mode  = False
    last_auto  = 0.0
    goal       = args.count  # 0 = 무제한

    tag = f"{args.prefix}_{args.class_name}" if args.prefix else args.class_name
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  웹캠 수집 시작")
    print(f"  클래스  : {args.class_name}")
    print(f"  접두사  : {args.prefix if args.prefix else '(없음)'}")
    print(f"  저장 예시: {tag}_00000.jpg")
    print(f"  저장 폴더: {out_dir}")
    print(f"  기존 이미지: {count}장")
    if goal:
        print(f"  목표 장수: {goal}장 (달성 시 자동 종료)")
    print(f"  조작: c=저장  스페이스=연속모드  q=종료")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패")
            time.sleep(0.05)
            continue

        now = time.time()

        # ── 연속 캡처 ────────────────────────────────────────
        if auto_mode and (now - last_auto) >= args.auto_interval:
            stem = _make_stem(args.prefix, args.class_name, count)
            _save_frame(frame, out_dir, stem)
            count += 1
            last_auto = now
            print(f"[AUTO] {stem}.jpg  ({count}장)")
            if goal and count >= goal:
                print(f"\n[완료] 목표 {goal}장 달성 → 자동 종료")
                break

        # ── HUD ──────────────────────────────────────────────
        display    = frame.copy()
        mode_text  = "AUTO" if auto_mode else "MANUAL"
        color      = (0, 100, 255) if auto_mode else (0, 220, 60)
        goal_text  = f"/{goal}" if goal else ""
        cv2.rectangle(display, (0, 0), (display.shape[1], 72), (0, 0, 0), -1)
        cv2.putText(display,
                    f"[{mode_text}]  {tag}  saved={count}{goal_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display,
                    "c=capture  SPACE=auto  q=quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # 목표 진행 바
        if goal:
            bar_w = int(display.shape[1] * min(count, goal) / goal)
            cv2.rectangle(display, (0, 0), (bar_w, 3), (0, 200, 100), -1)

        cv2.imshow("Webcam Collector", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            stem = _make_stem(args.prefix, args.class_name, count)
            _save_frame(frame, out_dir, stem)
            count += 1
            print(f"[SAVE] {stem}.jpg  ({count}장)")
            if goal and count >= goal:
                print(f"\n[완료] 목표 {goal}장 달성 → 자동 종료")
                break
        elif key == ord(' '):
            auto_mode = not auto_mode
            last_auto = now
            print(f"[MODE] {'연속 캡처 ON' if auto_mode else '수동 캡처'}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n수집 완료.  총 저장: {count}장 → {out_dir}")
    print(f"다음 단계:")
    print(f"  python start_labeling.py --images {out_dir} --target {args.class_name}")


if __name__ == "__main__":
    main()
