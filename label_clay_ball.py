"""
label_clay_ball.py  --  보라색 구형 클레이 YOLO 라벨링 도구 v1.3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

이미지를 열면 보라색 HSV 범위로 BBox 를 자동 제안합니다.
[Enter] 로 수락하거나, 마우스로 직접 그릴 수 있습니다.

UI 구조:
  ┌──────────────────────── HUD (상단 고정) ─────────────────────────┐
  │  [진행 바]  파일명 | 완료=N장(XX%)  |  상태(AUTO/NO DETECT/boxes) │
  │  Enter=저장  n=빈라벨  r=거부  a=재탐지  d=삭제  z=초기화  q=종료  │
  ├──────────── 이미지 (비율 유지 Letterbox, 창 크기 자동 대응) ────────┤
  │                     [letterbox 여백: 어두운 배경]                   │
  └─────────────────── 버튼 바 (하단 고정) ───────────────────────────┘

저장 시 YOLO 표준 4폴더 구조 자동 생성:
  dataset/train/images/   dataset/train/labels/
  dataset/val/images/     dataset/val/labels/
  dataset/dataset.yaml    ← 자동 생성

파일명 MD5 해시로 train/val 분할 → 같은 이름은 항상 같은 폴더
→ 팀원끼리 나눠 라벨링해도 train/val 오염 없음

사용법:
  python label_clay_ball.py --images raw/clay_ball
  python label_clay_ball.py --images raw/clay_ball --dataset-dir my_dataset
  python label_clay_ball.py --images raw/clay_ball --range 0 249   # 팀원 A 담당
  python label_clay_ball.py --images raw/clay_ball --range 250 499 # 팀원 B 담당

조작키:
  마우스 드래그   BBox 수동 그리기
  Enter / s       현재 BBox 저장 후 다음 이미지
  n               빈 라벨(객체 없음) 저장 후 다음 이미지 (네거티브)
  r               자동 탐지 거부 → 수동 드래그 모드
  a               자동 탐지 재실행
  d               마지막 BBox 삭제
  z               전체 초기화 + 자동 재탐지
  p               이전 이미지 (저장 없이)
  q               종료

HSV 범위 조정 (보라색이 잘 안 잡힐 때):
  --sat-lo 20     채도 기준 낮추기 (연한 보라)
  --hue-lo 100    Hue 범위 넓히기 (청보라 포함)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── 의존성 체크 ────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Required package missing: {e}")
    print("  pip install opencv-python numpy")
    sys.exit(1)

# ── dataset_utils 임포트 ──────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset_utils import (
    create_yaml,
    ensure_yolo_dirs,
    find_in_dataset,
    label_exists_in_dataset,
    load_label_from_dataset,
    print_stats,
    save_labeled,
)

# ── UI 레이아웃 상수 ──────────────────────────────────────────────────
_WIN   = "Clay Ball Labeler"
_HUD_H = 88    # 상단 HUD 바 높이 (px)
_BTN_H = 52    # 하단 버튼 바 높이 (px)
_DEF_W = 1400  # 기본 창 너비
_DEF_H = 900   # 기본 창 높이
_DEFAULT_ZOOM = 1.35
_BOX_DISPLAY_THICKNESS = 4

_ARROW_DELTAS = {
    2424832: (-1, 0),  # Windows left
    2555904: (1, 0),   # Windows right
    2490368: (0, -1),  # Windows up
    2621440: (0, 1),   # Windows down
    65361: (-1, 0),    # X11 left
    65363: (1, 0),     # X11 right
    65362: (0, -1),    # X11 up
    65364: (0, 1),     # X11 down
    81: (-1, 0),       # OpenCV fallback left
    83: (1, 0),        # OpenCV fallback right
    82: (0, -1),       # OpenCV fallback up
    84: (0, 1),        # OpenCV fallback down
}
_FAST_NUDGE_KEYS = {
    ord("j"): (-5, 0),
    ord("l"): (5, 0),
    ord("i"): (0, -5),
    ord("k"): (0, 5),
}

# ── UI 색상 (BGR) ─────────────────────────────────────────────────────
_C_BG  = ( 36,  36,  36)  # letterbox 배경
_C_HUD = ( 22,  22,  22)  # HUD / 버튼 바 배경
_C_SEP = ( 65,  65,  65)  # 구분선
_C_BAR = ( 50, 200, 100)  # 진행 바 (녹색)

# ── BBox 색상 (BGR) ───────────────────────────────────────────────────
C_AUTO   = (  0, 210, 255)  # 자동 탐지 (황색)
C_MANUAL = (  0, 220,  60)  # 수동 (녹색)
C_DRAG   = (220, 100,   0)  # 드래그 중 (파랑)
C_WARN   = (  0,  50, 220)  # 경고 (빨강)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ["blueberry", "strawberry"]

# ── 버튼 정의 (action, 레이블, BGR 배경색) ────────────────────────────
_BTNS: list[tuple[str, str, tuple[int,int,int]]] = [
    ("prev",   "< Prev",    ( 55,  55,  55)),
    ("save",   "Save+Next", ( 35, 110,  60)),
    ("empty",  "Empty/Neg", ( 80,  80,  35)),
    ("reject", "Manual",    ( 55,  55,  55)),
    ("auto",   "Auto",      ( 65,  75, 105)),
    ("undo",   "Undo",      ( 55,  55,  55)),
    ("clear",  "Clear",     ( 55,  55,  55)),
    ("quit",   "Quit",      ( 65,  45,  45)),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 보라색 자동 탐지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_purple(
    img: np.ndarray,
    hue_lo: int, hue_hi: int,
    sat_lo: int, val_lo: int,
    pad: float = 0.08,
    min_area: float = 0.001,
) -> tuple[int, int, int, int] | None:
    """보라색 HSV 마스크 → 최대 컨투어 BBox. 없으면 None."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([hue_lo, sat_lo, val_lo]),
                       np.array([hue_hi, 255,    255]))
    k    = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    h, w = img.shape[:2]
    cands = [(cv2.contourArea(c), c)
             for c in cnts if cv2.contourArea(c) >= h * w * min_area]
    if not cands:
        return None
    _, best = max(cands, key=lambda t: t[0])
    x, y, bw, bh = cv2.boundingRect(best)
    p = int(max(bw, bh) * pad)
    return (max(0, x - p), max(0, y - p),
            min(w - 1, x + bw + p), min(h - 1, y + bh + p))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO 라벨 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def to_yolo(bbox: tuple[int, int, int, int], w: int, h: int, cls: int) -> str:
    x1, y1, x2, y2 = bbox
    return (f"{cls} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
            f"{abs(x2-x1)/w:.6f} {abs(y2-y1)/h:.6f}")


def read_image(path: Path) -> np.ndarray | None:
    """한글 경로 대응 이미지 로드."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _put(img: np.ndarray, text: str, x: int, y: int,
         scale: float = 0.55, color=(220,220,220), thick: int = 1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 라벨러 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Labeler:
    def __init__(
        self,
        images_dir:  Path,
        dataset_dir: Path,
        class_id:    int,
        class_name:  str,
        hsv:         dict,
        val_ratio:   float = 0.2,
        range_start: int   = 0,
        range_end:   int   = -1,
        auto_assist: bool  = True,
    ) -> None:
        self.images_dir  = images_dir
        self.dataset_dir = dataset_dir
        self.cls         = class_id
        self.cls_name    = class_name
        self.hsv         = hsv
        self.val_ratio   = val_ratio
        self.auto_assist = auto_assist

        ensure_yolo_dirs(dataset_dir)
        all_paths = sorted(p for p in images_dir.iterdir()
                           if p.suffix.lower() in IMAGE_EXTS)
        if not all_paths:
            raise SystemExit(f"[ERROR] No images found: {images_dir.resolve()}")

        end = (range_end + 1) if range_end >= 0 else len(all_paths)
        self.paths = all_paths[range_start:end]
        if not self.paths:
            raise SystemExit(
                f"[ERROR] No images in --range {range_start} {range_end}. "
                f"Total images: {len(all_paths)}"
            )

        # 완료된 stem 집합 (O(1))
        self._done_stems: set[str] = {
            p.stem for p in self.paths
            if label_exists_in_dataset(p.stem, dataset_dir)
        }
        # 첫 미완료 이미지로 자동 이동
        self.idx = next(
            (i for i, p in enumerate(self.paths) if p.stem not in self._done_stems),
            0,
        )

        self.img:          np.ndarray
        self.iw = self.ih  = 1
        self.boxes:        list[tuple[int, int, int, int, int]] = []
        self.selected_box_idx: int | None = None
        self.auto_bbox:    tuple[int, int, int, int] | None = None
        self.auto_rej      = False
        self.drawing       = False
        self.drag_s        = self.drag_e = None   # 이미지 좌표계
        self._yaml_created = False
        self.pending_action: str | None = None

        # 렌더링 변환 상태 (캔버스 ↔ 이미지 좌표)
        self._cw    = _DEF_W
        self._ch    = _DEF_H
        self._scale = 1.0
        self._off_x = 0        # 이미지 X 오프셋 (캔버스 기준)
        self._off_y = _HUD_H   # 이미지 Y 오프셋 (캔버스 기준)
        self._zoom = _DEFAULT_ZOOM
        self._view_x0 = 0
        self._view_y0 = 0
        self._view_cx = 0.5
        self._view_cy = 0.5
        self._btns: list[tuple[str, tuple[int,int,int,int], tuple[int,int,int]]] = []

    # ── 좌표 변환 헬퍼 ────────────────────────────────────────────────
    def _to_img(self, cx: int, cy: int) -> tuple[int, int]:
        """캔버스 좌표 → 이미지 좌표."""
        return (int((cx - self._off_x) / self._scale + self._view_x0),
                int((cy - self._off_y) / self._scale + self._view_y0))

    def _clip_img(self, ix: int, iy: int) -> tuple[int, int]:
        return max(0, min(ix, self.iw - 1)), max(0, min(iy, self.ih - 1))

    def _in_img(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.iw and 0 <= iy < self.ih

    # ── 로드 ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        img = read_image(self.paths[self.idx])
        self.img  = img if img is not None else np.zeros((480, 640, 3), np.uint8)
        self.ih, self.iw = self.img.shape[:2]
        self.boxes = load_label_from_dataset(
            self.paths[self.idx].stem, self.dataset_dir,
            self.iw, self.ih, self.val_ratio,
        )
        self.selected_box_idx = len(self.boxes) - 1 if self.boxes else None
        self.auto_rej = False
        self.drawing  = False
        self.drag_s = self.drag_e = None
        self.auto_bbox = detect_purple(self.img, **self.hsv) if self.auto_assist and not self.boxes else None
        if self.auto_bbox:
            x1, y1, x2, y2 = self.auto_bbox
            self._view_cx = (x1 + x2) / 2 / max(self.iw, 1)
            self._view_cy = (y1 + y2) / 2 / max(self.ih, 1)
        else:
            self._view_cx = 0.5
            self._view_cy = 0.5

    # ── 저장 ──────────────────────────────────────────────────────────
    def _save(self) -> None:
        boxes   = list(self.boxes)
        is_auto = not self.boxes and self.auto_bbox and not self.auto_rej
        if is_auto:
            boxes.append((self.cls, *self.auto_bbox))

        lines = [to_yolo((x1, y1, x2, y2), self.iw, self.ih, c)
                 for c, x1, y1, x2, y2 in boxes]

        split, _, _ = save_labeled(
            self.img, lines,
            self.paths[self.idx].stem,
            self.dataset_dir, self.val_ratio,
        )
        self._done_stems.add(self.paths[self.idx].stem)

        if not self._yaml_created:
            names = list(DEFAULT_CLASSES)
            if self.cls >= len(names):
                names.extend(f"class_{i}" for i in range(len(names), self.cls + 1))
            names[self.cls] = self.cls_name
            create_yaml(self.dataset_dir, nc=len(names), names=names)
            self._yaml_created = True

        tag = "AUTO" if is_auto else "MANUAL"
        print(f"  [{tag:6s}] {self.paths[self.idx].name:<40s} ({len(lines)} boxes) -> {split}/")

    # ── 이미지 어노테이션 그리기 (이미지 좌표계) ──────────────────────
    def _draw_img(self) -> np.ndarray:
        """이미지 복사본에 BBox / 드래그 / 경고를 그려 반환 (이미지 좌표 기준)."""
        d = self.img.copy()
        box_t = self._box_thickness()
        text_t = max(1, box_t // 2)

        # 자동 BBox
        if self.auto_bbox and not self.auto_rej and not self.boxes:
            x1, y1, x2, y2 = self.auto_bbox
            cv2.rectangle(d, (x1, y1), (x2, y2), C_AUTO, box_t)
            lbl = f"AUTO: {self.cls_name}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_t)
            ty = max(y1 - 6, th + 6)
            cv2.rectangle(d, (x1, ty - th - 5), (x1 + tw + 8, ty + 3), C_AUTO, -1)
            cv2.putText(d, lbl, (x1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), text_t, cv2.LINE_AA)

        # 수동 BBox
        for idx, (_, x1, y1, x2, y2) in enumerate(self.boxes):
            selected = idx == self.selected_box_idx
            cv2.rectangle(d, (x1, y1), (x2, y2), C_MANUAL, box_t)
            if selected:
                cv2.rectangle(d, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), max(1, text_t))
            (tw, th), _ = cv2.getTextSize(self.cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_t)
            ty = max(y1 - 6, th + 6)
            cv2.rectangle(d, (x1, ty - th - 5), (x1 + tw + 8, ty + 3), C_MANUAL, -1)
            cv2.putText(d, self.cls_name, (x1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), text_t, cv2.LINE_AA)

        # 드래그 미리보기 (이미지 좌표, 유효 크기 이상일 때만 표시)
        if self.drawing and self.drag_s and self.drag_e:
            x1, x2 = sorted([self.drag_s[0], self.drag_e[0]])
            y1, y2 = sorted([self.drag_s[1], self.drag_e[1]])
            if x2 - x1 >= self._min_box_size() and y2 - y1 >= self._min_box_size():
                cv2.rectangle(d, self.drag_s, self.drag_e, C_DRAG, box_t)

        # NO DETECT 경고
        if not self.boxes and (not self.auto_bbox or self.auto_rej):
            ih, iw = d.shape[:2]
            cv2.rectangle(d, (0, ih - 34), (iw, ih), (0, 0, 50), -1)
            cv2.putText(d, "NO DETECT -- drag to draw a box",
                        (10, ih - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WARN, 1,
                        cv2.LINE_AA)
        return d

    # ── 버튼 레이아웃 (캔버스 좌표계) ────────────────────────────────
    def _layout_buttons(self, cw: int, ch: int) -> None:
        bx  = 10
        by1 = ch - _BTN_H + 9
        by2 = ch - _BTN_H + 43
        self._btns = []
        for action, label, fill in _BTNS:
            bw = max(72, 14 + len(label) * 9)
            self._btns.append((action, (bx, by1, bx + bw, by2), fill))
            bx += bw + 8

    def _draw_btns(self, canvas: np.ndarray) -> None:
        for action, (x1, y1, x2, y2), fill in self._btns:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 120, 120), 1)
            label = next(lb for a, lb, _ in _BTNS if a == action)
            cv2.putText(canvas, label, (x1 + 8, y1 + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (240, 240, 240), 1, cv2.LINE_AA)

    def _btn_at(self, cx: int, cy: int) -> str | None:
        for action, (x1, y1, x2, y2), _ in self._btns:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return action
        return None

    def _min_box_size(self) -> int:
        return max(6, int(10 / max(self._scale, 0.01)))

    def _box_thickness(self) -> int:
        return max(2, int(round(_BOX_DISPLAY_THICKNESS / max(self._scale, 0.01))))

    def _set_zoom(self, zoom: float) -> None:
        self._zoom = max(1.0, min(3.0, zoom))

    def _add_box_from_points(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        x1, x2 = sorted([a[0], b[0]])
        y1, y2 = sorted([a[1], b[1]])
        min_size = self._min_box_size()
        if x2 - x1 >= min_size and y2 - y1 >= min_size:
            self.boxes.append((self.cls, x1, y1, x2, y2))
            self.selected_box_idx = len(self.boxes) - 1
            self.auto_bbox = None
            return True
        return False

    def _box_at(self, pt: tuple[int, int]) -> int | None:
        px, py = pt
        for idx in range(len(self.boxes) - 1, -1, -1):
            _, x1, y1, x2, y2 = self.boxes[idx]
            if x1 <= px <= x2 and y1 <= py <= y2:
                return idx
        return None

    def _move_bbox(self, bbox: tuple[int, int, int, int], dx: int, dy: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        dx = max(-x1, min(dx, self.iw - 1 - x2))
        dy = max(-y1, min(dy, self.ih - 1 - y2))
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    def _nudge_selected_box(self, dx: int, dy: int) -> None:
        if self.boxes:
            if self.selected_box_idx is None or self.selected_box_idx >= len(self.boxes):
                self.selected_box_idx = len(self.boxes) - 1
            cls, x1, y1, x2, y2 = self.boxes[self.selected_box_idx]
            nx1, ny1, nx2, ny2 = self._move_bbox((x1, y1, x2, y2), dx, dy)
            self.boxes[self.selected_box_idx] = (cls, nx1, ny1, nx2, ny2)
            return
        if self.auto_bbox and not self.auto_rej:
            self.auto_bbox = self._move_bbox(self.auto_bbox, dx, dy)

    # ── 렌더링 ────────────────────────────────────────────────────────
    def _render(self) -> np.ndarray:
        # 현재 창 크기 조회 (adaptive letterbox)
        try:
            _, _, cw, ch = cv2.getWindowImageRect(_WIN)
            if cw < 400 or ch < 300:
                cw, ch = self._cw, self._ch
        except Exception:
            cw, ch = self._cw, self._ch
        self._cw, self._ch = cw, ch

        # 이미지 표시 영역 (HUD 아래, 버튼 바 위)
        ia_w = cw
        ia_h = ch - _HUD_H - _BTN_H
        fit_scale = min(ia_w / self.iw, ia_h / self.ih)
        scale = fit_scale * self._zoom
        nw    = int(self.iw * scale)
        nh    = int(self.ih * scale)
        self._scale = scale
        disp_w = min(ia_w, nw)
        disp_h = min(ia_h, nh)
        self._off_x = (ia_w - disp_w) // 2
        self._off_y = _HUD_H + (ia_h - disp_h) // 2

        view_w = min(self.iw, max(1, int(round(disp_w / scale))))
        view_h = min(self.ih, max(1, int(round(disp_h / scale))))
        cx = int(round(self._view_cx * self.iw))
        cy = int(round(self._view_cy * self.ih))
        self._view_x0 = max(0, min(self.iw - view_w, cx - view_w // 2))
        self._view_y0 = max(0, min(self.ih - view_h, cy - view_h // 2))
        view_x1 = min(self.iw, self._view_x0 + view_w)
        view_y1 = min(self.ih, self._view_y0 + view_h)

        # 캔버스 생성 (창과 동일 크기 → 스트레칭 없음)
        canvas = np.full((ch, cw, 3), _C_BG, dtype=np.uint8)

        # 이미지 배치 (letterbox)
        ann     = self._draw_img()
        cropped = ann[self._view_y0:view_y1, self._view_x0:view_x1]
        resized = cv2.resize(cropped, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        canvas[self._off_y:self._off_y + disp_h, self._off_x:self._off_x + disp_w] = resized

        # ── HUD 영역 ─────────────────────────────────────────────────
        canvas[:_HUD_H] = _C_HUD

        # 진행 바 (HUD 최상단 4px)
        n_done  = len(self._done_stems)
        n_total = len(self.paths)
        bar_w   = int(cw * n_done / max(n_total, 1))
        canvas[:4, :bar_w] = _C_BAR

        # Row 1: 파일명 + 진행 정보
        fname = self.paths[self.idx].name
        row1  = (f"[{self.idx+1}/{n_total}]  {fname}"
                 f"  |  done={n_done} ({n_done/max(n_total,1)*100:.0f}%)")
        _put(canvas, row1, 12, 28, 0.55, (220, 220, 220))

        # 상태 표시 (우측)
        if self.auto_bbox and not self.auto_rej and not self.boxes:
            state, scol = "AUTO DETECTED", C_AUTO
        elif not self.boxes and (not self.auto_bbox or self.auto_rej):
            state, scol = "NO DETECT", C_WARN
        else:
            state, scol = f"boxes: {len(self.boxes)}", (100, 220, 100)
        _put(canvas, state, cw - 185, 28, 0.55, scol)

        # Row 2: 키 안내
        _put(canvas,
             "drag=draw/select  arrows=nudge 1px  i/j/k/l=5px  +/-=zoom  s=save  n=empty  r=manual  a=auto",
             12, 62, 0.38, (145, 145, 145))

        # HUD 하단 구분선
        cv2.line(canvas, (0, _HUD_H - 1), (cw, _HUD_H - 1), _C_SEP, 1)

        # ── 버튼 바 ───────────────────────────────────────────────────
        canvas[ch - _BTN_H:] = _C_HUD
        cv2.line(canvas, (0, ch - _BTN_H), (cw, ch - _BTN_H), _C_SEP, 1)
        self._layout_buttons(cw, ch)
        self._draw_btns(canvas)

        return canvas

    # ── 마우스 ────────────────────────────────────────────────────────
    def _mouse(self, ev: int, x: int, y: int, flags: int, param) -> None:  # noqa
        if ev == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self._set_zoom(self._zoom * 1.15)
            else:
                self._set_zoom(self._zoom / 1.15)
            return

        # 버튼 클릭 체크 (캔버스 좌표)
        if ev == cv2.EVENT_LBUTTONDOWN:
            action = self._btn_at(x, y)
            if action:
                self.pending_action = action
                self.drawing = False
                self.drag_s = self.drag_e = None
                return

        # 이미지 좌표로 변환
        ix, iy  = self._to_img(x, y)
        in_img  = self._in_img(ix, iy)

        if ev == cv2.EVENT_RBUTTONDOWN:
            if self.boxes:
                self.boxes.pop()
                self.selected_box_idx = len(self.boxes) - 1 if self.boxes else None
            self.drawing = False
            self.drag_s = self.drag_e = None
            return

        if ev == cv2.EVENT_LBUTTONDOWN and in_img:
            hit = self._box_at((ix, iy))
            if hit is not None:
                self.selected_box_idx = hit
                self.drawing = False
                self.drag_s = self.drag_e = None
                return
            self.drawing = True
            self.drag_s  = (ix, iy)
            self.drag_e  = (ix, iy)
        elif ev == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.drag_e = self._clip_img(ix, iy)
        elif ev == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.drag_s:
                ex, ey = self._clip_img(ix, iy)
                self._add_box_from_points(self.drag_s, (ex, ey))
                self.drag_s = self.drag_e = None

    # ── 다음 이미지로 이동 ──────────────────────────────────────────────
    def _next(self) -> bool:
        if self.idx < len(self.paths) - 1:
            self.idx += 1
            self._load()
            return True
        print("\n[완료] 모든 이미지 라벨링 완료!")
        return False

    # ── 메인 루프 ─────────────────────────────────────────────────────
    def run(self) -> None:
        self._load()
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WIN, _DEF_W, _DEF_H)
        cv2.setMouseCallback(_WIN, self._mouse)
        print(f"\nLabeling start  images={len(self.paths)}  done={len(self._done_stems)}")
        print(f"Dataset path    : {self.dataset_dir.resolve()}")
        print("  keys: Enter/s=save  n=empty  r=manual  a=auto  d=undo  z=clear  p=prev  q=quit")
        print("        arrow keys=nudge selected box 1px, i/j/k/l=nudge 5px\n")

        while True:
            cv2.imshow(_WIN, self._render())
            raw_key = cv2.waitKeyEx(20)
            key    = raw_key & 0xFF
            action = self.pending_action
            self.pending_action = None

            if raw_key in _ARROW_DELTAS:
                dx, dy = _ARROW_DELTAS[raw_key]
                self._nudge_selected_box(dx, dy)
            elif key in _FAST_NUDGE_KEYS:
                dx, dy = _FAST_NUDGE_KEYS[key]
                self._nudge_selected_box(dx, dy)
            elif key == ord('q') or action == "quit":
                break
            elif key in (ord('+'), ord('=')):
                self._set_zoom(self._zoom * 1.15)
            elif key in (ord('-'), ord('_')):
                self._set_zoom(self._zoom / 1.15)
            elif key == ord('0'):
                self._zoom = 1.0
            elif key in (ord('s'), 13) or action == "save":          # 저장 + 다음
                self._save()
                if not self._next():
                    break
            elif key == ord('n') or action == "empty":                # 빈 라벨 + 다음
                self.boxes, self.auto_bbox, self.auto_rej = [], None, True
                self._save()
                print(f"  [NEG]  {self.paths[self.idx].name} -> empty label saved")
                if not self._next():
                    break
            elif key == ord('p') or action == "prev":
                self.idx = max(self.idx - 1, 0)
                self._load()
            elif key == ord('r') or action == "reject":
                self.auto_rej, self.auto_bbox = True, None
                print("  [INFO] Auto suggestion rejected. Drag with the mouse.")
            elif key == ord('a') or action == "auto":
                self.auto_rej  = False
                self.auto_bbox = detect_purple(self.img, **self.hsv)
                print(f"  [AUTO] {'detect success' if self.auto_bbox else 'detect failed'}")
            elif (key == ord('d') or action == "undo") and self.boxes:
                self.boxes.pop()
                self.selected_box_idx = len(self.boxes) - 1 if self.boxes else None
            elif key == ord('z') or action == "clear":
                self.boxes.clear()
                self.selected_box_idx = None
                self.auto_rej  = False
                self.auto_bbox = detect_purple(self.img, **self.hsv) if self.auto_assist else None
            elif key == 27:
                self.drawing = False
                self.drag_s = self.drag_e = None

        cv2.destroyAllWindows()
        print(f"\nLabeling finished  done={len(self._done_stems)}/{len(self.paths)}")
        print_stats(self.dataset_dir)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# argparse + main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blueberry clay YOLO labeler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: Enter/S save, arrows nudge selected box 1px, I/J/K/L nudge 5px, N empty, R manual, A auto, D undo, Z clear, P prev, Q quit.",
    )
    p.add_argument("--images",     required=True,         help="Source image folder")
    p.add_argument("--dataset-dir", default="dataset",    help="YOLO dataset root folder")
    p.add_argument("--val-ratio",  type=float, default=0.2, help="Validation ratio")
    p.add_argument("--class-id",   type=int,   default=0,   help="YOLO class id")
    p.add_argument("--class-name", default="blueberry",    help="Class name")
    p.add_argument("--hue-lo",     type=int,   default=110, metavar="0-179", help="Purple hue low")
    p.add_argument("--hue-hi",     type=int,   default=165, metavar="0-179", help="Purple hue high")
    p.add_argument("--sat-lo",     type=int,   default=35,  metavar="0-255", help="Saturation low")
    p.add_argument("--val-lo",     type=int,   default=30,  metavar="0-255", help="Value low")
    p.add_argument("--pad-ratio",  type=float, default=0.08, help="Auto bbox padding ratio")
    p.add_argument("--min-area",   type=float, default=0.001, help="Minimum detection area ratio")
    p.add_argument("--no-auto-assist", action="store_true",
                   help="Start in manual-only mode without automatic purple HSV suggestions")
    p.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), default=None,
                   help="Image index range, 0-based and inclusive")
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    images_dir  = Path(args.images).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    if not images_dir.exists():
        raise SystemExit(f"[ERROR] Image folder not found: {images_dir}")

    range_start, range_end = (args.range[0], args.range[1]) if args.range else (0, -1)

    print(f"Image folder : {images_dir}")
    print(f"Dataset      : {dataset_dir}")
    print(f"Class        : [{args.class_id}] {args.class_name}")
    print(f"Val ratio    : {args.val_ratio:.0%}")
    print(f"Auto assist  : {'off' if args.no_auto_assist else 'on'}")
    print(f"Purple HSV   : H={args.hue_lo}~{args.hue_hi}  S>={args.sat_lo}  V>={args.val_lo}")
    if args.range:
        print(f"Range        : image index {range_start}..{range_end}")

    hsv = dict(
        hue_lo=args.hue_lo, hue_hi=args.hue_hi,
        sat_lo=args.sat_lo, val_lo=args.val_lo,
        pad=args.pad_ratio, min_area=args.min_area,
    )
    Labeler(images_dir, dataset_dir, args.class_id, args.class_name,
            hsv, args.val_ratio, range_start, range_end,
            auto_assist=not args.no_auto_assist).run()


if __name__ == "__main__":
    main()
