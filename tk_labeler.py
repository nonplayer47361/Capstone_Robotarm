from __future__ import annotations

import hashlib
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from dataset_utils import (
    create_yaml,
    ensure_yolo_dirs,
    label_exists_in_dataset,
    load_label_from_dataset,
    print_stats,
    save_labeled,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = ["#27ae60", "#e74c3c", "#3498db", "#f1c40f", "#9b59b6", "#1abc9c"]


def read_image(path: Path) -> np.ndarray | None:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


class TkAnnotator(tk.Tk):
    """Canvas-based YOLO bbox labeler inspired by LabelImg/LabelMe.

    The image and annotation preview are separate canvas layers. Mouse movement
    never mutates the image pixels, which avoids OpenCV window-event quirks.
    """

    def __init__(
        self,
        images_dir: str | Path,
        dataset_dir: str | Path,
        classes: list[str],
        val_ratio: float = 0.2,
        locked_class: int | None = None,
        range_start: int = 0,
        range_end: int = -1,
    ) -> None:
        super().__init__()
        self.title("Berry YOLO Canvas Labeler")
        self.geometry("1280x820")
        self.minsize(900, 620)

        self.images_dir = Path(images_dir)
        self.dataset_dir = Path(dataset_dir)
        self.classes = classes
        self.val_ratio = val_ratio
        self.locked_class = locked_class
        self.cur_class = locked_class if locked_class is not None else 0

        ensure_yolo_dirs(self.dataset_dir)
        all_paths = sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not all_paths:
            raise RuntimeError(f"No supported images found: {self.images_dir}")
        end = range_end + 1 if range_end >= 0 else len(all_paths)
        self.image_paths = all_paths[range_start:end]
        if not self.image_paths:
            raise RuntimeError("No images in selected range.")

        self._done_stems = {
            p.stem for p in self.image_paths if label_exists_in_dataset(p.stem, self.dataset_dir)
        }
        self.idx = next((i for i, p in enumerate(self.image_paths) if p.stem not in self._done_stems), 0)

        self.img_bgr: np.ndarray | None = None
        self.img_rgb: Image.Image | None = None
        self.tk_img: ImageTk.PhotoImage | None = None
        self.img_w = 1
        self.img_h = 1
        self.boxes: list[tuple[int, int, int, int, int]] = []
        self.selected_box_idx: int | None = None

        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self.view_w = 1
        self.view_h = 1
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None

        self._build_ui()
        self._bind_events()
        self.load_image()

    def _build_ui(self) -> None:
        self.configure(bg="#101216")
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill=tk.X)

        self.status_var = tk.StringVar()
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)

        btns = ttk.Frame(top)
        btns.pack(side=tk.RIGHT)
        for text, cmd in [
            ("< Prev", self.prev_image),
            ("Save+Next", self.save_next),
            ("Empty/Neg", self.empty_next),
            ("Undo", self.undo),
            ("Clear", self.clear_boxes),
            ("Quit", self.destroy),
        ]:
            ttk.Button(btns, text=text, command=cmd).pack(side=tk.LEFT, padx=3)

        self.canvas = tk.Canvas(self, bg="#202225", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(self, padding=(10, 6))
        bottom.pack(fill=tk.X)
        self.help_var = tk.StringVar(
            value=(
                "Drag to draw. Click a box to select. Arrow keys nudge 1px, Shift+Arrow 5px. "
                "S/Enter saves. N saves empty. Z clears. Esc cancels drag."
            )
        )
        ttk.Label(bottom, textvariable=self.help_var).pack(side=tk.LEFT)

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", lambda _e: self.render())
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<ButtonPress-3>", lambda _e: self.undo())
        self.bind("<s>", lambda _e: self.save_next())
        self.bind("<Return>", lambda _e: self.save_next())
        self.bind("<n>", lambda _e: self.empty_next())
        self.bind("<d>", lambda _e: self.undo())
        self.bind("<z>", lambda _e: self.clear_boxes())
        self.bind("<Escape>", lambda _e: self.cancel_drag())
        self.bind("<p>", lambda _e: self.prev_image())
        self.bind("<q>", lambda _e: self.destroy())
        self.bind("<Left>", lambda _e: self.nudge_selected_box(-1, 0))
        self.bind("<Right>", lambda _e: self.nudge_selected_box(1, 0))
        self.bind("<Up>", lambda _e: self.nudge_selected_box(0, -1))
        self.bind("<Down>", lambda _e: self.nudge_selected_box(0, 1))
        self.bind("<Shift-Left>", lambda _e: self.nudge_selected_box(-5, 0))
        self.bind("<Shift-Right>", lambda _e: self.nudge_selected_box(5, 0))
        self.bind("<Shift-Up>", lambda _e: self.nudge_selected_box(0, -5))
        self.bind("<Shift-Down>", lambda _e: self.nudge_selected_box(0, 5))
        for i in range(10):
            self.bind(str(i), lambda _e, n=i: self.set_class(n))

    def set_class(self, cls: int) -> None:
        if self.locked_class is None and cls < len(self.classes):
            self.cur_class = cls
            self.render()

    def load_image(self) -> None:
        path = self.image_paths[self.idx]
        img = read_image(path)
        self.img_bgr = img if img is not None else np.zeros((480, 640, 3), np.uint8)
        self.img_h, self.img_w = self.img_bgr.shape[:2]
        rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.img_rgb = Image.fromarray(rgb)
        self.boxes = load_label_from_dataset(path.stem, self.dataset_dir, self.img_w, self.img_h, self.val_ratio)
        self.selected_box_idx = len(self.boxes) - 1 if self.boxes else None
        self.cancel_drag()
        self.render()

    def canvas_to_img(self, x: int, y: int) -> tuple[int, int] | None:
        ix = int((x - self.off_x) / self.scale)
        iy = int((y - self.off_y) / self.scale)
        if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
            return ix, iy
        return None

    def img_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        return int(self.off_x + x * self.scale), int(self.off_y + y * self.scale)

    def render(self) -> None:
        if self.img_rgb is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.scale = min(cw / self.img_w, ch / self.img_h)
        self.view_w = max(1, int(self.img_w * self.scale))
        self.view_h = max(1, int(self.img_h * self.scale))
        self.off_x = (cw - self.view_w) // 2
        self.off_y = (ch - self.view_h) // 2

        resized = self.img_rgb.resize((self.view_w, self.view_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y, image=self.tk_img, anchor=tk.NW)
        self.canvas.create_rectangle(self.off_x, self.off_y, self.off_x + self.view_w, self.off_y + self.view_h, outline="#555")

        for idx, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            self.draw_box(cls, x1, y1, x2, y2, width=4 if idx == self.selected_box_idx else 2, selected=idx == self.selected_box_idx)

        if self.drag_start and self.drag_current:
            x1, y1 = self.img_to_canvas(*self.drag_start)
            x2, y2 = self.img_to_canvas(*self.drag_current)
            color = COLORS[self.cur_class % len(COLORS)]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, dash=(4, 3), tags="preview")

        cls_name = self.classes[self.cur_class] if self.cur_class < len(self.classes) else str(self.cur_class)
        done = len(self._done_stems)
        total = len(self.image_paths)
        self.status_var.set(
            f"[{self.idx + 1}/{total}] {self.image_paths[self.idx].name} | done={done} ({done / max(total, 1) * 100:.0f}%) | class[{self.cur_class}]={cls_name} | boxes={len(self.boxes)}"
        )

    def draw_box(self, cls: int, x1: int, y1: int, x2: int, y2: int, width: int = 2, selected: bool = False) -> None:
        color = COLORS[cls % len(COLORS)]
        cx1, cy1 = self.img_to_canvas(x1, y1)
        cx2, cy2 = self.img_to_canvas(x2, y2)
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=width)
        if selected:
            self.canvas.create_rectangle(cx1 - 2, cy1 - 2, cx2 + 2, cy2 + 2, outline="#ffffff", width=1, dash=(3, 2))
        label = self.classes[cls] if cls < len(self.classes) else str(cls)
        self.canvas.create_rectangle(cx1, cy1 - 20, cx1 + max(70, len(label) * 9), cy1, fill=color, outline=color)
        self.canvas.create_text(cx1 + 4, cy1 - 10, text=label, fill="black", anchor=tk.W, font=("Segoe UI", 9, "bold"))

    def box_at(self, pt: tuple[int, int]) -> int | None:
        px, py = pt
        for idx in range(len(self.boxes) - 1, -1, -1):
            _, x1, y1, x2, y2 = self.boxes[idx]
            if x1 <= px <= x2 and y1 <= py <= y2:
                return idx
        return None

    def nudge_selected_box(self, dx: int, dy: int) -> None:
        if not self.boxes:
            return
        if self.selected_box_idx is None or self.selected_box_idx >= len(self.boxes):
            self.selected_box_idx = len(self.boxes) - 1
        cls, x1, y1, x2, y2 = self.boxes[self.selected_box_idx]
        dx = max(-x1, min(dx, self.img_w - 1 - x2))
        dy = max(-y1, min(dy, self.img_h - 1 - y2))
        if dx == 0 and dy == 0:
            return
        self.boxes[self.selected_box_idx] = (cls, x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        self.cancel_drag()

    def on_down(self, event: tk.Event) -> None:
        pt = self.canvas_to_img(event.x, event.y)
        if pt is None:
            return
        hit = self.box_at(pt)
        if hit is not None:
            self.selected_box_idx = hit
            self.cancel_drag()
            return
        self.drag_start = pt
        self.drag_current = pt
        self.render()

    def on_drag(self, event: tk.Event) -> None:
        if not self.drag_start:
            return
        pt = self.canvas_to_img(event.x, event.y)
        if pt is None:
            ix = max(0, min(self.img_w - 1, int((event.x - self.off_x) / self.scale)))
            iy = max(0, min(self.img_h - 1, int((event.y - self.off_y) / self.scale)))
            pt = (ix, iy)
        self.drag_current = pt
        self.render()

    def on_up(self, event: tk.Event) -> None:
        if not self.drag_start:
            return
        end = self.canvas_to_img(event.x, event.y) or self.drag_current
        if end is None:
            self.cancel_drag()
            return
        x1, x2 = sorted([self.drag_start[0], end[0]])
        y1, y2 = sorted([self.drag_start[1], end[1]])
        min_size = max(3, int(6 / max(self.scale, 0.01)))
        if x2 - x1 >= min_size and y2 - y1 >= min_size:
            self.boxes.append((self.cur_class, x1, y1, x2, y2))
            self.selected_box_idx = len(self.boxes) - 1
        self.cancel_drag()

    def cancel_drag(self) -> None:
        self.drag_start = None
        self.drag_current = None
        if hasattr(self, "canvas"):
            self.render()

    def undo(self) -> None:
        if self.boxes:
            self.boxes.pop()
        self.selected_box_idx = len(self.boxes) - 1 if self.boxes else None
        self.cancel_drag()

    def clear_boxes(self) -> None:
        self.boxes.clear()
        self.selected_box_idx = None
        self.cancel_drag()

    def save_current(self) -> None:
        lines = []
        for cls, x1, y1, x2, y2 in self.boxes:
            cx = (x1 + x2) / 2 / self.img_w
            cy = (y1 + y2) / 2 / self.img_h
            bw = abs(x2 - x1) / self.img_w
            bh = abs(y2 - y1) / self.img_h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        path = self.image_paths[self.idx]
        save_labeled(self.img_bgr, lines, path.stem, self.dataset_dir, self.val_ratio)
        create_yaml(self.dataset_dir, nc=len(self.classes), names=self.classes)
        self._done_stems.add(path.stem)

    def save_next(self) -> None:
        self.save_current()
        self.next_image()

    def empty_next(self) -> None:
        self.boxes = []
        self.selected_box_idx = None
        self.save_current()
        self.next_image()

    def next_image(self) -> None:
        if self.idx < len(self.image_paths) - 1:
            self.idx += 1
            self.load_image()
        else:
            messagebox.showinfo("Done", "All images are labeled.")
            print_stats(self.dataset_dir)

    def prev_image(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self.load_image()


def run_tk_annotator(*args, **kwargs) -> None:
    app = TkAnnotator(*args, **kwargs)
    app.mainloop()
