from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = ["blueberry", "strawberry"]
COLORS = ["#27ae60", "#e74c3c", "#3498db", "#f1c40f"]


def read_class_names(dataset_dir: Path) -> list[str]:
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        return DEFAULT_CLASS_NAMES
    names: dict[int, str] = {}
    in_names = False
    for raw_line in yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names and not raw_line.startswith((" ", "\t")):
            break
        if in_names and ":" in stripped:
            idx_text, name = stripped.split(":", 1)
            try:
                names[int(idx_text)] = name.strip().strip("'\"")
            except ValueError:
                continue
    if not names:
        return DEFAULT_CLASS_NAMES
    return [names[i] for i in sorted(names)]


def infer_default_class(dataset_dir: Path, class_names: list[str]) -> int | None:
    text = str(dataset_dir).lower()
    for idx, name in enumerate(class_names):
        if name.lower() in text:
            return idx
    return None


@dataclass
class LabelItem:
    stem: str
    split: str
    image_path: Path
    label_path: Path


def parse_yolo_label(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    boxes: list[tuple[int, int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        x1 = max(0, min(width - 1, int((cx - bw / 2) * width)))
        y1 = max(0, min(height - 1, int((cy - bh / 2) * height)))
        x2 = max(0, min(width - 1, int((cx + bw / 2) * width)))
        y2 = max(0, min(height - 1, int((cy + bh / 2) * height)))
        if x2 > x1 and y2 > y1:
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_preview(image_path: Path, label_path: Path, max_size: tuple[int, int], class_names: list[str]) -> Image.Image:
    with Image.open(image_path) as src:
        img = src.convert("RGB")
    orig_w, orig_h = img.size
    boxes = parse_yolo_label(label_path, orig_w, orig_h)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    scale_x = img.width / orig_w
    scale_y = img.height / orig_h
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for cls, x1, y1, x2, y2 in boxes:
        color = COLORS[cls % len(COLORS)]
        px1, py1 = int(x1 * scale_x), int(y1 * scale_y)
        px2, py2 = int(x2 * scale_x), int(y2 * scale_y)
        draw.rectangle((px1, py1, px2, py2), outline=color, width=3)
        label = class_names[cls] if cls < len(class_names) else str(cls)
        bbox = draw.textbbox((px1, py1), label, font=font)
        draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill=color)
        draw.text((px1, py1), label, fill="black", font=font)
    return img


def box_to_yolo(cls: int, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> str:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def save_boxes(label_path: Path, boxes: list[tuple[int, int, int, int, int]], width: int, height: int) -> None:
    lines = [box_to_yolo(cls, x1, y1, x2, y2, width, height) for cls, x1, y1, x2, y2 in boxes]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def label_stems(dataset_dir: Path) -> set[str]:
    stems: set[str] = set()
    for split in ("train", "val"):
        for path in (dataset_dir / split / "labels").glob("*.txt"):
            stems.add(path.stem)
    return stems


def load_review_priority(dataset_dir: Path) -> dict[str, tuple[int, float]]:
    report_path = dataset_dir / "auto_accept_report.json"
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    priority: dict[str, tuple[int, float]] = {}
    for item in report.get("items", []):
        stem = str(item.get("stem", ""))
        if not stem:
            continue
        accepted = bool(item.get("accepted", False))
        audit = bool(item.get("audit", False))
        confidence = float(item.get("confidence", 0.0) or 0.0)
        # Non-accepted and audit samples first, then lower confidence first.
        group = 0 if (not accepted or audit) else 1
        priority[stem] = (group, confidence)
    return priority


def collect_items(
    dataset_dir: Path,
    backup_dir: Path | None,
    auto_only: bool,
    unchecked_only: bool = False,
    review_status: dict[str, bool] | None = None,
) -> list[LabelItem]:
    backup_stems = label_stems(backup_dir) if backup_dir and backup_dir.exists() else set()
    review_status = review_status or {}
    items: list[LabelItem] = []
    for split in ("train", "val"):
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            if auto_only and label_path.stem in backup_stems:
                continue
            if unchecked_only and bool(review_status.get(label_path.stem)):
                continue
            image_path = next((image_dir / f"{label_path.stem}{ext}" for ext in IMAGE_EXTS if (image_dir / f"{label_path.stem}{ext}").exists()), None)
            if image_path:
                items.append(LabelItem(label_path.stem, split, image_path, label_path))
    priority = load_review_priority(dataset_dir)
    if priority:
        items.sort(key=lambda item: (*priority.get(item.stem, (2, 1.0)), item.stem))
    return items


class ReviewGallery(tk.Tk):
    def __init__(
        self,
        dataset_dir: Path,
        backup_dir: Path | None,
        auto_only: bool,
        sequential: bool = False,
        unchecked_only: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.backup_dir = backup_dir
        self.auto_only = auto_only
        self.unchecked_only = unchecked_only
        self.class_names = read_class_names(dataset_dir)
        self.default_class = infer_default_class(dataset_dir, self.class_names)
        target = self.class_names[self.default_class] if self.default_class is not None else "YOLO"
        self.title(f"{target.title()} Label Review Gallery")
        self.geometry("1280x820")
        self.minsize(900, 620)
        self.vars: dict[str, tk.BooleanVar] = {}
        self.photos: list[ImageTk.PhotoImage] = []
        self.active_editor: LabelEditWindow | None = None
        self.status_path = dataset_dir / "review_status.json"
        self.review_status = self.load_status()
        self.items = collect_items(dataset_dir, backup_dir, auto_only, unchecked_only, self.review_status)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.build_ui()
        if sequential and self.items:
            self.after(250, lambda: self.open_index(0))

    def load_status(self) -> dict[str, bool]:
        if not self.status_path.exists():
            return {}
        try:
            data = json.loads(self.status_path.read_text(encoding="utf-8"))
            return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            return {}

    def save_status(self) -> None:
        data = dict(self.review_status)
        data.update({stem: var.get() for stem, var in self.vars.items()})
        self.status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.review_status = data
        self.update_status()

    def close(self) -> None:
        self.save_status()
        self.destroy()

    def build_ui(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(
            top,
            text=f"{len(self.items)} labels shown | unchecked-only={self.unchecked_only} | dataset: {self.dataset_dir}",
            font=("Segoe UI", 11, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(top, text="Save checks", command=self.save_status).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Check all", command=lambda: self.set_all(True)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Uncheck all", command=lambda: self.set_all(False)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Detailed review", command=lambda: self.open_index(0)).pack(side=tk.RIGHT, padx=4)

        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, padding=(10, 0)).pack(fill=tk.X)

        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(outer, bg="#f2f4f6", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        self.grid_frame = ttk.Frame(canvas, padding=10)
        self.grid_frame.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.grid_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        if not self.items:
            ttk.Label(self.grid_frame, text="No labels to review.").grid(row=0, column=0, padx=10, pady=10)
        for idx, item in enumerate(self.items):
            self.add_card(idx, item)
        self.update_status()

    def add_card(self, idx: int, item: LabelItem) -> None:
        card = ttk.Frame(self.grid_frame, padding=8, relief=tk.RIDGE)
        row, col = divmod(idx, 4)
        card.grid(row=row, column=col, padx=8, pady=8, sticky=tk.N)

        preview = draw_preview(item.image_path, item.label_path, (260, 180), self.class_names)
        photo = ImageTk.PhotoImage(preview)
        self.photos.append(photo)
        img_label = ttk.Label(card, image=photo)
        img_label.pack()
        img_label.bind("<Double-Button-1>", lambda _e, i=idx: self.open_index(i))

        var = tk.BooleanVar(value=self.review_status.get(item.stem, False))
        self.vars[item.stem] = var
        ttk.Checkbutton(card, text="OK / reviewed", variable=var, command=self.update_status).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(card, text=f"{item.split} | {item.stem}", width=34).pack(anchor=tk.W)

    def open_index(self, index: int) -> None:
        if not self.items:
            return
        index = max(0, min(index, len(self.items) - 1))
        if self.active_editor is not None and self.active_editor.winfo_exists():
            self.active_editor.destroy()
        self.active_editor = LabelEditWindow(self, index)
        self.active_editor.focus_force()

    def mark_reviewed(self, stem: str, value: bool = True) -> None:
        if stem in self.vars:
            self.vars[stem].set(value)
        self.save_status()

    def set_all(self, value: bool) -> None:
        for var in self.vars.values():
            var.set(value)
        self.update_status()

    def update_status(self) -> None:
        checked = sum(1 for v in self.vars.values() if v.get())
        self.status_var.set(f"Reviewed: {checked}/{len(self.vars)} | saved to {self.status_path}")


class LabelEditWindow(tk.Toplevel):
    def __init__(self, parent: ReviewGallery, index: int) -> None:
        super().__init__(parent)
        self.parent = parent
        self.index = index
        self.item = parent.items[index]
        self.title(f"Detailed review {index + 1}/{len(parent.items)} - {self.item.stem}")
        self.geometry("1120x820")
        self.minsize(820, 620)

        with Image.open(self.item.image_path) as src:
            self.orig_img = src.convert("RGB")
        self.orig_w, self.orig_h = self.orig_img.size
        self.boxes = parse_yolo_label(self.item.label_path, self.orig_w, self.orig_h)
        self.cur_class = self.default_edit_class()
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self.view_w = 1
        self.view_h = 1
        self.tk_img: ImageTk.PhotoImage | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.selected_box_idx: int | None = len(self.boxes) - 1 if self.boxes else None
        self.replace_var = tk.BooleanVar(value=parent.default_class is not None)

        self.build_ui()
        self.bind_events()
        self.render()

    def default_edit_class(self) -> int:
        if self.parent.default_class is not None:
            return self.parent.default_class
        if self.boxes:
            return self.boxes[0][0]
        return 0

    def build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(
            top,
            text="Drag to replace/add box. Click a box to select. Arrow = nudge 1px, Shift+Arrow = 5px.",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT)
        self.class_var = tk.IntVar(value=self.cur_class)
        class_frame = ttk.Frame(top)
        class_frame.pack(side=tk.LEFT, padx=16)
        # 단일 클래스 모델: target class만 노출해 교차 라벨 실수 방지
        locked = self.parent.default_class is not None
        classes_to_show = (
            [(self.parent.default_class, self.parent.class_names[self.parent.default_class])]
            if locked
            else list(enumerate(self.parent.class_names))
        )
        for idx, name in classes_to_show:
            ttk.Radiobutton(
                class_frame,
                text=name,
                value=idx,
                variable=self.class_var,
                command=self.update_class,
            ).pack(side=tk.LEFT, padx=4)
        if locked:
            ttk.Label(class_frame, text="(잠금)", foreground="#e74c3c").pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(top, text="Replace on drag", variable=self.replace_var).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Next >", command=self.save_ok_next).pack(side=tk.RIGHT, padx=3)
        ttk.Button(top, text="< Prev", command=self.previous_item).pack(side=tk.RIGHT, padx=3)
        ttk.Button(top, text="Save", command=self.save).pack(side=tk.RIGHT, padx=3)
        ttk.Button(top, text="Save + OK", command=self.save_ok).pack(side=tk.RIGHT, padx=3)
        ttk.Button(top, text="Undo", command=self.undo).pack(side=tk.RIGHT, padx=3)
        ttk.Button(top, text="Clear", command=self.clear).pack(side=tk.RIGHT, padx=3)

        self.canvas = tk.Canvas(self, bg="#eeeeee", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, padding=8).pack(fill=tk.X)

    def update_class(self) -> None:
        self.cur_class = int(self.class_var.get())

    def bind_events(self) -> None:
        self.canvas.bind("<Configure>", lambda _e: self.render())
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.bind("<Control-s>", lambda _e: self.save())
        self.bind("<Return>", lambda _e: self.save_ok_next())
        self.bind("<Control-Right>", lambda _e: self.save_ok_next())
        self.bind("<Control-Down>", lambda _e: self.save_ok_next())
        self.bind("<Control-Left>", lambda _e: self.previous_item())
        self.bind("<Control-Up>", lambda _e: self.previous_item())
        self.bind("<Prior>", lambda _e: self.previous_item())
        self.bind("<Next>", lambda _e: self.save_ok_next())
        self.bind("<Left>", lambda _e: self.nudge_selected_box(-1, 0))
        self.bind("<Right>", lambda _e: self.nudge_selected_box(1, 0))
        self.bind("<Up>", lambda _e: self.nudge_selected_box(0, -1))
        self.bind("<Down>", lambda _e: self.nudge_selected_box(0, 1))
        self.bind("<Shift-Left>", lambda _e: self.nudge_selected_box(-5, 0))
        self.bind("<Shift-Right>", lambda _e: self.nudge_selected_box(5, 0))
        self.bind("<Shift-Up>", lambda _e: self.nudge_selected_box(0, -5))
        self.bind("<Shift-Down>", lambda _e: self.nudge_selected_box(0, 5))
        self.bind("<z>", lambda _e: self.undo())
        self.bind("<Escape>", lambda _e: self.cancel_drag())

    def canvas_to_img(self, x: int, y: int, clamp: bool = False) -> tuple[int, int] | None:
        ix = int((x - self.off_x) / self.scale)
        iy = int((y - self.off_y) / self.scale)
        if 0 <= ix < self.orig_w and 0 <= iy < self.orig_h:
            return ix, iy
        if clamp:
            return max(0, min(self.orig_w - 1, ix)), max(0, min(self.orig_h - 1, iy))
        return None

    def img_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        return int(self.off_x + x * self.scale), int(self.off_y + y * self.scale)

    def render(self) -> None:
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.scale = min(cw / self.orig_w, ch / self.orig_h)
        self.view_w = max(1, int(self.orig_w * self.scale))
        self.view_h = max(1, int(self.orig_h * self.scale))
        self.off_x = (cw - self.view_w) // 2
        self.off_y = (ch - self.view_h) // 2
        # Only redo the expensive LANCZOS resize when the canvas actually changed size.
        if (self.view_w, self.view_h) != getattr(self, "_last_view_size", None):
            resized = self.orig_img.resize((self.view_w, self.view_h), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(resized)
            self._last_view_size = (self.view_w, self.view_h)

        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y, image=self.tk_img, anchor=tk.NW)
        for idx, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            self.draw_box(cls, x1, y1, x2, y2, width=5 if idx == self.selected_box_idx else 3, selected=idx == self.selected_box_idx)
        if self.drag_start and self.drag_current:
            x1, y1 = self.img_to_canvas(*self.drag_start)
            x2, y2 = self.img_to_canvas(*self.drag_current)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#3498db", width=2, dash=(5, 3))
        checked = self.parent.vars.get(self.item.stem)
        reviewed = bool(checked.get()) if checked else False
        self.status_var.set(
            f"[{self.index + 1}/{len(self.parent.items)}] {self.item.label_path} | boxes={len(self.boxes)} | reviewed={reviewed}"
        )

    def draw_box(self, cls: int, x1: int, y1: int, x2: int, y2: int, width: int = 2, selected: bool = False) -> None:
        color = COLORS[cls % len(COLORS)]
        cx1, cy1 = self.img_to_canvas(x1, y1)
        cx2, cy2 = self.img_to_canvas(x2, y2)
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=width)
        if selected:
            self.canvas.create_rectangle(cx1 - 2, cy1 - 2, cx2 + 2, cy2 + 2, outline="#ffffff", width=1, dash=(3, 2))
        label = self.parent.class_names[cls] if cls < len(self.parent.class_names) else str(cls)
        self.canvas.create_rectangle(cx1, max(0, cy1 - 20), cx1 + max(78, len(label) * 9), cy1, fill=color, outline=color)
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
        dx = max(-x1, min(dx, self.orig_w - 1 - x2))
        dy = max(-y1, min(dy, self.orig_h - 1 - y2))
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
        self.drag_current = self.canvas_to_img(event.x, event.y, clamp=True)
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
        min_size = max(4, int(6 / max(self.scale, 0.01)))
        if x2 - x1 >= min_size and y2 - y1 >= min_size:
            self.update_class()
            if self.replace_var.get():
                self.boxes.clear()
            self.boxes.append((self.cur_class, x1, y1, x2, y2))
            self.selected_box_idx = len(self.boxes) - 1
        self.cancel_drag()

    def cancel_drag(self) -> None:
        self.drag_start = None
        self.drag_current = None
        self.render()

    def undo(self) -> None:
        if self.boxes:
            self.boxes.pop()
        self.selected_box_idx = len(self.boxes) - 1 if self.boxes else None
        self.render()

    def clear(self) -> None:
        self.boxes.clear()
        self.selected_box_idx = None
        self.render()

    def persist(self, mark_ok: bool = False) -> None:
        # Locked mode: 저장 시 모든 box의 class를 target class로 강제 교정.
        # self.boxes도 함께 갱신해야 render()에서 보정된 class 색상이 표시된다.
        locked_class = self.parent.default_class
        if locked_class is not None:
            boxes_to_save = [(locked_class, x1, y1, x2, y2) for (_, x1, y1, x2, y2) in self.boxes]
            self.boxes = boxes_to_save
        else:
            boxes_to_save = self.boxes
        save_boxes(self.item.label_path, boxes_to_save, self.orig_w, self.orig_h)
        if mark_ok:
            self.parent.mark_reviewed(self.item.stem, True)
        else:
            self.parent.save_status()

    def save(self, show_message: bool = True) -> None:
        self.persist(mark_ok=False)
        self.render()
        if not show_message:
            return
        messagebox.showinfo("Saved", f"Saved label:\n{self.item.label_path}", parent=self)

    def save_ok(self) -> None:
        self.persist(mark_ok=True)
        self.destroy()

    def save_ok_next(self) -> None:
        self.persist(mark_ok=True)
        next_index = self.index + 1
        self.destroy()
        if next_index < len(self.parent.items):
            self.parent.after(50, lambda: self.parent.open_index(next_index))
        else:
            messagebox.showinfo("Review complete", "Last image reviewed.", parent=self.parent)
            self.parent.update_status()

    def previous_item(self) -> None:
        self.persist(mark_ok=False)
        prev_index = max(0, self.index - 1)
        if prev_index == self.index:
            self.render()
            return
        self.destroy()
        self.parent.after(50, lambda: self.parent.open_index(prev_index))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview YOLO labels as thumbnail checkboxes.")
    p.add_argument("--dataset-dir", default="", help="YOLO dataset directory to review (required).")
    p.add_argument("--backup-dir", default="", help="Backup dataset directory used to identify auto-labeled images.")
    p.add_argument("--all", action="store_true", help="Show all labels instead of only labels absent from backup.")
    p.add_argument("--sequential", action="store_true", help="Open a large one-by-one review window immediately.")
    p.add_argument("--unchecked-only", action="store_true", help="Hide labels already marked reviewed in review_status.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_dir:
        raise SystemExit("--dataset-dir is required. Example: --dataset-dir path/to/dataset")
    dataset_dir = Path(args.dataset_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else None
    app = ReviewGallery(
        dataset_dir,
        backup_dir,
        auto_only=not args.all,
        sequential=args.sequential,
        unchecked_only=args.unchecked_only,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
