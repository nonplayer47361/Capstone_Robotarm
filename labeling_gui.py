"""
labeling_gui.py -- Simple GUI launcher for berry YOLO labeling.

The real annotation window is still the OpenCV labeler. This launcher makes the
team workflow easier by handling folder selection, target selection, dataset
creation, and command execution from one small window.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

HERE = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def _count_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def normalize_windows_path(raw: str) -> str:
    """Accept won/yen signs that users may paste instead of backslashes."""
    return (
        raw.strip().strip('"')
        .replace("\u20a9", "\\")   # won sign
        .replace("\uffe6", "\\")   # fullwidth won sign
        .replace("\u00a5", "\\")   # yen sign, often rendered for backslash
    )


class LabelingGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Berry YOLO Labeling")
        self.geometry("900x660")
        self.minsize(820, 600)

        self.target_var = tk.StringVar(value="blueberry")
        self.images_var = tk.StringVar(value="")
        self.dataset_var = tk.StringVar(value=str((HERE / "dataset").resolve()))
        self.val_ratio_var = tk.StringVar(value="0.2")
        self.range_start_var = tk.StringVar(value="")
        self.range_end_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.images_info_var = tk.StringVar(value="")
        self.dataset_info_var = tk.StringVar(value="")

        self.proc: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self._refresh_folder_info()
        self.after(100, self._drain_log_queue)

    def _build_ui(self) -> None:
        self.configure(bg="#f4f6f8")
        self._configure_style()

        root = ttk.Frame(self, padding=16, style="App.TFrame")
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root, style="App.TFrame")
        header.pack(fill=tk.X)
        title = ttk.Label(
            header,
            text="Berry YOLO Labeling",
            font=("Segoe UI", 18, "bold"),
            style="Title.TLabel",
        )
        title.pack(anchor=tk.W)
        subtitle = ttk.Label(
            header,
            text="사진 폴더를 선택하고 대상 과일을 고르면 YOLO 형식 데이터셋을 바로 생성합니다.",
            style="Muted.TLabel",
        )
        subtitle.pack(anchor=tk.W, pady=(4, 0))

        target_frame = ttk.LabelFrame(root, text="1. 라벨링 대상 선택", padding=12)
        target_frame.pack(fill=tk.X, pady=(12, 8))
        targets = [
            ("블루베리 모형 - 보라색 구형, 자동 탐지 보조", "blueberry"),
            ("딸기 모형 - 원뿔형, 딸기 클래스로 고정", "strawberry"),
        ]
        for label, value in targets:
            ttk.Radiobutton(target_frame, text=label, value=value, variable=self.target_var).pack(
                anchor=tk.W, pady=2
            )

        path_frame = ttk.LabelFrame(root, text="2. 폴더 선택", padding=12)
        path_frame.pack(fill=tk.X, pady=8)
        self._path_row(path_frame, "사진 폴더", self.images_var, self._browse_images, 0)
        self._path_row(path_frame, "데이터셋 폴더", self.dataset_var, self._browse_dataset, 1)

        info_row = ttk.Frame(path_frame)
        info_row.grid(row=2, column=1, columnspan=2, sticky=tk.EW, pady=(2, 8))
        ttk.Label(info_row, textvariable=self.images_info_var, style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Label(info_row, textvariable=self.dataset_info_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=(18, 0))

        options = ttk.Frame(path_frame)
        options.grid(row=3, column=1, sticky=tk.W, pady=(4, 0))
        ttk.Label(options, text="검증 비율").pack(side=tk.LEFT)
        ttk.Spinbox(
            options,
            from_=0.05,
            to=0.5,
            increment=0.05,
            width=6,
            textvariable=self.val_ratio_var,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(options, text="담당 범위").pack(side=tk.LEFT, padx=(18, 0))
        ttk.Entry(options, width=6, textvariable=self.range_start_var).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Label(options, text="부터").pack(side=tk.LEFT)
        ttk.Entry(options, width=6, textvariable=self.range_end_var).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(options, text="까지").pack(side=tk.LEFT, padx=(4, 0))

        quick_frame = ttk.LabelFrame(root, text="3. 빠른 선택", padding=12)
        quick_frame.pack(fill=tk.X, pady=8)
        quick_buttons = [
            ("블루베리", HERE / "blueberry_jpg",   HERE / "blueberry_dataset",  "blueberry"),
            ("딸기",     HERE / "strawberries_jpg", HERE / "strawberry_dataset", "strawberry"),
        ]
        for i, (label, images, dataset, target) in enumerate(quick_buttons):
            ttk.Button(
                quick_frame,
                text=label,
                command=lambda im=images, ds=dataset, tg=target: self._set_quick_paths(im, ds, tg),
            ).grid(row=i // 3, column=i % 3, sticky=tk.EW, padx=(0, 8), pady=3)
        ttk.Button(quick_frame, text="폴더 상태 새로고침", command=self._refresh_folder_info).grid(
            row=1, column=2, sticky=tk.EW, padx=(0, 8), pady=3
        )
        for col in range(3):
            quick_frame.columnconfigure(col, weight=1)

        action_frame = ttk.Frame(root)
        action_frame.pack(fill=tk.X, pady=(8, 8))
        self.start_button = ttk.Button(action_frame, text="라벨링 시작", command=self._start_labeling)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(action_frame, text="중지", command=self._stop_labeling, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(action_frame, text="데이터셋 폴더 열기", command=self._open_dataset_folder).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Button(action_frame, text="로그 지우기", command=self._clear_log).pack(side=tk.LEFT)
        ttk.Label(action_frame, textvariable=self.status_var).pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(root, text="실행 로그", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scroll.set)

        help_text = (
            "라벨링 창: 마우스로 드래그해서 박스를 그리고, 하단 버튼 또는 키보드로 저장/이전/취소를 조작합니다. "
            "Save+Next=저장 후 다음, Empty/Neg=객체 없음 저장, Undo=마지막 박스 취소, Clear=전체 삭제."
        )
        ttk.Label(root, text=help_text, style="Muted.TLabel", wraplength=850).pack(anchor=tk.W, pady=(8, 0))

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.configure("App.TFrame", background="#f4f6f8")
        style.configure("Title.TLabel", background="#f4f6f8", foreground="#151924")
        style.configure("Muted.TLabel", background="#f4f6f8", foreground="#5d6675")

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        label: str,
        variable: tk.StringVar,
        command,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=8, pady=4)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky=tk.E, pady=4)
        parent.columnconfigure(1, weight=1)

    def _browse_images(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            self.images_var.set(path)
            self._refresh_folder_info()

    def _browse_dataset(self) -> None:
        path = filedialog.askdirectory(title="Select dataset output folder")
        if path:
            self.dataset_var.set(path)
            self._refresh_folder_info()

    def _set_quick_paths(self, images: Path, dataset: Path, target: str | None = None) -> None:
        self.images_var.set(str(images.resolve()))
        self.dataset_var.set(str(dataset.resolve()))
        if target:
            self.target_var.set(target)
        self._refresh_folder_info()

    def _refresh_folder_info(self) -> None:
        raw_images = normalize_windows_path(self.images_var.get()).strip()
        dataset = Path(normalize_windows_path(self.dataset_var.get())).expanduser()
        if raw_images:
            images = Path(raw_images).expanduser()
            if images.exists() and images.is_dir():
                self.images_info_var.set(f"사진 {_count_images(images)}장 감지")
            else:
                self.images_info_var.set("사진 폴더 없음")
        else:
            self.images_info_var.set("← Browse 버튼으로 사진 폴더를 선택해 주세요")
        if dataset.exists() and dataset.is_dir():
            label_count = len(list(dataset.glob("**/labels/*.txt")))
            self.dataset_info_var.set(f"라벨 {label_count}개")
        else:
            self.dataset_info_var.set("데이터셋 폴더는 시작 시 생성")

    def _open_dataset_folder(self) -> None:
        path = Path(self.dataset_var.get()).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(path)])

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", tk.END)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _validate(self) -> tuple[Path, Path, float, tuple[int, int] | None] | None:
        self.images_var.set(normalize_windows_path(self.images_var.get()))
        self.dataset_var.set(normalize_windows_path(self.dataset_var.get()))
        if not self.images_var.get().strip():
            messagebox.showerror("사진 폴더 없음", "사진 폴더를 선택해 주세요.\nBrowse 버튼으로 폴더를 지정하거나 빠른 선택 버튼을 사용하세요.")
            return None
        images = Path(self.images_var.get()).expanduser().resolve()
        dataset = Path(self.dataset_var.get()).expanduser().resolve()
        if not images.exists() or not images.is_dir():
            messagebox.showerror("Image folder missing", f"Image folder not found:\n{images}")
            return None
        if _count_images(images) == 0:
            messagebox.showerror("No images", f"No supported images found:\n{images}")
            return None
        try:
            val_ratio = float(self.val_ratio_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Validation ratio must be a number.")
            return None
        if not 0.0 < val_ratio < 1.0:
            messagebox.showerror("Invalid value", "Validation ratio must be between 0 and 1.")
            return None
        start_raw = self.range_start_var.get().strip()
        end_raw = self.range_end_var.get().strip()
        range_pair = None
        if start_raw or end_raw:
            if not start_raw or not end_raw:
                messagebox.showerror("Invalid range", "Enter both range start and range end, or leave both blank.")
                return None
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError:
                messagebox.showerror("Invalid range", "Range start/end must be whole numbers.")
                return None
            if start < 0 or end < start:
                messagebox.showerror("Invalid range", "Range must be 0 or higher, and end must be >= start.")
                return None
            range_pair = (start, end)
        return images, dataset, val_ratio, range_pair

    def _start_labeling(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Already running", "A labeling session is already running.")
            return
        values = self._validate()
        if values is None:
            return
        images, dataset, val_ratio, range_pair = values
        dataset.mkdir(parents=True, exist_ok=True)
        self._refresh_folder_info()

        cmd = self._labeling_command()
        if cmd is None:
            return
        cmd.extend([
            "--images",
            str(images),
            "--dataset-dir",
            str(dataset),
            "--target",
            self.target_var.get(),
            "--val-ratio",
            str(val_ratio),
        ])
        if range_pair is not None:
            cmd.extend(["--range", str(range_pair[0]), str(range_pair[1])])
        self._append_log("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        self.status_var.set("Running")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def _labeling_command(self) -> list[str] | None:
        if getattr(sys, "frozen", False):
            cli_exe = HERE / "BerryLabelingCLI.exe"
            if not cli_exe.exists():
                messagebox.showerror(
                    "CLI exe missing",
                    f"BerryLabelingCLI.exe not found:\n{cli_exe}\n\n"
                    "Build or copy the full BerryLabelingTool folder again.",
                )
                return None
            return [str(cli_exe)]
        return [sys.executable, str(HERE / "start_labeling.py")]

    def _stop_labeling(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._append_log("\nStop requested. Closing labeling process...\n")

    def _run_process(self, cmd: list[str]) -> None:
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_queue.put(line)
            code = self.proc.wait()
            self.log_queue.put(f"\nProcess finished with exit code {code}\n")
        except Exception as exc:
            self.log_queue.put(f"\nError: {exc}\n")
        finally:
            self.log_queue.put("__GUI_DONE__")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__GUI_DONE__":
                    self.status_var.set("Ready")
                    self.start_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    self._refresh_folder_info()
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)


def main() -> None:
    app = LabelingGui()
    app.mainloop()


if __name__ == "__main__":
    main()
