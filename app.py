"""Sudoku Vision: giao diện Tkinter cho pipeline giải Sudoku từ ảnh chụp.

Chạy từ mã nguồn:  python app.py
Đóng gói .exe:      build_exe.bat
"""
import random
import sys
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from sudoku_vision import grid as gridmod
from sudoku_vision import paths, preprocess, review
from sudoku_vision.classifier import DigitClassifier
from sudoku_vision.imageio import imread, imwrite
from sudoku_vision.pipeline import STAGE_TITLES, SudokuPipeline
from sudoku_vision.user_data import save_cells

APP_TITLE = "Sudoku Vision · CS231"
IMAGE_TYPES = [("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")]
COLOR_LOW = "#c26a00"
COLOR_SOLVED = "#2e8b3d"
COLOR_ERROR = "#b3261e"


class ImageView(tk.Canvas):
    """Canvas hiển thị ảnh OpenCV, tự co giãn theo kích thước cửa sổ."""

    def __init__(self, master, placeholder: str):
        super().__init__(master, background="#20242a", highlightthickness=0)
        self.placeholder = placeholder
        self.image = None
        self._photo = None
        self.bind("<Configure>", lambda e: self.redraw())

    def set_image(self, image):
        self.image = image
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        if self.image is None:
            self.create_text(w / 2, h / 2, text=self.placeholder, fill="#a1a8b1", font=("Segoe UI", 11),
                             width=w - 40, justify="center")
            return
        ih, iw = self.image.shape[:2]
        scale = min(w / iw, h / ih)
        size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, size, interpolation=interp)))
        self.create_image(w / 2, h / 2, image=self._photo)


class ReviewBoard(tk.Canvas):
    """Lưới 9x9 để rà soát kết quả nhận dạng.

    Mỗi ô hiển thị ảnh ô cắt từ ảnh gốc, kèm nhãn nhỏ ở góc là số máy đọc được, để người dùng so sánh
    ngay. Ô đáng ngờ được tô màu (độ tin cậy thấp, trùng số, có nét mực nhưng bị coi là trống).
    Người dùng đánh dấu ô sai (chuột phải hoặc phím X) rồi bấm "Nhận dạng lại": ô đánh dấu chuyển sang
    lựa chọn kế tiếp của bộ phân loại; bấm tiếp nếu vẫn sai. Cũng có thể gõ thẳng số đúng.
    """

    CELL = 50
    PAD = 3

    def __init__(self, master, on_change):
        size = 9 * self.CELL + 2 * self.PAD
        super().__init__(master, width=size, height=size, background="#ffffff", highlightthickness=0, takefocus=1)
        self.on_change = on_change
        self._photo = None
        self.bind("<Button-1>", self._on_click)
        self.bind("<Button-3>", self._on_right_click)
        self.bind("<Double-Button-1>", self._on_right_click)
        self.bind("<Key>", self._on_key)
        self.reset(None)

    # ---------- trạng thái ----------
    def reset(self, result):
        self.result = result
        self.values = np.zeros((9, 9), np.int32)
        self.source = np.full((9, 9), "auto", dtype=object)  # auto | fixed | alt | user
        self.marked = set()  # chỉ số ô r*9+c
        self.rejected = {}  # chỉ số ô -> tập nhãn đã bị người dùng bác bỏ
        self.solution = None
        self.selected = None
        self._photo = None
        if result is not None and result.grid is not None:
            self.values = result.grid.copy()
            for r, c, _, _ in result.corrections:
                self.source[r, c] = "fixed"
            if "warped" in result.stages:
                side = 9 * self.CELL
                gray = cv2.cvtColor(cv2.resize(result.stages["warped"], (side, side)), cv2.COLOR_BGR2GRAY)
                light = (255 - (255 - gray.astype(np.float32)) * 0.6).astype(np.uint8)  # làm nhạt để nhãn dễ đọc
                self._photo = ImageTk.PhotoImage(Image.fromarray(light))
        self.redraw()

    def grid_values(self) -> np.ndarray:
        return self.values.copy()

    def has_alternatives(self, i) -> bool:
        return self.result is not None and i in self.result.proba

    def set_value(self, i, v, source="user"):
        self.values.flat[i] = v
        self.source.flat[i] = source
        self.marked.discard(i)
        self.solution = None
        self.redraw()

    def toggle_mark(self, i):
        if i in self.marked:
            self.marked.discard(i)
        else:
            self.marked.add(i)
        self.redraw()

    def clear_marks(self):
        self.marked.clear()
        self.redraw()

    def clear_all(self):
        self.values[:] = 0
        self.source[:] = "user"
        self.marked.clear()
        self.solution = None
        self.redraw()

    def show_solution(self, solution):
        self.solution = solution
        self.redraw()

    def clear_solution(self):
        self.solution = None
        self.redraw()

    def reconsider(self) -> tuple[dict, list]:
        """Đổi các ô đánh dấu sang lựa chọn kế tiếp. Trả về (thay đổi, các ô không còn lựa chọn)."""
        res = self.result
        if res is None or not res.proba:
            return {}, sorted(self.marked)
        for i in self.marked:
            self.rejected.setdefault(i, set()).add(int(self.values.flat[i]))
        changes = review.reconsider(self.values, self.marked, res.classes, res.proba, self.rejected)
        for i, v in changes.items():
            self.values.flat[i] = v
            self.source.flat[i] = "alt"
        stuck = sorted(i for i in self.marked if i not in changes)
        self.solution = None
        self.redraw()
        return changes, stuck

    def summary(self) -> review.ReviewSummary:
        res = self.result
        if res is None or res.confidence is None:
            return review.ReviewSummary(conflicts=review.conflicts(np.where(self.values > 0, self.values, 0)))
        skip = {divmod(i, 9) for i in range(81) if self.source.flat[i] != "auto"}
        return review.summarize(self.values, res.confidence, res.proba, res.classes, skip)

    def cell_info(self, i) -> str:
        r, c = divmod(i, 9)
        v = int(self.values.flat[i])
        cur = "trống" if v == 0 else ("chưa nhận dạng" if v < 0 else str(v))
        text = f"Ô hàng {r + 1}, cột {c + 1}: đang là {cur}"
        if self.has_alternatives(i):
            opts = review.ranked_options(self.result.classes, self.result.proba[i])[:4]
            text += " · máy đọc: " + ", ".join(f"{k if k else 'trống'} {p:.0%}" for k, p in opts)
        else:
            text += " · ô không có nét mực, chỉ sửa được bằng cách gõ số"
        if self.rejected.get(i):
            text += " · đã loại: " + ", ".join(str(k) if k else "trống" for k in sorted(self.rejected[i]))
        return text

    # ---------- vẽ ----------
    def _xy(self, r, c):
        return self.PAD + c * self.CELL, self.PAD + r * self.CELL

    def redraw(self):
        self.delete("all")
        cs, pad = self.CELL, self.PAD
        if self._photo is not None:
            self.create_image(pad, pad, image=self._photo, anchor="nw")
        s = self.summary()
        low, missed = set(s.low_confidence), set(s.maybe_missed)
        for i in range(81):
            r, c = divmod(i, 9)
            x, y = self._xy(r, c)
            v = int(self.values[r, c])
            if (r, c) in s.conflicts:
                self.create_rectangle(x + 1, y + 1, x + cs - 1, y + cs - 1, fill="#fde2e1", outline="", stipple="gray50")
            if v != 0:
                src = self.source[r, c]
                if (r, c) in s.conflicts:
                    fill = "#c5221f"
                elif src in ("user", "alt"):
                    fill = "#2456c9"
                elif src == "fixed":
                    fill = "#8e2a8a"
                elif v < 0 or (r, c) in low:
                    fill = "#d56e00"
                else:
                    fill = "#2e7d32"
                self.create_rectangle(x + 2, y + 2, x + 22, y + 22, fill=fill, outline="")
                self.create_text(x + 12, y + 12, text="?" if v < 0 else str(v), fill="white",
                                 font=("Segoe UI", 11, "bold"))
            elif self.solution is not None:
                self.create_text(x + cs / 2, y + cs / 2, text=str(int(self.solution[r, c])), fill="#2e8b3d",
                                 font=("Segoe UI", 18, "bold"))
            elif (r, c) in missed:
                self.create_rectangle(x + 2, y + 2, x + 22, y + 22, fill="#ffffff", outline="#d56e00", dash=(2, 2))
                self.create_text(x + 12, y + 12, text="?", fill="#d56e00", font=("Segoe UI", 11, "bold"))
            if i in self.marked:
                self.create_rectangle(x + 3, y + 3, x + cs - 3, y + cs - 3, outline="#c5221f", width=3)
                self.create_text(x + cs - 9, y + cs - 10, text="✗", fill="#c5221f", font=("Segoe UI", 11, "bold"))
        for k in range(10):
            w = 2 if k % 3 == 0 else 1
            color = "#1d2126" if k % 3 == 0 else "#9aa1a9"
            self.create_line(pad + k * cs, pad, pad + k * cs, pad + 9 * cs, fill=color, width=w)
            self.create_line(pad, pad + k * cs, pad + 9 * cs, pad + k * cs, fill=color, width=w)
        if self.selected is not None:
            r, c = divmod(self.selected, 9)
            x, y = self._xy(r, c)
            self.create_rectangle(x + 1, y + 1, x + cs - 1, y + cs - 1, outline="#2456c9", width=2)
        self.on_change(s)

    # ---------- tương tác ----------
    def _cell_at(self, event):
        c = (event.x - self.PAD) // self.CELL
        r = (event.y - self.PAD) // self.CELL
        return r * 9 + c if 0 <= r < 9 and 0 <= c < 9 else None

    def _on_click(self, event):
        self.focus_set()
        i = self._cell_at(event)
        if i is not None:
            self.selected = i
            self.redraw()

    def _on_right_click(self, event):
        self.focus_set()
        i = self._cell_at(event)
        if i is not None:
            self.selected = i
            self.toggle_mark(i)

    def _on_key(self, event):
        if self.selected is None:
            self.selected = 0
        i = self.selected
        r, c = divmod(i, 9)
        key = event.keysym
        moves = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
        if key in moves:
            dr, dc = moves[key]
            self.selected = ((r + dr) % 9) * 9 + (c + dc) % 9
            self.redraw()
        elif event.char in "123456789" and event.char:
            self.set_value(i, int(event.char))
        elif key in ("0", "Delete", "BackSpace", "period"):
            self.set_value(i, 0)
        elif key.lower() in ("x", "m"):
            self.toggle_mark(i)
        else:
            return None
        return "break"


class WebcamDialog(tk.Toplevel):
    """Xem trước webcam, vẽ khung lưới tìm được theo thời gian thực; bấm Chụp (hoặc Space) để lấy ảnh."""

    def __init__(self, master, on_capture, camera_index: int = 0):
        super().__init__(master)
        self.title("Chụp từ webcam")
        self.geometry("860x640")
        self.on_capture = on_capture
        self.frame = None
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
        self.view = ImageView(self, "Đang mở webcam…")
        self.view.pack(fill="both", expand=True)
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")
        self.hint = ttk.Label(bar, text="Đưa cả lưới Sudoku vào khung hình.", style="Status.TLabel")
        self.hint.pack(side="left")
        ttk.Button(bar, text="Đóng", command=self.close).pack(side="right")
        ttk.Button(bar, text="Chụp (Space)", command=self.capture).pack(side="right", padx=6)
        self.bind("<space>", lambda e: self.capture())
        self.bind("<Escape>", lambda e: self.close())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.grab_set()
        if not self.cap.isOpened():
            self.view.placeholder = "Không mở được webcam. Kiểm tra camera hoặc quyền truy cập camera của Windows."
            self.view.redraw()
            return
        self.after(30, self.tick)

    def tick(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
            # Tìm lưới trên ảnh nhỏ cho nhanh, rồi vẽ khung lên ảnh xem trước
            small, scale = preprocess.resize_max_side(frame, 480)
            binary = preprocess.binarize(preprocess.to_gray(small), block=15, c=3, blur_ksize=5)
            binary = cv2.dilate(binary, np.ones((3, 3), np.uint8))
            corners, _, _ = gridmod.find_grid(binary, 0.08)
            preview = frame.copy()
            if corners is not None:
                pts = np.round(corners / scale).astype(np.int32)
                cv2.polylines(preview, [pts], True, (60, 170, 40), 3, cv2.LINE_AA)
            self.hint.configure(text="Đã thấy lưới: bấm Chụp." if corners is not None
                                else "Đưa cả lưới Sudoku vào khung hình.")
            self.view.set_image(preview)
        self.after(30, self.tick)

    def capture(self):
        if self.frame is None:
            return
        frame = self.frame.copy()
        self.close()
        self.on_capture(frame)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.grab_release()
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x760")
        self.minsize(980, 620)
        self.pipeline = SudokuPipeline()
        self.result = None
        self.image = None
        self.image_path = None
        self._build_ui()
        self.after(100, self.load_default_model)

    # ---------- giao diện ----------
    def _build_ui(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Status.TLabel", foreground="#5b626b")

        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(fill="x")
        ttk.Button(bar, text="Mở ảnh…", command=self.open_image).pack(side="left")
        ttk.Button(bar, text="Ảnh mẫu ngẫu nhiên", command=self.open_sample).pack(side="left", padx=6)
        ttk.Button(bar, text="Webcam…", command=self.open_webcam).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Chạy lại", command=self.run_pipeline).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(bar, text="Chọn model…", command=self.choose_model).pack(side="left")
        ttk.Button(bar, text="Lưu kết quả…", command=self.save_result).pack(side="left", padx=6)
        self.model_label = ttk.Label(bar, text="Model: đang tìm…", style="Status.TLabel")
        self.model_label.pack(side="right")

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        self.tabs = ttk.Notebook(body)
        self.views = {}
        for key, title in STAGE_TITLES.items():
            view = ImageView(self.tabs, "Mở một ảnh Sudoku để bắt đầu." if key == "original" else "Chưa có kết quả cho bước này.")
            self.tabs.add(view, text=title)
            self.views[key] = view
        body.add(self.tabs, weight=3)

        side = ttk.Frame(body, padding=(12, 4))
        body.add(side, weight=1)
        ttk.Label(side, text="Kiểm tra kết quả nhận dạng", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(side, text="So số ở góc mỗi ô với chữ trong ảnh. Ô sai: chuột phải (hoặc phím X) để đánh dấu, "
                             "rồi bấm Nhận dạng lại (R); vẫn sai thì bấm tiếp để thử lựa chọn kế tiếp. "
                             "Biết số đúng thì gõ 1–9, phím 0/Delete để xoá.",
                  wraplength=460, style="Status.TLabel").pack(anchor="w", pady=(2, 4))
        legend = ttk.Frame(side)
        legend.pack(anchor="w", pady=(0, 4))
        for color, text in (("#2e7d32", "chắc chắn"), ("#d56e00", "nghi ngờ"), ("#c5221f", "trùng số"),
                            ("#8e2a8a", "máy tự sửa"), ("#2456c9", "đã sửa/chọn lại")):
            tk.Label(legend, text="  ", background=color).pack(side="left", padx=(0, 3))
            ttk.Label(legend, text=text, style="Status.TLabel").pack(side="left", padx=(0, 10))
        self.review_label = ttk.Label(side, text="", wraplength=460, font=("Segoe UI", 10, "bold"))
        self.review_label.pack(anchor="w", pady=(0, 4))
        self.board = ReviewBoard(side, self.on_board_change)
        self.board.pack(anchor="w")
        self.board.bind("<KeyPress-r>", lambda e: self.reconsider_marked())
        self.board.bind("<Return>", lambda e: self.solve_from_board())
        self.cell_label = ttk.Label(side, text="", wraplength=460, style="Status.TLabel")
        self.cell_label.pack(anchor="w", pady=(4, 0))
        btns = ttk.Frame(side)
        btns.pack(anchor="w", pady=(8, 4))
        ttk.Button(btns, text="Nhận dạng lại ô đánh dấu (R)", command=self.reconsider_marked).pack(side="left")
        ttk.Button(btns, text="Bỏ đánh dấu", command=self.board.clear_marks).pack(side="left", padx=6)
        ttk.Button(btns, text="Giải (Enter)", command=self.solve_from_board).pack(side="left")
        btns2 = ttk.Frame(side)
        btns2.pack(anchor="w")
        ttk.Button(btns2, text="Xoá đáp án", command=self.board.clear_solution).pack(side="left")
        ttk.Button(btns2, text="Xoá lưới", command=self.board.clear_all).pack(side="left", padx=6)
        ttk.Button(btns2, text="Lưu làm dữ liệu train", command=self.save_training_cells).pack(side="left")

        self.message = ttk.Label(side, text="", wraplength=460, font=("Segoe UI", 10, "bold"))
        self.message.pack(anchor="w", pady=(6, 4))
        self.timing = ttk.Label(side, text="", wraplength=460, style="Status.TLabel", justify="left")
        self.timing.pack(anchor="w")

        self.status = ttk.Label(self, text="Sẵn sàng.", style="Status.TLabel", padding=(10, 4))
        self.status.pack(fill="x", side="bottom")

    # ---------- model ----------
    def load_default_model(self):
        path = paths.find_model()
        if path is None:
            self.model_label.configure(text="Model: chưa có (chạy scripts/train.py)")
            self.set_status("Chưa có model nhận dạng chữ số. Vẫn chạy được tìm lưới, cắt ô; có thể nhập số bằng tay để giải.")
            return
        self.load_model(path)

    def load_model(self, path: Path):
        try:
            clf = DigitClassifier.load(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Không đọc được model:\n{e}")
            return
        self.pipeline.classifier = clf
        val = clf.meta.get("val", {})
        acc = val.get("mean_accuracy", clf.meta.get("val_accuracy"))
        extra = f", val acc {acc:.3f}" if isinstance(acc, float) else ""
        self.model_label.configure(text=f"Model: {clf.name.upper()} + {clf.feature.upper()}{extra}")
        self.set_status(f"Đã nạp model {path}")
        if self.image is not None:
            self.run_pipeline()

    def choose_model(self):
        path = filedialog.askopenfilename(title="Chọn model", filetypes=[("Model", "*.joblib")])
        if path:
            self.load_model(Path(path))

    # ---------- ảnh ----------
    def open_image(self):
        path = filedialog.askopenfilename(title="Chọn ảnh Sudoku", filetypes=IMAGE_TYPES)
        if path:
            self.open_path(Path(path))

    def open_sample(self):
        folder = paths.find_sample_dir()
        files = sorted([*folder.glob("*.jpg"), *folder.glob("*.png")]) if folder else []
        if not files:
            messagebox.showinfo(APP_TITLE, "Không thấy ảnh mẫu trong data/raw/v2_test.\nChạy scripts/download_data.py trước.")
            return
        self.open_path(random.choice(files))

    def open_path(self, path: Path):
        # Ảnh trong dataset (có file .dat cạnh bên): bỏ qua thẻ xoay EXIF để khớp nhãn
        in_dataset = path.suffix.lower() == ".jpg" and path.with_suffix(".dat").exists()
        image = imread(path, ignore_orientation=in_dataset)
        if image is None:
            messagebox.showerror(APP_TITLE, f"Không đọc được ảnh:\n{path}")
            return
        self.image, self.image_path = image, path
        self.run_pipeline()

    def open_webcam(self):
        WebcamDialog(self, self.on_webcam_capture)

    def on_webcam_capture(self, frame):
        self.image, self.image_path = frame, Path(f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.png")
        self.run_pipeline()

    def run_pipeline(self):
        if self.image is None:
            return
        self.set_status(f"Đang xử lý {self.image_path.name}…")
        self.update_idletasks()
        try:
            self.result = self.pipeline.run(self.image)
        except Exception:  # noqa: BLE001
            messagebox.showerror(APP_TITLE, traceback.format_exc())
            return
        self.show_result()
        self.set_status(f"{self.image_path}")

    def show_result(self):
        res = self.result
        for key, view in self.views.items():
            view.set_image(res.stages.get(key))
        self.board.reset(res)
        if res.solve and res.solve.ok:
            self.board.show_solution(res.solve.solution)
            self.tabs.select(self.views["solution"])
        elif res.grid_found:
            self.tabs.select(self.views["recognized"])
        else:
            self.tabs.select(self.views["contour"])
        self.show_message()

    def show_message(self):
        res = self.result
        ok = bool(res.solve and res.solve.ok)
        color = (COLOR_LOW if res.solve.unique is False else COLOR_SOLVED) if ok else COLOR_ERROR
        self.message.configure(text=res.message, foreground=color)
        names = {"preprocess": "Tiền xử lý", "find_grid": "Tìm lưới", "cells": "Nắn + cắt ô",
                 "recognize": "Nhận dạng", "solve": "Giải"}
        lines = [f"{names[k]}: {v * 1000:.0f} ms" for k, v in res.timings.items() if k in names]
        if res.solve and res.solve.nodes:
            lines.append(f"Số nút tìm kiếm: {res.solve.nodes}")
        self.timing.configure(text="\n".join(lines))

    def solve_from_board(self):
        if self.result is None or not self.result.grid_found:
            messagebox.showinfo(APP_TITLE, "Hãy mở một ảnh có lưới Sudoku trước.")
            return
        given = self.board.grid_values()
        self.board.clear_solution()
        # Lưới người dùng đã xem/sửa: giải đúng như nhập, không tự sửa nữa
        solve = self.pipeline.solve(self.result, given, auto_correct=False)
        self.views["solution"].set_image(self.result.stages.get("solution"))
        if solve.ok:
            self.board.show_solution(solve.solution)
            self.tabs.select(self.views["solution"])
        self.show_message()

    def reconsider_marked(self):
        if not self.board.marked:
            messagebox.showinfo(APP_TITLE, "Chưa có ô nào được đánh dấu.\n\n"
                                "Chuột phải vào ô bị nhận sai (hoặc chọn ô rồi bấm X), sau đó bấm Nhận dạng lại.")
            return
        changes, stuck = self.board.reconsider()
        parts = [f"({i // 9 + 1},{i % 9 + 1}) → {v or 'trống'}" for i, v in sorted(changes.items())]
        text = "Đã thử lựa chọn kế tiếp: " + ", ".join(parts) + "." if parts else ""
        if stuck:
            cells = ", ".join(f"({i // 9 + 1},{i % 9 + 1})" for i in stuck)
            text += f" Ô {cells} không còn lựa chọn nào khác: hãy gõ số đúng."
        self.set_status(text)
        if self.result is not None and self.result.grid_found and changes:
            self.solve_from_board()
            if self.result.solve and self.result.solve.ok:
                self.message.configure(text="Đã giải với các lựa chọn mới. Nếu ô đánh dấu nay đã đúng thì bỏ đánh dấu; "
                                       "nếu vẫn sai, bấm Nhận dạng lại lần nữa.", foreground=COLOR_SOLVED)

    def on_board_change(self, summary: review.ReviewSummary):
        """Cập nhật dòng tóm tắt các ô đáng ngờ và thông tin ô đang chọn."""
        if not hasattr(self, "board"):  # lần vẽ đầu tiên khi bảng đang được tạo
            return
        parts = []
        if summary.conflicts:
            parts.append(f"{len(summary.conflicts)} ô trùng số (đỏ)")
        if summary.low_confidence:
            parts.append(f"{len(summary.low_confidence)} ô độ tin cậy thấp (cam)")
        if summary.maybe_missed:
            parts.append(f"{len(summary.maybe_missed)} ô có nét mực nhưng bị coi là trống (?)")
        if self.board.marked:
            parts.append(f"{len(self.board.marked)} ô đang đánh dấu sai")
        if self.result is None:
            text, color = "", COLOR_SOLVED
        elif parts:
            text, color = "Cần kiểm tra: " + "; ".join(parts) + ".", COLOR_LOW
        else:
            text, color = "Không thấy dấu hiệu nhận dạng sai, nhưng vẫn nên so lại với ảnh.", COLOR_SOLVED
        self.review_label.configure(text=text, foreground=color)
        sel = self.board.selected
        self.cell_label.configure(text="" if sel is None or self.result is None else self.board.cell_info(sel))

    def save_training_cells(self):
        """Lưu các ô đã cắt với nhãn theo lưới hiện tại (đã sửa tay) để lần train sau học nét chữ này."""
        if self.result is None or not self.result.digits:
            messagebox.showinfo(APP_TITLE, "Hãy mở một ảnh có lưới Sudoku trước.")
            return
        grid = self.board.grid_values()
        n_cells = sum(d is not None for d in self.result.digits)
        if not messagebox.askyesno(APP_TITLE, f"Lưu {n_cells} ô với nhãn theo lưới bên phải?\n\n"
                                   "Hãy chắc chắn mọi số đề bài trên lưới đã đúng: ô nào lưới để trống "
                                   "sẽ được lưu là nhiễu (lớp 0)."):
            return
        root = paths.user_digits_dir()
        count = save_cells(self.result.digits, grid, root)
        messagebox.showinfo(APP_TITLE, f"Đã lưu {count} ô vào\n{root}\n\n"
                            "Chạy lại scripts\\train.py để model học thêm các mẫu này.")
        self.set_status(f"Đã lưu {count} ô vào {root}")

    def save_result(self):
        if self.result is None:
            return
        img = self.result.stages.get("solution")
        key = "solution"
        if img is None:
            current = self.tabs.index(self.tabs.select())
            key = list(STAGE_TITLES)[current]
            img = self.result.stages.get(key)
        if img is None:
            return
        stem = self.image_path.stem if self.image_path else "sudoku"
        path = filedialog.asksaveasfilename(title="Lưu ảnh", defaultextension=".png", initialfile=f"{stem}_{key}.png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            imwrite(path, img)
            self.set_status(f"Đã lưu {path}")

    def set_status(self, text):
        self.status.configure(text=text)


def selftest(image_path: str, out_path: str, model_path: str | None = None) -> int:
    """Chạy pipeline không mở giao diện, ghi kết quả ra file (dùng để kiểm tra bản .exe)."""
    import json
    model_path = model_path or paths.find_model()
    clf = DigitClassifier.load(model_path) if model_path else None
    res = SudokuPipeline(clf).run(imread(image_path, ignore_orientation=True))
    info = {"grid_found": res.grid_found, "message": res.message,
            "grid": None if res.grid is None else res.grid.tolist(),
            "solve": res.solve.status if res.solve else None}
    Path(out_path).write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return 0 if res.grid_found else 1


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--selftest":
        return selftest(*sys.argv[2:5])
    try:  # chữ nét trên màn hình độ phân giải cao
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001
        pass
    app = App()
    # Kéo thả ảnh lên file .exe (hoặc truyền đường dẫn) để mở ngay
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        app.after(300, lambda: app.open_path(Path(sys.argv[1])))
    app.mainloop()


if __name__ == "__main__":
    sys.exit(main())
