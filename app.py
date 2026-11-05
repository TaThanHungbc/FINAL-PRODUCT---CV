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
from sudoku_vision import paths, preprocess
from sudoku_vision.classifier import DigitClassifier
from sudoku_vision.imageio import imread, imwrite
from sudoku_vision.pipeline import STAGE_TITLES, UNKNOWN, SudokuPipeline

APP_TITLE = "Sudoku Vision · CS231"
IMAGE_TYPES = [("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")]
COLOR_GIVEN = "#1d2126"
COLOR_LOW = "#c26a00"
COLOR_EDIT = "#2456c9"
COLOR_SOLVED = "#2e8b3d"
COLOR_FIXED = "#8e2a8a"
LOW_CONF = 0.6


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


class SudokuBoard(ttk.Frame):
    """Lưới 9x9 ô nhập: hiện số nhận dạng được, cho phép sửa tay, hiện đáp án."""

    def __init__(self, master):
        super().__init__(master)
        self.vars, self.entries = [], []
        self.solution_mask = np.zeros((9, 9), bool)
        self._updating = False
        vcmd = (self.register(lambda s: s == "" or (len(s) == 1 and s in "123456789")), "%P")
        for br in range(3):
            for bc in range(3):
                box = tk.Frame(self, background="#1d2126", padx=1, pady=1)
                box.grid(row=br, column=bc, padx=2, pady=2)
                for r in range(3):
                    for c in range(3):
                        var = tk.StringVar()
                        e = tk.Entry(box, textvariable=var, width=2, justify="center", font=("Segoe UI", 15, "bold"),
                                     relief="flat", validate="key", validatecommand=vcmd)
                        e.grid(row=r, column=c, padx=1, pady=1, ipady=3)
                        self.vars.append((br * 3 + r, bc * 3 + c, var))
                        self.entries.append(e)
                        var.trace_add("write", lambda *_, rr=br * 3 + r, cc=bc * 3 + c: self._on_edit(rr, cc))
        # Sắp lại theo thứ tự hàng để truy cập theo chỉ số r*9+c
        order = sorted(range(81), key=lambda i: (self.vars[i][0], self.vars[i][1]))
        self.vars = [self.vars[i][2] for i in order]
        self.entries = [self.entries[i] for i in order]

    def _cell(self, r, c):
        return self.vars[r * 9 + c], self.entries[r * 9 + c]

    def _on_edit(self, r, c):
        if self._updating:
            return
        # Người dùng sửa một ô: giữ số vừa gõ, xoá các ô đáp án còn lại
        self.solution_mask[r, c] = False
        self.clear_solution()
        self._cell(r, c)[1].configure(foreground=COLOR_EDIT, background="#ffffff")

    def set_grid(self, grid, confidence=None, fixed=()):
        """fixed: các ô (hàng, cột) đã được tự sửa theo luật Sudoku, tô tím."""
        self._updating = True
        self.solution_mask[:] = False
        for r in range(9):
            for c in range(9):
                var, e = self._cell(r, c)
                v = int(grid[r, c]) if grid is not None else 0
                var.set("" if v <= 0 else str(v))
                low = confidence is not None and v > 0 and confidence[r, c] < LOW_CONF
                bg = "#ffe1e1" if v == UNKNOWN else ("#fff1d6" if low else "#ffffff")
                e.configure(foreground=COLOR_LOW if low else COLOR_GIVEN, background=bg)
        for r, c in fixed:
            self._cell(r, c)[1].configure(foreground=COLOR_FIXED, background="#f6e6f5")
        self._updating = False

    def get_grid(self) -> np.ndarray:
        g = np.zeros((9, 9), np.int32)
        for r in range(9):
            for c in range(9):
                if not self.solution_mask[r, c]:
                    v = self._cell(r, c)[0].get()
                    g[r, c] = int(v) if v else 0
        return g

    def show_solution(self, given, solution):
        self._updating = True
        for r in range(9):
            for c in range(9):
                if given[r, c] == 0:
                    var, e = self._cell(r, c)
                    var.set(str(int(solution[r, c])))
                    e.configure(foreground=COLOR_SOLVED, background="#e9f6ec")
                    self.solution_mask[r, c] = True
        self._updating = False

    def clear_solution(self):
        self._updating = True
        for r, c in zip(*np.nonzero(self.solution_mask)):
            var, e = self._cell(r, c)
            var.set("")
            e.configure(foreground=COLOR_GIVEN, background="#ffffff")
        self.solution_mask[:] = False
        self._updating = False


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
        ttk.Label(side, text="Lưới nhận dạng", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(side, text="Sửa ô sai rồi bấm Giải. Cam: độ tin cậy thấp · Tím: đã tự sửa theo luật Sudoku · "
                             "Đỏ: chưa nhận dạng · Xanh lá: đáp án.",
                  wraplength=330, style="Status.TLabel").pack(anchor="w", pady=(2, 8))
        self.board = SudokuBoard(side)
        self.board.pack(anchor="w")
        btns = ttk.Frame(side)
        btns.pack(anchor="w", pady=10)
        ttk.Button(btns, text="Giải", command=self.solve_from_board).pack(side="left")
        ttk.Button(btns, text="Xoá đáp án", command=self.board.clear_solution).pack(side="left", padx=6)
        ttk.Button(btns, text="Xoá lưới", command=lambda: self.board.set_grid(np.zeros((9, 9), int))).pack(side="left")

        self.message = ttk.Label(side, text="", wraplength=330, font=("Segoe UI", 10, "bold"))
        self.message.pack(anchor="w", pady=(6, 4))
        self.timing = ttk.Label(side, text="", wraplength=330, style="Status.TLabel", justify="left")
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
        self.board.set_grid(res.grid, res.confidence, [(r, c) for r, c, _, _ in res.corrections])
        if res.solve and res.solve.ok:
            self.board.show_solution(res.grid, res.solve.solution)
            self.tabs.select(self.views["solution"])
        elif res.grid_found:
            self.tabs.select(self.views["recognized"])
        else:
            self.tabs.select(self.views["contour"])
        self.show_message()

    def show_message(self):
        res = self.result
        ok = bool(res.solve and res.solve.ok)
        color = (COLOR_LOW if res.solve.unique is False else COLOR_SOLVED) if ok else "#b3261e"
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
        given = self.board.get_grid()
        self.board.clear_solution()
        # Lưới người dùng đã xem/sửa: giải đúng như nhập, không tự sửa nữa
        solve = self.pipeline.solve(self.result, given, auto_correct=False)
        self.views["solution"].set_image(self.result.stages.get("solution"))
        if solve.ok:
            self.board.show_solution(given, solve.solution)
            self.tabs.select(self.views["solution"])
        self.show_message()

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
