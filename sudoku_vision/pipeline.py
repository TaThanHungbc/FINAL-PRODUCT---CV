"""Ghép các bước thành pipeline hoàn chỉnh: ảnh chụp -> lưới 9x9 -> lời giải vẽ lên ảnh."""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import cells, correction, grid, overlay, preprocess, solver
from .classifier import DigitClassifier, decide
from .config import PipelineParams

UNKNOWN = -1  # ô có chữ số nhưng chưa có model để nhận dạng

STAGE_TITLES = {
    "original": "Ảnh gốc",
    "binary": "Nhị phân",
    "contour": "Tìm lưới",
    "warped": "Nắn phẳng",
    "cells": "Các ô",
    "recognized": "Nhận dạng",
    "solution": "Lời giải",
}


@dataclass
class PipelineResult:
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    message: str = ""
    work_image: np.ndarray | None = None
    scale: float = 1.0
    corners: np.ndarray | None = None  # trong toạ độ ảnh đã thu nhỏ
    matrix: np.ndarray | None = None
    digits: list = field(default_factory=list)  # 81 ảnh chữ số hoặc None
    grid: np.ndarray | None = None  # 0 = trống, -1 = chưa nhận dạng (sau bước tự sửa, nếu có)
    recognized_grid: np.ndarray | None = None  # kết quả nhận dạng gốc, trước bước tự sửa
    confidence: np.ndarray | None = None
    classes: np.ndarray | None = None
    proba: dict = field(default_factory=dict)  # chỉ số ô -> vector xác suất theo `classes`
    corrections: list = field(default_factory=list)  # (hàng, cột, số cũ, số mới) do bước tự sửa
    solve: solver.SolveResult | None = None

    @property
    def grid_found(self) -> bool:
        return self.corners is not None

    @property
    def corners_original(self) -> np.ndarray | None:
        return None if self.corners is None else self.corners / self.scale


class SudokuPipeline:
    def __init__(self, classifier: DigitClassifier | None = None, params: PipelineParams | None = None):
        self.classifier = classifier
        self.p = params or PipelineParams()

    def run(self, image: np.ndarray, do_solve: bool = True) -> PipelineResult:
        p, res = self.p, PipelineResult()
        t0 = time.perf_counter()

        # 1. Tiền xử lý
        work, res.scale = preprocess.resize_max_side(image, p.work_max_side)
        res.work_image = work
        res.stages["original"] = work
        gray = preprocess.to_gray(work)
        binary = preprocess.binarize(gray, p.thresh_method, p.thresh_block, p.thresh_c, p.blur_ksize)
        res.stages["binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        t1 = time.perf_counter()

        # 2. Tìm lưới
        if p.grid_dilate_iter:
            binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=p.grid_dilate_iter)
        corners, chosen, candidates = grid.find_grid(binary, p.min_grid_area_ratio)
        res.stages["contour"] = overlay.draw_contours(work, candidates, chosen, corners)
        res.timings["preprocess"] = t1 - t0
        res.timings["find_grid"] = time.perf_counter() - t1
        if corners is None:
            res.message = "Không tìm thấy lưới Sudoku trong ảnh."
            return res
        res.corners = corners

        # 3. Nắn phẳng
        t2 = time.perf_counter()
        warped_gray, res.matrix = grid.warp(gray, corners, p.warp_size)
        warped_bgr, _ = grid.warp(work, corners, p.warp_size)
        res.stages["warped"] = warped_bgr

        # 4. Cắt ô, phát hiện ô trống
        warped_bin = cells.binarize_warped(warped_gray, p)
        res.digits = cells.split_and_extract(warped_bin, p)
        res.stages["cells"] = overlay.digits_montage(res.digits)
        res.timings["cells"] = time.perf_counter() - t2

        # 5. Nhận dạng chữ số
        t3 = time.perf_counter()
        res.grid, res.confidence, res.classes, res.proba = self.recognize(res.digits)
        res.recognized_grid = res.grid.copy()
        res.stages["recognized"] = overlay.draw_recognized(warped_bgr, res.grid, res.confidence, p.cell_size)
        res.timings["recognize"] = time.perf_counter() - t3

        if self.classifier is None:
            res.message = "Chưa có model nhận dạng chữ số: hãy chạy train hoặc nhập số bằng tay rồi giải."
            return res
        if do_solve:
            self.solve(res, res.grid, auto_correct=p.auto_correct)
        return res

    def recognize(self, digits):
        """Trả về (lưới, độ tin cậy, các lớp, xác suất theo ô). Lớp 0 (nhiễu) được coi là ô trống."""
        g = np.zeros((9, 9), np.int32)
        conf = np.ones((9, 9), np.float32)
        idx = [i for i, d in enumerate(digits) if d is not None]
        if not idx:
            return g, conf, None, {}
        if self.classifier is None:
            g.flat[idx] = UNKNOWN
            conf.flat[idx] = 0
            return g, conf, None, {}
        classes, proba = self.classifier.predict_proba([digits[i] for i in idx])
        g.flat[idx], conf.flat[idx] = decide(classes, proba, self.p.noise_min_prob)
        return g, conf, classes, dict(zip(idx, proba))

    def solve(self, res: PipelineResult, given: np.ndarray, auto_correct: bool = True) -> solver.SolveResult:
        """Giải với lưới `given` (có thể là lưới người dùng đã sửa) và vẽ đáp án lên ảnh.

        auto_correct: nếu đề không giải được, thử sửa các ô kém tin cậy bằng luật Sudoku.
        """
        t = time.perf_counter()
        given = np.asarray(given, dtype=np.int32)
        res.corrections = []
        if (given < 0).any():
            res.solve = solver.SolveResult("invalid", None, 0)
            res.message = "Còn ô chưa nhận dạng (?)."
            return res.solve
        res.solve = solver.solve(given, check_unique=True)
        if auto_correct and res.proba and (not res.solve.ok or res.solve.unique is False):
            # Ưu tiên cách sửa cho ra đề có đúng một lời giải; nếu đề gốc vô nghiệm thì chấp nhận cách sửa bất kỳ
            fix = correction.correct(given, res.classes, res.proba, require_unique=True)
            if fix is None and not res.solve.ok:
                fix = correction.correct(given, res.classes, res.proba)
            if fix is not None:
                given, res.corrections = fix.grid, fix.changes
                res.solve = solver.solve(given, check_unique=True)
                res.grid = given
                res.stages["recognized"] = overlay.draw_recognized(
                    res.stages["warped"], given, res.confidence, self.p.cell_size,
                    highlight=[(r, c) for r, c, _, _ in fix.changes])
        res.timings["solve"] = time.perf_counter() - t
        messages = {
            "solved": "Đã giải xong.",
            "invalid": "Đề có số bị trùng trong hàng/cột/khối: kiểm tra lại các ô nhận dạng.",
            "unsolvable": "Đề không có lời giải: có thể nhận dạng sai một số ô.",
            "timeout": "Quá giới hạn tìm kiếm: có thể nhận dạng thiếu hoặc sai số.",
        }
        res.message = messages[res.solve.status]
        if res.corrections:
            fixes = ", ".join(f"({r + 1},{c + 1}): {a or '_'}→{b or '_'}" for r, c, a, b in res.corrections)
            res.message = f"Đã giải sau khi tự sửa {len(res.corrections)} ô theo luật Sudoku: {fixes}."
        if res.solve.ok and res.solve.unique is False:
            res.message += (" Lưu ý: đề có nhiều hơn một lời giải, có thể đã bỏ sót một số đề bài;"
                            " hãy kiểm tra lưới và nhập thêm số còn thiếu.")
        if res.solve.ok:
            res.stages["solution"] = overlay.render_solution(
                res.work_image, res.matrix, given, res.solve.solution, self.p.warp_size)
        else:
            res.stages.pop("solution", None)
        return res.solve
