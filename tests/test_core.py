"""Kiểm thử nhanh các thành phần chính. Chạy: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sudoku_vision import correction, features, grid, review, solver  # noqa: E402
from sudoku_vision.dataset import read_dat  # noqa: E402
from sudoku_vision.imageio import imread  # noqa: E402
from sudoku_vision.pipeline import UNKNOWN, SudokuPipeline  # noqa: E402

SAMPLE = ROOT / "samples" / "handwritten_01.png"
PUZZLE = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
])


def is_valid_solution(sol: np.ndarray) -> bool:
    units = [*sol, *sol.T, *(sol[r:r + 3, c:c + 3].ravel() for r in (0, 3, 6) for c in (0, 3, 6))]
    return all(sorted(u.tolist()) == list(range(1, 10)) for u in units)


class SolverTest(unittest.TestCase):
    def test_solves_and_keeps_givens(self):
        res = solver.solve(PUZZLE, check_unique=True)
        self.assertTrue(res.ok)
        self.assertTrue(res.unique)
        self.assertTrue(is_valid_solution(res.solution))
        self.assertTrue((res.solution[PUZZLE > 0] == PUZZLE[PUZZLE > 0]).all())

    def test_duplicate_is_invalid(self):
        bad = PUZZLE.copy()
        bad[0, 2] = 5
        self.assertEqual(solver.solve(bad).status, "invalid")

    def test_missing_given_is_not_unique(self):
        sparse = np.zeros((9, 9), int)
        sparse[0, 0] = 1
        res = solver.solve(sparse, check_unique=True)
        self.assertTrue(res.ok)
        self.assertFalse(res.unique)


class CorrectionTest(unittest.TestCase):
    def test_fixes_low_confidence_misread(self):
        classes = np.arange(10)
        misread = PUZZLE.copy()
        misread[0, 1] = 5  # số 3 bị đọc nhầm thành 5 -> trùng với số 5 cùng hàng
        proba = {}
        for i in np.flatnonzero(misread):
            p = np.full(10, 0.01)
            p[misread.flat[i]] = 0.9
            proba[int(i)] = p / p.sum()
        p = np.full(10, 0.02)
        p[5], p[3] = 0.5, 0.4  # SVM phân vân giữa 5 và 3
        proba[1] = p / p.sum()
        fix = correction.correct(misread, classes, proba)
        self.assertIsNotNone(fix)
        self.assertEqual(fix.changes, [(0, 1, 5, 3)])
        self.assertTrue((fix.grid == PUZZLE).all())


class GeometryTest(unittest.TestCase):
    def test_order_corners(self):
        pts = np.array([[90, 10], [10, 12], [12, 95], [92, 90]], np.float32)
        tl, tr, br, bl = grid.order_corners(pts)
        self.assertEqual(tuple(tl), (10, 12))
        self.assertEqual(tuple(tr), (90, 10))
        self.assertEqual(tuple(br), (92, 90))
        self.assertEqual(tuple(bl), (12, 95))

    def test_hog_length(self):
        self.assertEqual(features.hog(np.zeros((28, 28), np.uint8)).shape, (324,))


@unittest.skipUnless(SAMPLE.exists(), "thiếu samples/handwritten_01.png")
class PipelineTest(unittest.TestCase):
    def test_detects_grid_and_digits_without_model(self):
        _, truth = read_dat(SAMPLE.with_suffix(".dat"))
        res = SudokuPipeline().run(imread(SAMPLE))
        self.assertTrue(res.grid_found)
        # Mọi chữ số đều phải được phát hiện; vài vết nhiễu thừa sẽ do lớp 0 của bộ phân loại loại bỏ
        self.assertTrue((res.grid[truth > 0] == UNKNOWN).all())
        self.assertLessEqual(int(((res.grid == UNKNOWN) & (truth == 0)).sum()), 3)

    @unittest.skipUnless((ROOT / "models" / "digit_model.joblib").exists(), "chưa train model")
    def test_solves_with_model(self):
        from sudoku_vision.classifier import DigitClassifier
        _, truth = read_dat(SAMPLE.with_suffix(".dat"))
        res = SudokuPipeline(DigitClassifier.load(ROOT / "models" / "digit_model.joblib")).run(imread(SAMPLE))
        self.assertTrue(res.solve.ok)
        self.assertTrue((res.solve.solution == solver.solve(truth).solution).all())

    @unittest.skipUnless((ROOT / "models" / "digit_model.joblib").exists(), "chưa train model")
    def test_detects_rotation(self):
        import cv2
        from sudoku_vision.classifier import DigitClassifier
        _, truth = read_dat(SAMPLE.with_suffix(".dat"))
        pipe = SudokuPipeline(DigitClassifier.load(ROOT / "models" / "digit_model.joblib"))
        for deg, code in ((90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180)):
            res = pipe.run(cv2.rotate(imread(SAMPLE), code))
            self.assertEqual(res.rotation, deg)
            self.assertTrue((res.solve.solution == solver.solve(truth).solution).all())


class ReviewTest(unittest.TestCase):
    def test_conflicts(self):
        g = PUZZLE.copy()
        g[0, 2] = 5  # trùng với số 5 ở ô (0, 0)
        self.assertEqual(review.conflicts(g), {(0, 0), (0, 2)})
        self.assertEqual(review.conflicts(PUZZLE), set())

    def test_reconsider_cycles_through_options(self):
        classes = np.arange(10)
        grid_ = PUZZLE.copy()
        grid_[0, 1] = 8  # số 3 bị đọc thành 8; máy xếp hạng 8 > 9 > 3
        p = np.full(10, 0.01)
        p[8], p[9], p[3] = 0.6, 0.2, 0.15
        proba = {1: p / p.sum()}
        rejected = {1: {8}}
        # Số 9 bị trùng (hàng 3 cột 2 đã có 9) nên bỏ qua 9, chọn 3 làm đề hợp lệ và có lời giải duy nhất
        self.assertEqual(review.reconsider(grid_, {1}, classes, proba, rejected), {1: 3})
        # Ô không có ảnh chữ số thì không có lựa chọn thay thế
        self.assertEqual(review.reconsider(grid_, {2}, classes, proba, {}), {})


if __name__ == "__main__":
    unittest.main()
