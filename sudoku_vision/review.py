"""Hỗ trợ người dùng rà soát kết quả nhận dạng: chỉ ra ô đáng ngờ và đề xuất lựa chọn thay thế.

Luồng dùng trong app: người dùng đánh dấu những ô bị nhận sai, bấm "Nhận dạng lại". Mỗi ô đánh dấu
được thay bằng lựa chọn có xác suất cao nhất trong số các nhãn chưa bị từ chối, ưu tiên tổ hợp làm
đề hợp lệ và giải được. Ô vẫn sai thì bấm tiếp để chuyển sang lựa chọn kế tiếp.
"""
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from . import solver
from .correction import has_duplicates

LOW_CONF = 0.6


def ranked_options(classes: np.ndarray, p: np.ndarray) -> list[tuple[int, float]]:
    """Các nhãn của một ô theo xác suất giảm dần, nhãn 0 = ô trống."""
    order = np.argsort(-p)
    return [(int(classes[k]), float(p[k])) for k in order]


def conflicts(grid: np.ndarray) -> set[tuple[int, int]]:
    """Các ô có số bị trùng trong cùng hàng, cột hoặc khối 3x3."""
    bad = set()
    units = [[(r, c) for c in range(9)] for r in range(9)]
    units += [[(r, c) for r in range(9)] for c in range(9)]
    units += [[(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)] for br in (0, 3, 6) for bc in (0, 3, 6)]
    for unit in units:
        seen = {}
        for r, c in unit:
            v = int(grid[r, c])
            if v > 0:
                seen.setdefault(v, []).append((r, c))
        for cells in seen.values():
            if len(cells) > 1:
                bad.update(cells)
    return bad


@dataclass
class ReviewSummary:
    low_confidence: list = field(default_factory=list)  # (hàng, cột)
    conflicts: set = field(default_factory=set)
    maybe_missed: list = field(default_factory=list)  # ô có nét mực nhưng được coi là trống

    @property
    def suspicious(self) -> set:
        return set(self.low_confidence) | self.conflicts | set(self.maybe_missed)


def summarize(grid: np.ndarray, confidence: np.ndarray | None, proba: dict, classes, skip=()) -> ReviewSummary:
    """skip: các ô (hàng, cột) người dùng đã tự sửa, không cần nhắc lại về độ tin cậy."""
    s = ReviewSummary(conflicts=conflicts(np.where(grid > 0, grid, 0)))
    for i, p in proba.items():
        r, c = divmod(i, 9)
        if (r, c) in skip:
            continue
        if grid[r, c] > 0 and confidence is not None and confidence[r, c] < LOW_CONF:
            s.low_confidence.append((r, c))
        elif grid[r, c] == 0 and classes is not None:
            # Ô có nét mực nhưng bị coi là nhiễu: nếu chữ số tốt nhất cũng có xác suất đáng kể thì nhắc người dùng
            digit_p = max(float(q) for k, q in zip(classes, p) if k > 0)
            if digit_p >= 0.15:
                s.maybe_missed.append((r, c))
    return s


def reconsider(grid: np.ndarray, marked, classes: np.ndarray, proba: dict, rejected: dict,
               options_per_cell: int = 3, max_trials: int = 500) -> dict:
    """Chọn nhãn mới cho các ô bị đánh dấu sai.

    rejected: chỉ số ô -> tập nhãn người dùng đã bác bỏ (gồm cả nhãn hiện tại).
    Trả về {chỉ số ô: nhãn mới}. Ô không còn lựa chọn nào (hoặc không có ảnh chữ số) được bỏ qua.
    Thứ tự ưu tiên: tổ hợp làm đề giải được (có lời giải duy nhất càng tốt), rồi tổ hợp không trùng số,
    rồi đơn giản là lựa chọn kế tiếp của từng ô; trong mỗi nhóm chọn tổ hợp có tổng log-xác suất lớn nhất.
    """
    cells, options = [], []
    for i in marked:
        if i not in proba:
            continue
        opts = [(v, p) for v, p in ranked_options(classes, proba[i]) if v not in rejected.get(i, set())]
        if opts:
            cells.append(i)
            options.append(opts[:options_per_cell])
    if not cells:
        return {}

    combos = []
    for picks in itertools.product(*options):
        cost = -sum(math.log(p + 1e-6) for _, p in picks)
        combos.append((cost, [v for v, _ in picks]))
    combos.sort(key=lambda t: t[0])

    best_valid = None
    best_unique = best_solvable = None
    for _, values in combos[:max_trials]:
        g = grid.copy()
        g.flat[cells] = values
        if has_duplicates(g):
            continue
        if best_valid is None:
            best_valid = values
        res = solver.solve(g, max_nodes=20_000, check_unique=True)
        if res.ok and res.unique:
            best_unique = values
            break
        if res.ok and best_solvable is None:
            best_solvable = values
    chosen = best_unique or best_solvable or best_valid or [opts[0][0] for opts in options]
    return dict(zip(cells, chosen))
