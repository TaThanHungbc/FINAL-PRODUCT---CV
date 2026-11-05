"""Dùng luật Sudoku để sửa lỗi nhận dạng.

Khi đề nhận dạng được bị trùng số hoặc không giải được, thử thay một vài ô kém tin cậy nhất
bằng lựa chọn tốt thứ hai, thứ ba của bộ phân loại (kể cả "ô trống"). Các tổ hợp được thử theo
thứ tự tổng log-xác suất giảm dần, và dừng ở tổ hợp đầu tiên cho ra đề giải được.
"""
import itertools
import math
from dataclasses import dataclass

import numpy as np

from . import solver


@dataclass
class Correction:
    grid: np.ndarray
    result: solver.SolveResult
    changes: list[tuple[int, int, int, int]]  # (hàng, cột, số cũ, số mới)
    trials: int


def has_duplicates(grid: np.ndarray) -> bool:
    for unit in (*grid, *grid.T, *(grid[r:r + 3, c:c + 3].ravel() for r in (0, 3, 6) for c in (0, 3, 6))):
        vals = unit[unit > 0]
        if len(vals) != len(set(vals.tolist())):
            return True
    return False


def correct(grid: np.ndarray, classes: np.ndarray, proba: dict[int, np.ndarray],
            max_cells: int = 10, max_changes: int = 3, alternatives: int = 2,
            max_trials: int = 2000, max_nodes: int = 20_000) -> Correction | None:
    """proba: chỉ số ô (r*9+c) -> vector xác suất trên `classes` (lớp 0 = trống/nhiễu)."""
    eps = 1e-4
    # Các ô kém tin cậy nhất là ứng viên để sửa
    cells = sorted(proba, key=lambda i: proba[i].max())[:max_cells]
    options = {}
    for i in cells:
        p = proba[i]
        current = int(grid.flat[i])
        p_cur = p[classes == current][0] if current in classes else p.max()
        alts = [(int(classes[k]), math.log(p_cur + eps) - math.log(p[k] + eps))
                for k in np.argsort(-p) if int(classes[k]) != current][:alternatives]
        if 0 not in [a for a, _ in alts] and current != 0:
            p0 = p[classes == 0][0] if 0 in classes else 0.0
            alts.append((0, math.log(p_cur + eps) - math.log(p0 + eps)))  # luôn cho phép bỏ ô
        options[i] = alts

    combos = []
    for k in range(1, max_changes + 1):
        for chosen in itertools.combinations(cells, k):
            for picks in itertools.product(*(options[i] for i in chosen)):
                combos.append((sum(cost for _, cost in picks), chosen, [v for v, _ in picks]))
    combos.sort(key=lambda t: t[0])

    for trials, (_, chosen, values) in enumerate(combos[:max_trials], start=1):
        g = grid.copy()
        g.flat[list(chosen)] = values
        if has_duplicates(g):
            continue
        res = solver.solve(g, max_nodes=max_nodes)
        if res.ok:
            changes = [(i // 9, i % 9, int(grid.flat[i]), v) for i, v in zip(chosen, values)]
            return Correction(g, res, changes, trials)
    return None
