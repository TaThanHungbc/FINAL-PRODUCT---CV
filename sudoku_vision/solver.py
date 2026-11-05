"""Bước 6: giải Sudoku bằng backtracking + heuristic MRV (chọn ô ít ứng viên nhất), dùng bitmask."""
from dataclasses import dataclass

import numpy as np

ALL = 0b1111111110  # bit 1..9


@dataclass
class SolveResult:
    status: str  # "solved" | "invalid" | "unsolvable" | "timeout"
    solution: np.ndarray | None
    nodes: int

    @property
    def ok(self) -> bool:
        return self.status == "solved"


class _Limit(Exception):
    pass


def solve(grid, max_nodes: int = 200_000) -> SolveResult:
    g = [[int(v) for v in row] for row in np.asarray(grid)]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    empties = []
    for r in range(9):
        for c in range(9):
            v = g[r][c]
            if not v:
                empties.append((r, c))
                continue
            bit, b = 1 << v, (r // 3) * 3 + c // 3
            if (rows[r] | cols[c] | boxes[b]) & bit:
                return SolveResult("invalid", None, 0)  # đề có số trùng
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

    nodes = 0

    def search() -> bool:
        nonlocal nodes
        if not empties:
            return True
        nodes += 1
        if nodes > max_nodes:
            raise _Limit
        best_i, best_mask, best_cnt = -1, 0, 10
        for i, (r, c) in enumerate(empties):
            mask = ALL & ~(rows[r] | cols[c] | boxes[(r // 3) * 3 + c // 3])
            cnt = mask.bit_count()
            if cnt == 0:
                return False
            if cnt < best_cnt:
                best_i, best_mask, best_cnt = i, mask, cnt
                if cnt == 1:
                    break
        r, c = empties.pop(best_i)
        b = (r // 3) * 3 + c // 3
        m = best_mask
        while m:
            bit = m & -m
            m ^= bit
            g[r][c] = bit.bit_length() - 1
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            if search():
                return True
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
        g[r][c] = 0
        empties.insert(best_i, (r, c))
        return False

    try:
        found = search()
    except _Limit:
        return SolveResult("timeout", None, nodes)
    if not found:
        return SolveResult("unsolvable", None, nodes)
    return SolveResult("solved", np.array(g, dtype=np.int32), nodes)
