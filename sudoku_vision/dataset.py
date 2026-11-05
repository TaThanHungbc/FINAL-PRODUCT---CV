"""Đọc dataset wichtounet/sudoku_dataset (bản V2).

Mỗi ảnh imageX.jpg đi kèm imageX.dat:
    dòng 1: tên máy chụp
    dòng 2: thông tin định dạng ảnh
    9 dòng tiếp theo: lưới 9x9, 0 = ô trống
"""
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    name: str
    image_path: Path
    grid: np.ndarray  # (9, 9) int32, 0 = ô trống
    device: str


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.stem)]


def read_dat(path) -> tuple[str, np.ndarray]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = [[int(v) for v in line.split()] for line in lines[2:11]]
    grid = np.array(rows, dtype=np.int32)
    if grid.shape != (9, 9):
        raise ValueError(f"File nhãn sai định dạng: {path}")
    return lines[0], grid


def load_split(root, split: str, limit: int | None = None) -> list[Sample]:
    """split là "train" hoặc "test" (thư mục data/raw/v2_train, v2_test)."""
    folder = Path(root) / f"v2_{split}"
    if not folder.is_dir():
        raise FileNotFoundError(f"Không thấy {folder}. Hãy chạy scripts/download_data.py trước.")
    return load_folder(folder, limit)


def load_folder(folder, limit: int | None = None) -> list[Sample]:
    """Mọi cặp ảnh (.jpg/.png) + nhãn .dat cùng tên trong thư mục, ví dụ samples/."""
    samples = []
    for dat in sorted(Path(folder).glob("*.dat"), key=_natural_key):
        img = next((dat.with_suffix(ext) for ext in (".jpg", ".png", ".jpeg") if dat.with_suffix(ext).exists()), None)
        if img is None:
            continue
        device, grid = read_dat(dat)
        samples.append(Sample(img.name, img, grid, device))
        if limit and len(samples) >= limit:
            break
    return samples


def load_outlines(csv_path) -> dict[str, np.ndarray]:
    """Tọa độ 4 góc lưới (TL, TR, BR, BL) theo pixel ảnh gốc, khóa là tên file."""
    outlines = {}
    path = Path(csv_path)
    if not path.exists():
        return outlines
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts = [float(row[f"p{i}_{a}"]) for i in range(1, 5) for a in ("x", "y")]
            outlines[Path(row["filepath"]).name] = np.array(pts, dtype=np.float32).reshape(4, 2)
    return outlines
