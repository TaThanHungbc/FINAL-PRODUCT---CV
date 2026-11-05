"""Tạo dữ liệu train cho bộ phân loại 10 lớp: 0 = nhiễu (không phải chữ số), 1-9 = chữ số.

Nguồn dữ liệu:
- Ô cắt từ ảnh Sudoku của dataset (chữ in thật), nhãn lấy từ file .dat. Ô trống mà bước tách ô
  vẫn nhận là có chữ số (vết bẩn, nhiễu) trở thành mẫu lớp 0.
- Chars74K EnglishFnt: chữ số in từ nhiều font máy tính.
- MNIST: chữ số viết tay.
"""
import gzip
from pathlib import Path

import cv2
import numpy as np

from .cells import binarize_glyph, normalize_digit
from .imageio import imread
from .pipeline import SudokuPipeline

NOISE = 0


def collect_from_dataset(samples, pipeline: SudokuPipeline, log=print):
    """Trả về (digits, labels, thống kê). Ô có nhãn > 0 thành chữ số, ô trống bị nhận nhầm thành lớp 0."""
    digits, labels = [], []
    stats = {"images": len(samples), "grid_not_found": [], "missed_digits": 0, "false_digits": 0}
    for s in samples:
        image = imread(s.image_path, ignore_orientation=True)
        res = pipeline.run(image, do_solve=False) if image is not None else None
        if res is None or not res.grid_found:
            stats["grid_not_found"].append(s.name)
            continue
        for i, d in enumerate(res.digits):
            label = int(s.grid.flat[i])
            if d is not None:
                digits.append(d)
                labels.append(label)
                stats["false_digits"] += label == NOISE
            elif label:
                stats["missed_digits"] += 1
    log(f"  {len(digits)} mẫu từ {len(samples) - len(stats['grid_not_found'])}/{len(samples)} ảnh "
        f"(bỏ sót {stats['missed_digits']} chữ số, {stats['false_digits']} mẫu nhiễu)")
    return digits, np.array(labels, np.int32), stats


def collect_chars74k(root, size: int = 28, inner: int = 20, per_class: int | None = None):
    """Đọc chữ số 1-9 từ Chars74K EnglishFnt (thư mục Sample002..Sample010)."""
    digits, labels = [], []
    for label in range(1, 10):
        folder = Path(root) / f"Sample{label + 1:03d}"
        files = sorted(folder.glob("*.png"))[:per_class]
        for f in files:
            gray = cv2.cvtColor(imread(f), cv2.COLOR_BGR2GRAY)
            glyph = binarize_glyph(gray)
            if glyph is not None:
                digits.append(normalize_digit(glyph, size, inner))
                labels.append(label)
    return digits, np.array(labels, np.int32)


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def collect_mnist(root, split: str = "train", per_class: int | None = 1000, size: int = 28,
                  inner: int = 20, seed: int = 0):
    """Đọc chữ số viết tay 1-9 từ MNIST, chuẩn hoá giống hệt ô cắt từ ảnh Sudoku."""
    prefix = "train" if split == "train" else "t10k"
    images = _read_idx(Path(root) / f"{prefix}-images-idx3-ubyte.gz")
    labels = _read_idx(Path(root) / f"{prefix}-labels-idx1-ubyte.gz")
    rng = np.random.default_rng(seed)
    digits, out = [], []
    for label in range(1, 10):
        idx = np.flatnonzero(labels == label)
        if per_class:
            idx = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for i in idx:
            pts = cv2.findNonZero((images[i] > 100).astype(np.uint8))
            if pts is None:
                continue
            x, y, w, h = cv2.boundingRect(pts)
            glyph = np.where(images[i][y:y + h, x:x + w] > 100, 255, 0).astype(np.uint8)
            digits.append(normalize_digit(glyph, size, inner))
            out.append(label)
    return digits, np.array(out, np.int32)


def augment(digits, labels, copies: int = 2, seed: int = 0):
    """Tăng cường nhẹ: xoay, dịch, co giãn và làm dày/mảnh nét."""
    rng = np.random.default_rng(seed)
    out_d, out_l = list(digits), list(labels)
    kernel = np.ones((2, 2), np.uint8)
    for d, l in zip(digits, labels):
        h, w = d.shape
        for _ in range(copies):
            m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(0.9, 1.1))
            m[:, 2] += rng.uniform(-1.5, 1.5, size=2)
            a = cv2.warpAffine(d, m, (w, h), borderValue=0)
            op = rng.integers(3)
            if op == 1:
                a = cv2.dilate(a, kernel)
            elif op == 2:
                a = cv2.erode(a, kernel)
            out_d.append(a)
            out_l.append(l)
    return out_d, np.array(out_l, np.int32)
