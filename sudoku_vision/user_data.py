"""Dữ liệu chữ số do người dùng thu thập từ app: ô đã được xác nhận nhãn (sau khi sửa tay) được lưu
thành ảnh PNG để lần train sau học thêm nét chữ viết tay của chính người dùng.

Cấu trúc thư mục: <root>/<nhãn>/<tên>.png, nhãn 0 = nhiễu (ô trống bị nhận nhầm là có số).
"""
import time
from pathlib import Path

import cv2
import numpy as np

from .imageio import imread, imwrite


def save_cells(digits, grid, root) -> int:
    """Lưu các ô có ảnh chữ số với nhãn lấy từ `grid` (0 = nhiễu). Trả về số ô đã lưu."""
    root = Path(root)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    count = 0
    for i, d in enumerate(digits):
        if d is None:
            continue
        label = int(np.asarray(grid).flat[i])
        if label < 0:
            continue
        folder = root / str(label)
        folder.mkdir(parents=True, exist_ok=True)
        imwrite(folder / f"{stamp}_{i // 9}{i % 9}.png", d)
        count += 1
    return count


def load_user_digits(root):
    """Đọc lại các ô đã lưu. Trả về (digits, labels); rỗng nếu chưa có thư mục."""
    digits, labels = [], []
    root = Path(root)
    if not root.is_dir():
        return digits, np.zeros(0, np.int32)
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        for f in sorted(folder.glob("*.png")):
            img = imread(f)
            if img is not None:
                digits.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                labels.append(int(folder.name))
    return digits, np.array(labels, np.int32)
