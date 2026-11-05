"""Đọc/ghi ảnh an toàn với đường dẫn có dấu tiếng Việt (cv2.imread không đọc được trên Windows)."""
from pathlib import Path

import cv2
import numpy as np


def imread(path, ignore_orientation: bool = False) -> np.ndarray | None:
    """ignore_orientation=True: bỏ qua thẻ xoay EXIF. Cần cho dataset, vì nhãn .dat của
    một số ảnh (ví dụ image1031) ứng với hướng pixel gốc chứ không phải hướng EXIF."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    flags = cv2.IMREAD_COLOR | (cv2.IMREAD_IGNORE_ORIENTATION if ignore_orientation else 0)
    return cv2.imdecode(data, flags)


def imwrite(path, image: np.ndarray) -> bool:
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if ok:
        buf.tofile(str(path))
    return ok
