"""Bước 1: tiền xử lý (thu nhỏ, ảnh xám, làm mờ, nhị phân hóa)."""
import cv2
import numpy as np


def resize_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image, 1.0
    size = (round(w * scale), round(h * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray, method: str = "adaptive", block: int = 15, c: int = 3,
             blur_ksize: int = 7) -> np.ndarray:
    """Trả về ảnh nhị phân đảo: nét mực = 255, nền giấy = 0."""
    if blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    if method == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block | 1, c)
    return th
