"""Trích đặc trưng cho ảnh chữ số 28x28."""
import cv2
import numpy as np

# HOG 28x28: block 14x14, bước 7, cell 7x7, 9 hướng -> 3*3 block * 4 cell * 9 = 324 chiều
_HOG = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)


def hog(digit: np.ndarray) -> np.ndarray:
    return _HOG.compute(digit).ravel()


def raw(digit: np.ndarray) -> np.ndarray:
    return digit.astype(np.float32).ravel() / 255.0


FEATURES = {"hog": hog, "raw": raw}


def extract(digits, kind: str = "hog") -> np.ndarray:
    fn = FEATURES[kind]
    return np.stack([fn(d) for d in digits]).astype(np.float32)
