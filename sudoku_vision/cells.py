"""Bước 4: cắt 81 ô, phát hiện ô trống và chuẩn hóa ảnh chữ số."""
import cv2
import numpy as np

from .config import PipelineParams


def binarize_warped(warped_gray: np.ndarray, p: PipelineParams) -> np.ndarray:
    blur = cv2.GaussianBlur(warped_gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, p.cell_thresh_block | 1, p.cell_thresh_c)


def remove_grid_lines(inner: np.ndarray, p: PipelineParams) -> np.ndarray:
    """Xoá đoạn kẻ lưới còn sót trong ô (khi lưới nắn hơi lệch): nét ngang dài ở dải trên/dưới
    và nét dọc dài ở dải trái/phải. Nét chữ số nằm giữa ô nên không bị ảnh hưởng."""
    h, w = inner.shape
    length = max(3, int(p.line_len_ratio * h))
    band = int(p.line_band_ratio * h)
    horiz = cv2.morphologyEx(inner, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    vert = cv2.morphologyEx(inner, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    lines = np.zeros_like(inner)
    lines[:band] |= horiz[:band]
    lines[-band:] |= horiz[-band:]
    lines[:, :band] |= vert[:, :band]
    lines[:, -band:] |= vert[:, -band:]
    if not lines.any():
        return inner
    return cv2.subtract(inner, cv2.dilate(lines, np.ones((3, 3), np.uint8)))


def extract_digit(cell: np.ndarray, p: PipelineParams) -> np.ndarray | None:
    """Tìm chữ số trong một ô nhị phân. Trả về ảnh cắt sát chữ số, hoặc None nếu ô trống."""
    h, w = cell.shape
    m = max(1, int(p.cell_border_ratio * h))
    inner = cell.copy()
    inner[:m, :] = 0
    inner[-m:, :] = 0
    inner[:, :m] = 0
    inner[:, -m:] = 0
    if p.line_len_ratio:
        inner = remove_grid_lines(inner, p)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(inner, connectivity=8)
    lo, hi = p.digit_center_ratio, 1 - p.digit_center_ratio
    # Bỏ chấm nhiễu nhỏ và đường kẻ lưới còn sót
    comps = [i for i in range(1, n)
             if stats[i, 4] >= p.digit_noise_area_ratio * h * w
             and stats[i, 2] <= 0.9 * w and stats[i, 3] <= 0.9 * h]
    # Thành phần chính: lớn nhất trong số các thành phần có tâm nằm giữa ô
    central = [i for i in comps if lo * w < centroids[i][0] < hi * w and lo * h < centroids[i][1] < hi * h]
    if not central:
        return None
    main = max(central, key=lambda i: stats[i, 4])
    # Gộp các mảnh nằm sát thành phần chính: nét bút chì mờ hay bị đứt thành nhiều mảnh
    x0, y0 = stats[main, 0], stats[main, 1]
    x1, y1 = x0 + stats[main, 2], y0 + stats[main, 3]
    gap = p.digit_merge_gap_ratio * h
    keep = [main]
    for i in sorted(comps, key=lambda i: -stats[i, 4]):
        if i == main:
            continue
        bx0, by0 = stats[i, 0], stats[i, 1]
        bx1, by1 = bx0 + stats[i, 2], by0 + stats[i, 3]
        if bx0 <= x1 + gap and bx1 >= x0 - gap and by0 <= y1 + gap and by1 >= y0 - gap:
            keep.append(i)
            x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
    area = sum(stats[i, 4] for i in keep)
    if area < p.digit_min_area_ratio * h * w or (y1 - y0) < p.digit_min_height_ratio * h:
        return None
    return np.where(np.isin(labels[y0:y1, x0:x1], keep), 255, 0).astype(np.uint8)


def normalize_digit(digit: np.ndarray, size: int = 28, inner: int = 20) -> np.ndarray:
    """Đặt chữ số (nét trắng trên nền đen) vào giữa khung size x size, giữ tỉ lệ."""
    h, w = digit.shape
    scale = inner / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def split_and_extract(warped_bin: np.ndarray, p: PipelineParams) -> list[np.ndarray | None]:
    """Trả về list 81 phần tử (theo hàng), mỗi phần tử là ảnh chữ số đã chuẩn hóa hoặc None."""
    cs = p.cell_size
    digits = []
    for r in range(9):
        for c in range(9):
            cell = warped_bin[r * cs:(r + 1) * cs, c * cs:(c + 1) * cs]
            d = extract_digit(cell, p)
            digits.append(None if d is None else normalize_digit(d, p.digit_size, p.digit_inner))
    return digits


def binarize_glyph(gray: np.ndarray) -> np.ndarray | None:
    """Dùng cho ảnh chữ số rời (ví dụ Chars74K: chữ đen trên nền trắng) -> ảnh cắt sát chữ số."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(th)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    return th[y:y + h, x:x + w]
