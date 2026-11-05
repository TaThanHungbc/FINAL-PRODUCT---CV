"""Bước 7: vẽ kết quả (ảnh minh họa từng bước và đáp án đè lên ảnh gốc)."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (60, 170, 40)
BLUE = (200, 90, 20)
ORANGE = (0, 140, 255)
RED = (40, 40, 220)
PURPLE = (160, 40, 150)


def _put_center(img, text, center, scale, color, thickness):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_contours(image, candidates, chosen, corners):
    out = image.copy()
    if candidates:
        cv2.drawContours(out, candidates, -1, ORANGE, 1, cv2.LINE_AA)
    if chosen is not None:
        cv2.drawContours(out, [chosen], -1, BLUE, 2, cv2.LINE_AA)
    if corners is not None:
        pts = np.round(corners).astype(np.int32)
        cv2.polylines(out, [pts], True, GREEN, 3, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(out, (int(x), int(y)), 8, RED, -1, cv2.LINE_AA)
            cv2.putText(out, "TL TR BR BL".split()[i], (int(x) + 10, int(y) - 10), FONT, 0.7, RED, 2, cv2.LINE_AA)
    return out


def digits_montage(digits, tile: int = 48) -> np.ndarray:
    """Lưới 9x9 các ảnh chữ số đã chuẩn hóa; ô trống tô xám."""
    gap = 2
    size = 9 * tile + 10 * gap
    out = np.full((size, size), 90, np.uint8)
    for i, d in enumerate(digits):
        r, c = divmod(i, 9)
        y, x = gap + r * (tile + gap), gap + c * (tile + gap)
        out[y:y + tile, x:x + tile] = 40 if d is None else cv2.resize(d, (tile, tile), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def draw_recognized(warped_bgr, grid, confidence, cell, low_conf: float = 0.6, highlight=()):
    """Ghi số nhận dạng được ở góc mỗi ô; độ tin cậy thấp tô cam, chưa nhận dạng là '?'.
    Các ô trong `highlight` (đã được tự sửa) có viền tím."""
    out = warped_bgr.copy()
    for r, c in highlight:
        cv2.rectangle(out, (c * cell + 1, r * cell + 1), ((c + 1) * cell - 2, (r + 1) * cell - 2), PURPLE, 2)
    for r in range(9):
        for c in range(9):
            v = int(grid[r, c])
            if v == 0:
                continue
            text = "?" if v < 0 else str(v)
            color = RED if v < 0 else (ORANGE if confidence[r, c] < low_conf else GREEN)
            x, y = c * cell, r * cell
            cv2.rectangle(out, (x + 2, y + 2), (x + 20, y + 22), (255, 255, 255), -1)
            cv2.putText(out, text, (x + 5, y + 19), FONT, 0.6, color, 2, cv2.LINE_AA)
    return out


def render_solution(work_bgr, matrix, given, solution, size):
    """Vẽ các số đáp án vào ô trống trên ảnh phẳng rồi warp ngược về ảnh gốc."""
    cell = size // 9
    layer = np.zeros((size, size, 3), np.uint8)
    alpha = np.zeros((size, size), np.uint8)
    scale = cell / 40
    for r in range(9):
        for c in range(9):
            if given[r, c] == 0:
                center = (c * cell + cell / 2, r * cell + cell / 2)
                _put_center(layer, str(int(solution[r, c])), center, scale, GREEN, max(2, int(cell / 20)))
                _put_center(alpha, str(int(solution[r, c])), center, scale, 255, max(2, int(cell / 20)))
    h, w = work_bgr.shape[:2]
    inv = np.linalg.inv(matrix)
    layer_w = cv2.warpPerspective(layer, inv, (w, h))
    alpha_w = cv2.warpPerspective(alpha, inv, (w, h)).astype(np.float32)[..., None] / 255.0
    return (work_bgr * (1 - alpha_w) + layer_w * alpha_w).astype(np.uint8)
