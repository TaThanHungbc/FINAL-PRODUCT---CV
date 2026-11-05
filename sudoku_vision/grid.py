"""Bước 2-3: tìm khung lưới Sudoku và nắn phẳng bằng biến đổi phối cảnh."""
import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Lấy 4 góc TL, TR, BR, BL từ một tập điểm (4 đỉnh tứ giác hoặc cả contour)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _is_reasonable_quad(quad: np.ndarray) -> bool:
    if len(np.unique(quad.round(), axis=0)) < 4:
        return False
    sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    return sides.min() > 0.3 * sides.max()


def find_grid(binary: np.ndarray, min_area_ratio: float = 0.08):
    """Trả về (4 góc TL-TR-BR-BL, contour được chọn, danh sách contour ứng viên)."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    img_area = binary.shape[0] * binary.shape[1]

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_ratio * img_area:
            break
        peri = cv2.arcLength(cnt, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = order_corners(approx)
                if _is_reasonable_quad(quad):
                    return quad, cnt, contours

    # Dự phòng: lấy 4 điểm cực trị của contour lớn nhất khi không xấp xỉ được tứ giác
    if contours and cv2.contourArea(contours[0]) >= min_area_ratio * img_area:
        quad = order_corners(contours[0])
        if _is_reasonable_quad(quad):
            return quad, contours[0], contours
    return None, None, contours


def warp(image: np.ndarray, corners: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Nắn vùng lưới về ảnh vuông size x size. Trả về (ảnh, ma trận M)."""
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (size, size)), matrix


def polygon_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    """IoU giữa hai tứ giác, tính bằng cách tô mask (dùng để đánh giá bước tìm lưới)."""
    ma = np.zeros(shape, np.uint8)
    mb = np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.round(a).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(b).astype(np.int32)], 1)
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0
