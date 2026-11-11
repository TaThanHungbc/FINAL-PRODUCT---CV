"""Các tham số của pipeline, gom một chỗ để dễ thử nghiệm và ghi vào báo cáo."""
from dataclasses import dataclass

MODEL_FILENAME = "digit_model.joblib"


@dataclass
class PipelineParams:
    # Ảnh được thu nhỏ để cạnh dài nhất không vượt quá giá trị này
    work_max_side: int = 1000
    # Kích thước ảnh lưới sau khi nắn phẳng (chia hết cho 9)
    warp_size: int = 450
    # Kích thước ảnh chữ số chuẩn hóa đưa vào bộ phân loại
    digit_size: int = 28
    digit_inner: int = 20

    # Tiền xử lý để tìm lưới
    blur_ksize: int = 7
    thresh_method: str = "adaptive"  # "adaptive" hoặc "otsu"
    thresh_block: int = 21
    thresh_c: int = 3
    # Giãn nét để nối các đường kẻ lưới bị đứt (ảnh mờ, nét mảnh)
    grid_dilate_iter: int = 1
    min_grid_area_ratio: float = 0.08

    # Nhị phân hóa ảnh lưới đã nắn để cắt chữ số
    cell_thresh_block: int = 35
    cell_thresh_c: int = 6
    # Tỉ lệ viền ô bị xóa để loại đường kẻ lưới
    cell_border_ratio: float = 0.10
    # Ngưỡng để coi một thành phần liên thông là chữ số
    digit_min_area_ratio: float = 0.015
    digit_min_height_ratio: float = 0.28
    digit_center_ratio: float = 0.25
    # Thành phần nhỏ hơn ngưỡng này bị coi là chấm nhiễu
    digit_noise_area_ratio: float = 0.002
    # Đoạn kẻ lưới còn sót: nét thẳng dài >= line_len_ratio * cạnh ô nằm trong dải sát mép (0 = tắt)
    line_len_ratio: float = 0.7
    line_band_ratio: float = 0.3
    # Khoảng cách tối đa (theo chiều cao ô) để gộp các mảnh của cùng một chữ số
    digit_merge_gap_ratio: float = 0.06

    # Chỉ coi một ô là nhiễu (trống) khi xác suất lớp 0 đạt ngưỡng này; nếu không thì lấy chữ số tốt nhất
    noise_min_prob: float = 0.7
    # Tự sửa ô nhận dạng sai bằng luật Sudoku khi đề không giải được
    auto_correct: bool = True

    @property
    def cell_size(self) -> int:
        return self.warp_size // 9
