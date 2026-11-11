"""Tìm thư mục dự án, model và ảnh mẫu, kể cả khi chạy từ file .exe."""
import sys
from pathlib import Path

from .config import MODEL_FILENAME


def app_dir() -> Path:
    if getattr(sys, "frozen", False):  # đang chạy từ file .exe (PyInstaller)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _candidates(*parts: str) -> list[Path]:
    # Cạnh file exe, thư mục cha (exe nằm trong dist/), hoặc thư mục làm việc hiện tại
    bases = [app_dir(), app_dir().parent, Path.cwd()]
    return [b.joinpath(*parts) for b in bases]


def find_model() -> Path | None:
    candidates = _candidates("models", MODEL_FILENAME)
    # Model đóng gói sẵn trong exe (PyInstaller --add-data) được dùng khi không có model bên ngoài
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "models" / MODEL_FILENAME)
    return next((p for p in candidates if p.is_file()), None)


def find_sample_dir() -> Path | None:
    dirs = _candidates("data", "raw", "v2_test") + _candidates("samples")
    return next((p for p in dirs if p.is_dir()), None)


def user_digits_dir() -> Path:
    """Nơi app lưu ô chữ số người dùng đã xác nhận: data/user_digits của dự án nếu có, không thì cạnh exe."""
    existing = next((p for p in _candidates("data") if p.is_dir()), None)
    return (existing or app_dir() / "data") / "user_digits"
