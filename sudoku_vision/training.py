"""Chuẩn bị dữ liệu train/validation cho bộ phân loại (dùng chung cho train.py và experiments.py)."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .dataset import load_folder, load_split
from .digits_data import augment, collect_chars74k, collect_from_dataset, collect_mnist
from .pipeline import SudokuPipeline
from .user_data import load_user_digits

# Ô do người dùng lưu từ app thường rất ít nên được nhân bản tăng cường nhiều hơn
USER_AUGMENT_COPIES = 6


@dataclass
class TrainingData:
    sudoku_train: tuple  # (digits, labels) từ ảnh Sudoku dùng để train
    sudoku_val: tuple  # (digits, labels) từ ảnh Sudoku giữ lại để validation
    extras: dict = field(default_factory=dict)  # tên nguồn -> (digits, labels)
    val_sets: dict = field(default_factory=dict)  # tên -> (digits, labels)
    stats: dict = field(default_factory=dict)

    def train_set(self, sources=("sudoku", "chars74k", "mnist", "synthetic", "user"), augment_copies: int = 2,
                  include_val: bool = False):
        """Ghép tập train. Ô Sudoku và ô người dùng được tăng cường; include_val=True để train model cuối cùng."""
        d, l = self.sudoku_train
        if include_val:
            d, l = d + self.sudoku_val[0], np.concatenate([l, self.sudoku_val[1]])
        if "sudoku" not in sources:
            d, l = [], np.zeros(0, np.int32)
        elif augment_copies:
            d, l = augment(d, l, copies=augment_copies)
        d = list(d)
        for name, (xd, xl) in self.extras.items():
            if name in sources:
                if name == "user" and augment_copies:
                    xd, xl = augment(xd, xl, copies=USER_AUGMENT_COPIES)
                d, l = d + list(xd), np.concatenate([l, xl])
        return d, l


def load_training_data(data_root, chars74k=None, mnist=None, mnist_per_class: int = 1500,
                       val_size: float = 0.2, limit: int | None = None, user_digits=None,
                       synthetic=None, synthetic_val=None, log=print) -> TrainingData:
    pipe = SudokuPipeline()
    samples = load_split(data_root, "train", limit)
    # Chia theo ảnh (không theo ô) để validation không lẫn ô của cùng một tờ báo
    tr_s, va_s = train_test_split(samples, test_size=val_size, random_state=42)
    log(f"Cắt ô từ ảnh Sudoku: {len(tr_s)} ảnh train, {len(va_s)} ảnh validation")
    tr_d, tr_l, stats = collect_from_dataset(tr_s, pipe, log)
    va_d, va_l, _ = collect_from_dataset(va_s, pipe, log)
    data = TrainingData((tr_d, tr_l), (va_d, va_l), stats={k: v for k, v in stats.items() if k != "grid_not_found"})
    data.val_sets["sudoku_print"] = (va_d, va_l)
    if chars74k and Path(chars74k).is_dir():
        data.extras["chars74k"] = collect_chars74k(chars74k)
        log(f"  + Chars74K: {len(data.extras['chars74k'][0])} chữ số in")
    if mnist and Path(mnist).is_dir():
        data.extras["mnist"] = collect_mnist(mnist, "train", mnist_per_class)
        log(f"  + MNIST: {len(data.extras['mnist'][0])} chữ số viết tay")
        data.val_sets["mnist_handwritten"] = collect_mnist(mnist, "test", 500, seed=1)
    if synthetic and Path(synthetic).is_dir():
        # Ô cắt từ ảnh Sudoku viết tay tổng hợp: chữ số MNIST đã qua đúng quy trình chụp + tách ô của pipeline
        sd, sl, _ = collect_from_dataset(load_folder(synthetic), pipe, lambda *_: None)
        data.extras["synthetic"] = (sd, sl)
        log(f"  + Sudoku viết tay tổng hợp: {len(sd)} ô")
    if synthetic_val and Path(synthetic_val).is_dir():
        vd, vl, _ = collect_from_dataset(load_folder(synthetic_val), pipe, lambda *_: None)
        data.val_sets["synthetic_handwritten"] = (vd, vl)
    if user_digits:
        ud, ul = load_user_digits(user_digits)
        if len(ud):
            data.extras["user"] = (ud, ul)
            log(f"  + Ô người dùng lưu từ app: {len(ud)} mẫu, theo lớp {np.bincount(ul, minlength=10).tolist()}")
    return data
