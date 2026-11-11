"""Bước 5: bộ phân loại 10 lớp (0 = nhiễu, 1-9 = chữ số) bằng k-NN hoặc SVM (scikit-learn)."""
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import features

MODEL_NAMES = ("knn", "svm")


def build_model(name: str):
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    raise ValueError(f"Model không hỗ trợ: {name}")


def decide(classes: np.ndarray, proba: np.ndarray, noise_min_prob: float = 0.0):
    """Chọn nhãn từ ma trận xác suất. Lớp 0 (nhiễu) chỉ được chọn khi xác suất >= noise_min_prob,
    nếu không thì lấy chữ số có xác suất cao nhất: bỏ sót một số đề bài làm hỏng cả đề, còn một
    "chữ số" thừa từ vết bẩn vẫn có thể được bước tự sửa bằng luật Sudoku loại bỏ."""
    idx = proba.argmax(axis=1)
    if noise_min_prob and 0 in classes:
        noise = classes == 0
        digit_idx = np.flatnonzero(~noise)[proba[:, ~noise].argmax(axis=1)]
        weak = noise[idx] & (proba[np.arange(len(idx)), idx] < noise_min_prob)
        idx = np.where(weak, digit_idx, idx)
    return classes[idx], proba[np.arange(len(idx)), idx]


class DigitClassifier:
    def __init__(self, model, feature: str = "hog", meta: dict | None = None):
        self.model = model
        self.feature = feature
        self.meta = meta or {}

    def fit(self, digits, labels) -> "DigitClassifier":
        self.model.fit(features.extract(digits, self.feature), np.asarray(labels))
        return self

    def predict_proba(self, digits) -> tuple[np.ndarray, np.ndarray]:
        """Trả về (các lớp, ma trận xác suất N x số lớp)."""
        classes = self.model.classes_.astype(np.int32)
        if len(digits) == 0:
            return classes, np.zeros((0, len(classes)), np.float32)
        x = features.extract(digits, self.feature)
        return classes, self.model.predict_proba(x).astype(np.float32)

    def predict(self, digits, noise_min_prob: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Trả về (nhãn, độ tin cậy) cho từng ảnh chữ số. Xem `decide` về noise_min_prob."""
        classes, proba = self.predict_proba(digits)
        if len(proba) == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        return decide(classes, proba, noise_min_prob)

    @property
    def name(self) -> str:
        return self.meta.get("name", type(self.model[-1]).__name__)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.meta, "sklearn_version": sklearn.__version__}
        joblib.dump({"model": self.model, "feature": self.feature, "meta": meta}, path)

    @classmethod
    def load(cls, path) -> "DigitClassifier":
        data = joblib.load(path)
        trained = data.get("meta", {}).get("sklearn_version")
        if trained and trained.rsplit(".", 1)[0] != sklearn.__version__.rsplit(".", 1)[0]:
            warnings.warn(f"Model train bằng scikit-learn {trained}, đang chạy {sklearn.__version__}: "
                          "nên train lại bằng môi trường .venv.")
        return cls(data["model"], data["feature"], data.get("meta"))
