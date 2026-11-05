"""Chạy các thí nghiệm so sánh cho báo cáo và xuất bảng kết quả.

Mỗi cấu hình được train trên phần train của ảnh Sudoku (+ nguồn bổ sung) rồi đánh giá trên:
validation chữ in, MNIST viết tay, 40 ảnh test của dataset và ảnh tự chụp trong samples/.

    python scripts/experiments.py            # ~10-15 phút
    python scripts/experiments.py --quick    # bỏ các cấu hình SVM dữ liệu lớn
Kết quả: reports/experiments.md và reports/experiments.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sudoku_vision.classifier import DigitClassifier, build_model  # noqa: E402
from sudoku_vision.config import PipelineParams  # noqa: E402
from sudoku_vision.dataset import load_folder, load_outlines, load_split  # noqa: E402
from sudoku_vision.evaluation import evaluate_samples  # noqa: E402
from sudoku_vision.pipeline import SudokuPipeline  # noqa: E402
from sudoku_vision.training import load_training_data  # noqa: E402

ALL = ("sudoku", "chars74k", "mnist")
CONFIGS = [
    # (tên, model, đặc trưng, nguồn dữ liệu, số bản tăng cường)
    ("k-NN + HOG, chỉ ô Sudoku", "knn", "hog", ("sudoku",), 2),
    ("SVM + HOG, chỉ ô Sudoku", "svm", "hog", ("sudoku",), 2),
    ("SVM + pixel thô, đủ dữ liệu", "svm", "raw", ALL, 2),
    ("k-NN + HOG, đủ dữ liệu", "knn", "hog", ALL, 2),
    ("SVM + HOG, đủ dữ liệu, không tăng cường", "svm", "hog", ALL, 0),
    ("SVM + HOG, đủ dữ liệu (model chính)", "svm", "hog", ALL, 2),
]
QUICK = {0, 1, 3}


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    data = load_training_data(args.data, ROOT / "data" / "chars74k" / "English" / "Fnt", ROOT / "data" / "mnist")
    test = load_split(args.data, "test")
    own = load_folder(ROOT / "samples")
    outlines = load_outlines(Path(args.data) / "outlines_sorted.csv")

    results = []
    configs = [c for i, c in enumerate(CONFIGS) if not args.quick or i in QUICK]
    for name, model, feature, sources, copies in configs:
        print(f"== {name}")
        x, y = data.train_set(sources, copies)
        t = time.perf_counter()
        clf = DigitClassifier(build_model(model), feature, {"name": model}).fit(x, y)
        row = {"config": name, "model": model, "feature": feature, "sources": list(sources), "augment": copies,
               "train_samples": len(x), "train_seconds": round(time.perf_counter() - t, 1)}
        for vname, (vd, vl) in data.val_sets.items():
            row[vname] = accuracy_score(vl, clf.predict(vd)[0])
        for correct in (False, True):
            pipe = SudokuPipeline(clf, PipelineParams(auto_correct=correct))
            m, _, _ = evaluate_samples(pipe, test, outlines)
            suffix = "_correct" if correct else ""
            row["test_cell_acc"] = m["cell_accuracy"]
            row["test_grid_acc"] = m["grid_accuracy"]
            row["test_solution" + suffix] = m["solution_correct_rate"]
        if own:
            m, _, _ = evaluate_samples(SudokuPipeline(clf), own, ignore_orientation=False)
            row["samples_cell_acc"] = m["cell_accuracy"]
            row["samples_solution"] = m["solution_correct_rate"]
        results.append(row)
        print("   " + ", ".join(f"{k} = {v:.4f}" for k, v in row.items() if isinstance(v, float)))

    # Thí nghiệm riêng cho bước tìm lưới: adaptive threshold so với Otsu (không cần model)
    grid_rows = []
    for method in ("adaptive", "otsu"):
        m, _, _ = evaluate_samples(SudokuPipeline(None, PipelineParams(thresh_method=method)), test, outlines)
        grid_rows.append({"method": method, "grid_found_rate": m["grid_found_rate"],
                          "grid_iou_mean": m["grid_iou_mean"], "grid_iou_over_0.9": m["grid_iou_over_0.9"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiments.json").write_text(json.dumps({"classifier": results, "grid": grid_rows}, indent=2,
                                                     ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Kết quả thực nghiệm", "",
        f"Tập test: {len(test)} ảnh của wichtounet/sudoku_dataset V2. Ảnh tự chụp: {len(own)} ảnh trong samples/.", "",
        "## Bộ phân loại", "",
        "| Cấu hình | Mẫu train | Val chữ in | MNIST viết tay | Test: đúng theo ô | Test: ảnh đúng 81 ô "
        "| Test: đáp án đúng | + tự sửa | Ảnh tự chụp: đáp án đúng |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['config']} | {r['train_samples']} | {pct(r['sudoku_print'])} | "
                     f"{pct(r.get('mnist_handwritten'))} | {pct(r['test_cell_acc'])} | {pct(r['test_grid_acc'])} | "
                     f"{pct(r['test_solution'])} | {pct(r['test_solution_correct'])} | "
                     f"{pct(r.get('samples_solution'))} |")
    lines += ["", "## Tìm lưới", "", "| Nhị phân hoá | Tìm thấy lưới | IoU trung bình | IoU > 0,9 |", "|---|---|---|---|"]
    for g in grid_rows:
        lines.append(f"| {g['method']} | {pct(g['grid_found_rate'])} | {g['grid_iou_mean']:.3f} | "
                     f"{pct(g['grid_iou_over_0.9'])} |")
    (out / "experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nĐã lưu {out / 'experiments.md'}")


if __name__ == "__main__":
    main()
