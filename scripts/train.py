"""Train bộ phân loại 10 lớp (0 = nhiễu, 1-9) cho cả chữ in lẫn chữ viết tay.

Dữ liệu train:
- Ô cắt từ ảnh train của wichtounet/sudoku_dataset (chữ in thật + mẫu nhiễu lớp 0)
- Chars74K EnglishFnt (chữ in nhiều font), nếu có data/chars74k
- MNIST (chữ viết tay), nếu có data/mnist

Validation tách riêng theo miền: ô từ 20% số ảnh Sudoku giữ lại (chia theo ảnh, không theo ô)
và MNIST t10k (viết tay). Model tốt nhất được train lại trên toàn bộ dữ liệu rồi lưu.

Ví dụ:
    python scripts/train.py
    python scripts/train.py --model svm --mnist-per-class 2000
    python scripts/train.py --no-mnist --no-chars74k     # chỉ dùng dữ liệu Sudoku
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sudoku_vision.classifier import MODEL_NAMES, DigitClassifier, build_model  # noqa: E402
from sudoku_vision.dataset import load_split  # noqa: E402
from sudoku_vision.digits_data import augment, collect_chars74k, collect_from_dataset, collect_mnist  # noqa: E402
from sudoku_vision.features import FEATURES  # noqa: E402
from sudoku_vision.pipeline import SudokuPipeline  # noqa: E402


def build_train_set(sudoku_digits, sudoku_labels, extras, copies):
    d, l = augment(sudoku_digits, sudoku_labels, copies=copies) if copies else (list(sudoku_digits), sudoku_labels)
    for xd, xl in extras:
        d, l = d + list(xd), np.concatenate([l, xl])
    return d, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--chars74k", default=str(ROOT / "data" / "chars74k" / "English" / "Fnt"))
    ap.add_argument("--mnist", default=str(ROOT / "data" / "mnist"))
    ap.add_argument("--no-chars74k", action="store_true")
    ap.add_argument("--no-mnist", action="store_true")
    ap.add_argument("--mnist-per-class", type=int, default=1000)
    ap.add_argument("--model", default="all", choices=[*MODEL_NAMES, "all"])
    ap.add_argument("--feature", default="hog", choices=list(FEATURES))
    ap.add_argument("--augment", type=int, default=2, help="số bản tăng cường cho mỗi ô Sudoku (0 = tắt)")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--limit", type=int, help="chỉ dùng N ảnh đầu (để chạy thử)")
    ap.add_argument("--out", default=str(ROOT / "models" / "digit_model.joblib"))
    ap.add_argument("--report", default=str(ROOT / "reports" / "train_report.json"))
    args = ap.parse_args()

    pipe = SudokuPipeline()
    samples = load_split(args.data, "train", args.limit)
    tr_s, va_s = train_test_split(samples, test_size=args.val_size, random_state=42)
    print(f"1) Cắt ô từ ảnh Sudoku: {len(tr_s)} ảnh train, {len(va_s)} ảnh validation")
    tr_d, tr_l, stats = collect_from_dataset(tr_s, pipe)
    va_d, va_l, _ = collect_from_dataset(va_s, pipe)

    extras, val_sets = [], {"sudoku_print": (va_d, va_l)}
    if not args.no_chars74k and Path(args.chars74k).is_dir():
        extras.append(collect_chars74k(args.chars74k))
        print(f"   + Chars74K: {len(extras[-1][0])} chữ số in")
    if not args.no_mnist and Path(args.mnist).is_dir():
        extras.append(collect_mnist(args.mnist, "train", args.mnist_per_class))
        print(f"   + MNIST: {len(extras[-1][0])} chữ số viết tay")
        val_sets["mnist_handwritten"] = collect_mnist(args.mnist, "test", 300, seed=1)

    x_tr, y_tr = build_train_set(tr_d, tr_l, extras, args.augment)
    print(f"   Tập train: {len(x_tr)} mẫu, lớp: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    print(f"2) Train và so sánh model (đặc trưng {args.feature})...")
    names = MODEL_NAMES if args.model == "all" else (args.model,)
    results, best_name, best_score = {}, None, -1.0
    for name in names:
        t = time.perf_counter()
        clf = DigitClassifier(build_model(name), args.feature, {"name": name}).fit(x_tr, y_tr)
        res = {"train_seconds": round(time.perf_counter() - t, 1)}
        for vname, (vd, vl) in val_sets.items():
            pred, _ = clf.predict(vd)
            res[vname] = accuracy_score(vl, pred)
            if vname == "sudoku_print":
                res["sudoku_print_confusion"] = confusion_matrix(vl, pred, labels=range(10)).tolist()
        score = float(np.mean([res[v] for v in val_sets]))
        res["mean_accuracy"] = score
        results[name] = res
        print("   " + name + ": " + ", ".join(f"{v} = {res[v]:.4f}" for v in val_sets)
              + f" ({res['train_seconds']}s)")
        # Hoà thì chọn SVM: file model nhỏ hơn k-NN nhiều
        if score > best_score or (score == best_score and name == "svm"):
            best_name, best_score = name, score

    print(f"3) Train lại {best_name} trên toàn bộ dữ liệu và lưu model...")
    x_all, y_all = build_train_set(tr_d + va_d, np.concatenate([tr_l, va_l]), extras, args.augment)
    meta = {"name": best_name, "val": {k: v for k, v in results[best_name].items() if not k.endswith("confusion")},
            "train_samples": len(x_all), "augment": args.augment,
            "sources": ["sudoku"] + (["chars74k"] if not args.no_chars74k and Path(args.chars74k).is_dir() else [])
            + (["mnist"] if "mnist_handwritten" in val_sets else [])}
    DigitClassifier(build_model(best_name), args.feature, meta).fit(x_all, y_all).save(args.out)
    print(f"   Đã lưu {args.out}")

    report = {"feature": args.feature, "best": best_name, "models": results, "sources": meta["sources"],
              "extraction": {k: v for k, v in stats.items() if k != "grid_not_found"}}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   Báo cáo: {args.report}")
    print("Xong. Chạy scripts/evaluate.py để đánh giá trên tập test.")


if __name__ == "__main__":
    main()
