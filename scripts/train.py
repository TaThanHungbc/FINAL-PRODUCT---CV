"""Train bộ phân loại 10 lớp (0 = nhiễu, 1-9) cho cả chữ in lẫn chữ viết tay.

Dữ liệu train:
- Ô cắt từ ảnh train của wichtounet/sudoku_dataset (chữ in thật + mẫu nhiễu lớp 0)
- Chars74K EnglishFnt (chữ in nhiều font), nếu có data/chars74k
- MNIST (chữ viết tay), nếu có data/mnist
- Ô cắt từ ảnh Sudoku viết tay tổng hợp (data/synthetic_hw_train, tạo bằng make_synthetic.py --split train)
- Ô chữ số người dùng lưu từ app (data/user_digits), nếu có

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sudoku_vision.classifier import MODEL_NAMES, DigitClassifier, build_model  # noqa: E402
from sudoku_vision.config import PipelineParams  # noqa: E402
from sudoku_vision.features import FEATURES  # noqa: E402
from sudoku_vision.training import load_training_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--chars74k", default=str(ROOT / "data" / "chars74k" / "English" / "Fnt"))
    ap.add_argument("--mnist", default=str(ROOT / "data" / "mnist"))
    ap.add_argument("--no-chars74k", action="store_true")
    ap.add_argument("--no-mnist", action="store_true")
    ap.add_argument("--mnist-per-class", type=int, default=1500)
    ap.add_argument("--synthetic", default=str(ROOT / "data" / "synthetic_hw_train"))
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--user-digits", default=str(ROOT / "data" / "user_digits"),
                    help="ô chữ số lưu từ app (nút Lưu làm dữ liệu train)")
    ap.add_argument("--no-user-digits", action="store_true")
    ap.add_argument("--model", default="all", choices=[*MODEL_NAMES, "all"])
    ap.add_argument("--feature", default="hog", choices=list(FEATURES))
    ap.add_argument("--augment", type=int, default=2, help="số bản tăng cường cho mỗi ô Sudoku (0 = tắt)")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--limit", type=int, help="chỉ dùng N ảnh đầu (để chạy thử)")
    ap.add_argument("--out", default=str(ROOT / "models" / "digit_model.joblib"))
    ap.add_argument("--report", default=str(ROOT / "reports" / "train_report.json"))
    args = ap.parse_args()

    print("1) Chuẩn bị dữ liệu")
    data = load_training_data(args.data, None if args.no_chars74k else args.chars74k,
                              None if args.no_mnist else args.mnist, args.mnist_per_class,
                              args.val_size, args.limit, None if args.no_user_digits else args.user_digits,
                              None if args.no_synthetic else args.synthetic, ROOT / "data" / "synthetic_hw")
    if not args.no_synthetic and "synthetic" not in data.extras:
        print("   (Chưa có ảnh viết tay tổng hợp: chạy scripts/make_synthetic.py --split train để nhận dạng"
              " chữ viết tay tốt hơn)")
    sources = ("sudoku", *data.extras)
    x_tr, y_tr = data.train_set(sources, args.augment)
    print(f"   Tập train: {len(x_tr)} mẫu, số mẫu theo lớp: {np.bincount(y_tr, minlength=10).tolist()}")

    print(f"2) Train và so sánh model (đặc trưng {args.feature})...")
    names = MODEL_NAMES if args.model == "all" else (args.model,)
    results, best_name, best_score = {}, None, -1.0
    for name in names:
        t = time.perf_counter()
        clf = DigitClassifier(build_model(name), args.feature, {"name": name}).fit(x_tr, y_tr)
        res = {"train_seconds": round(time.perf_counter() - t, 1)}
        for vname, (vd, vl) in data.val_sets.items():
            pred, _ = clf.predict(vd, PipelineParams().noise_min_prob)
            res[vname] = accuracy_score(vl, pred)
            if vname == "sudoku_print":
                res["sudoku_print_confusion"] = confusion_matrix(vl, pred, labels=range(10)).tolist()
        score = float(np.mean([res[v] for v in data.val_sets]))
        res["mean_accuracy"] = score
        results[name] = res
        print("   " + name + ": " + ", ".join(f"{v} = {res[v]:.4f}" for v in data.val_sets)
              + f" ({res['train_seconds']}s)")
        # Hoà thì chọn SVM: file model nhỏ hơn k-NN nhiều
        if score > best_score or (score == best_score and name == "svm"):
            best_name, best_score = name, score

    print(f"3) Train lại {best_name} trên toàn bộ dữ liệu và lưu model...")
    x_all, y_all = data.train_set(sources, args.augment, include_val=True)
    meta = {"name": best_name, "val": {k: v for k, v in results[best_name].items() if not k.endswith("confusion")},
            "train_samples": len(x_all), "augment": args.augment, "sources": list(sources)}
    DigitClassifier(build_model(best_name), args.feature, meta).fit(x_all, y_all).save(args.out)
    print(f"   Đã lưu {args.out}")

    report = {"feature": args.feature, "best": best_name, "models": results, "sources": list(sources),
              "extraction": data.stats}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   Báo cáo: {args.report}")
    print("Xong. Chạy scripts/evaluate.py để đánh giá trên tập test.")


if __name__ == "__main__":
    main()
