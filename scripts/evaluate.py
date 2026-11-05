"""Đánh giá pipeline trên tập test (hoặc train) của dataset, hoặc một thư mục ảnh tự chụp.

Ví dụ:
    python scripts/evaluate.py                    # dùng models/digit_model.joblib, tập test
    python scripts/evaluate.py --no-model         # chỉ đánh giá tìm lưới + phát hiện ô trống
    python scripts/evaluate.py --no-correct       # tắt bước tự sửa bằng luật Sudoku
    python scripts/evaluate.py --folder samples   # ảnh tự chụp kèm nhãn .dat
    python scripts/evaluate.py --save-vis 10      # lưu ảnh minh họa 10 ảnh đầu vào reports/vis
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sudoku_vision.classifier import DigitClassifier  # noqa: E402
from sudoku_vision.config import PipelineParams  # noqa: E402
from sudoku_vision.dataset import load_folder, load_outlines, load_split  # noqa: E402
from sudoku_vision.evaluation import evaluate_samples  # noqa: E402
from sudoku_vision.pipeline import SudokuPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--model", default=str(ROOT / "models" / "digit_model.joblib"))
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--folder", help="đánh giá thư mục ảnh + .dat bất kỳ (ví dụ samples) thay cho --split")
    ap.add_argument("--no-correct", action="store_true", help="tắt bước tự sửa bằng luật Sudoku")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--save-vis", type=int, default=0, help="số ảnh minh họa cần lưu")
    args = ap.parse_args()

    clf = None
    if not args.no_model:
        if not Path(args.model).exists():
            sys.exit(f"Không thấy model {args.model}. Chạy scripts/train.py trước, hoặc dùng --no-model.")
        clf = DigitClassifier.load(args.model)
        print(f"Model: {clf.name} ({clf.feature})")
    pipe = SudokuPipeline(clf, PipelineParams(auto_correct=not args.no_correct))
    samples = load_folder(args.folder, args.limit) if args.folder else load_split(args.data, args.split, args.limit)
    if not samples:
        sys.exit("Không có ảnh nào kèm nhãn .dat để đánh giá.")
    split_name = Path(args.folder).name if args.folder else args.split
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Ảnh dataset: bỏ qua thẻ xoay EXIF để khớp nhãn; ảnh tự chụp (--folder): giữ EXIF
    metrics, rows, cm = evaluate_samples(pipe, samples, load_outlines(Path(args.data) / "outlines_sorted.csv"),
                                         ignore_orientation=not args.folder,
                                         vis_dir=out / "vis", save_vis=args.save_vis)
    metrics = {"split": split_name, **metrics}

    tag = f"{split_name}_{'nomodel' if clf is None else 'model'}{'_nocorrect' if args.no_correct else ''}"
    (out / f"eval_{tag}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / f"eval_{tag}_images.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if clf is not None:
        np.savetxt(out / f"confusion_{tag}.csv", cm, fmt="%d", delimiter=",",
                   header="hàng = nhãn thật 0..9, cột = dự đoán 0..9 (0 = ô trống)")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Đã lưu kết quả vào {out}")


if __name__ == "__main__":
    main()
