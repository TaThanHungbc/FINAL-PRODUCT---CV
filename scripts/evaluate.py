"""Đánh giá pipeline trên tập test (hoặc train) của dataset.

Ví dụ:
    python scripts/evaluate.py                    # dùng models/digit_model.joblib, tập test
    python scripts/evaluate.py --no-model         # chỉ đánh giá tìm lưới + phát hiện ô trống
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
from sudoku_vision.grid import polygon_iou  # noqa: E402
from sudoku_vision.imageio import imread, imwrite  # noqa: E402
from sudoku_vision.pipeline import SudokuPipeline  # noqa: E402
from sudoku_vision.solver import solve  # noqa: E402


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
    split_name = Path(args.folder).name if args.folder else args.split
    outlines = load_outlines(Path(args.data) / "outlines_sorted.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "vis"

    # Ma trận nhầm lẫn 10x10: hàng = nhãn thật, cột = dự đoán, lớp 0 = ô trống
    cm = np.zeros((10, 10), np.int64)
    rows, ious = [], []
    n_found = n_grid_ok = n_solved = n_correct = n_final_ok = n_fixed = 0
    for k, s in enumerate(samples):
        # Ảnh dataset: bỏ qua thẻ xoay EXIF để khớp nhãn; ảnh tự chụp (--folder): giữ EXIF
        image = imread(s.image_path, ignore_orientation=not args.folder)
        res = pipe.run(image)
        row = {"image": s.name, "grid_found": res.grid_found, "iou": "", "cell_acc": "",
               "grid_correct": "", "solve": res.solve.status if res.solve else "",
               "corrections": len(res.corrections), "solution_correct": ""}
        pred = np.zeros((9, 9), np.int32)  # không tìm thấy lưới: coi như mọi ô trống
        if res.grid_found:
            n_found += 1
            if s.name in outlines:
                iou = polygon_iou(res.corners_original, outlines[s.name], image.shape[:2])
                ious.append(iou)
                row["iou"] = round(iou, 4)
            if clf is None:
                # Không có model: coi ô phát hiện có chữ số là đúng nhãn, chỉ đo phát hiện ô trống
                # (ô trống bị nhận là có chữ số được ghi vào cột 1 để vẫn tính là lỗi)
                pred = np.where(res.grid != 0, np.where(s.grid > 0, s.grid, 1), 0)
            else:
                pred = res.recognized_grid
                n_solved += bool(res.solve and res.solve.ok)
                n_final_ok += bool((res.grid == s.grid).all())
                n_fixed += bool(res.corrections)
                truth = solve(s.grid)
                ok = bool(res.solve and res.solve.ok and truth.ok and (res.solve.solution == truth.solution).all())
                row["solution_correct"] = ok
                n_correct += ok
            acc = float((pred == s.grid).mean())
            row["cell_acc"] = round(acc, 4)
            row["grid_correct"] = acc == 1.0
            n_grid_ok += acc == 1.0
        for t, q in zip(s.grid.ravel(), pred.ravel()):
            cm[t, q] += 1
        rows.append(row)
        if k < args.save_vis:
            vis_dir.mkdir(parents=True, exist_ok=True)
            for name, img in res.stages.items():
                imwrite(vis_dir / f"{Path(s.name).stem}_{name}.jpg", img)

    n = len(samples)
    total = cm.sum()
    empty_true = cm[0].sum()
    filled_true = cm[1:].sum()
    metrics = {
        "split": split_name,
        "images": n,
        "model": None if clf is None else {"name": clf.name, "feature": clf.feature},
        "grid_found_rate": n_found / n,
        "grid_iou_mean": float(np.mean(ious)) if ious else None,
        "grid_iou_over_0.9": float(np.mean(np.array(ious) > 0.9)) if ious else None,
        "empty_detection": {
            "empty_recall": float(cm[0, 0] / empty_true) if empty_true else None,
            "digit_recall": float(cm[1:, 1:].sum() / filled_true) if filled_true else None,
        },
    }
    if clf is not None:
        digit_cells = cm[1:, 1:].sum()
        metrics.update({
            "digit_accuracy": float(np.trace(cm[1:, 1:]) / digit_cells) if digit_cells else None,
            "cell_accuracy": float(np.trace(cm) / total) if total else None,
            "grid_accuracy": n_grid_ok / n,
            "grid_accuracy_after_correction": n_final_ok / n,
            "images_auto_corrected": n_fixed,
            "solve_rate": n_solved / n,
            "solution_correct_rate": n_correct / n,
            "auto_correct": not args.no_correct,
        })

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
