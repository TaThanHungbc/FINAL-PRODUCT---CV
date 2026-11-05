"""Đánh giá pipeline trên một tập ảnh có nhãn (dùng chung cho scripts/evaluate.py và scripts/experiments.py)."""
from pathlib import Path

import numpy as np

from .grid import polygon_iou
from .imageio import imread, imwrite
from .pipeline import SudokuPipeline
from .solver import solve


def evaluate_samples(pipe: SudokuPipeline, samples, outlines=None, ignore_orientation: bool = True,
                     vis_dir: Path | None = None, save_vis: int = 0):
    """Trả về (độ đo, danh sách theo ảnh, ma trận nhầm lẫn 10x10 với lớp 0 = ô trống)."""
    outlines = outlines or {}
    has_model = pipe.classifier is not None
    cm = np.zeros((10, 10), np.int64)
    rows, ious = [], []
    n_found = n_grid_ok = n_solved = n_correct = n_final_ok = n_fixed = n_multi = 0
    for k, s in enumerate(samples):
        image = imread(s.image_path, ignore_orientation=ignore_orientation)
        res = pipe.run(image)
        row = {"image": s.name, "grid_found": res.grid_found, "iou": "", "cell_acc": "",
               "grid_correct": "", "solve": res.solve.status if res.solve else "",
               "unique": "" if not (res.solve and res.solve.ok) else res.solve.unique,
               "corrections": len(res.corrections), "solution_correct": ""}
        pred = np.zeros((9, 9), np.int32)  # không tìm thấy lưới: coi như mọi ô trống
        if res.grid_found:
            n_found += 1
            if s.name in outlines:
                iou = polygon_iou(res.corners_original, outlines[s.name], image.shape[:2])
                ious.append(iou)
                row["iou"] = round(iou, 4)
            if not has_model:
                # Không có model: coi ô phát hiện có chữ số là đúng nhãn, chỉ đo phát hiện ô trống
                # (ô trống bị nhận là có chữ số được ghi vào cột 1 để vẫn tính là lỗi)
                pred = np.where(res.grid != 0, np.where(s.grid > 0, s.grid, 1), 0)
            else:
                pred = res.recognized_grid
                n_solved += bool(res.solve and res.solve.ok)
                n_multi += bool(res.solve and res.solve.ok and res.solve.unique is False)
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
        if vis_dir is not None and k < save_vis:
            vis_dir.mkdir(parents=True, exist_ok=True)
            for name, img in res.stages.items():
                imwrite(vis_dir / f"{Path(s.name).stem}_{name}.jpg", img)

    n = max(len(samples), 1)
    empty_true, filled_true = cm[0].sum(), cm[1:].sum()
    metrics = {
        "images": len(samples),
        "model": None if not has_model else {"name": pipe.classifier.name, "feature": pipe.classifier.feature},
        "grid_found_rate": n_found / n,
        "grid_iou_mean": float(np.mean(ious)) if ious else None,
        "grid_iou_over_0.9": float(np.mean(np.array(ious) > 0.9)) if ious else None,
        "empty_detection": {
            "empty_recall": float(cm[0, 0] / empty_true) if empty_true else None,
            "digit_recall": float(cm[1:, 1:].sum() / filled_true) if filled_true else None,
        },
    }
    if has_model:
        digit_cells = cm[1:, 1:].sum()
        metrics.update({
            "digit_accuracy": float(np.trace(cm[1:, 1:]) / digit_cells) if digit_cells else None,
            "cell_accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else None,
            "grid_accuracy": n_grid_ok / n,
            "grid_accuracy_after_correction": n_final_ok / n,
            "images_auto_corrected": n_fixed,
            "solve_rate": n_solved / n,
            "solved_not_unique": n_multi,
            "solution_correct_rate": n_correct / n,
            "auto_correct": pipe.p.auto_correct,
        })
    return metrics, rows, cm
