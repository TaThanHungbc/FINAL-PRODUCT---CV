"""Giải Sudoku cho một hoặc nhiều ảnh từ dòng lệnh.

    python scripts/solve.py samples/handwritten_01.png
    python scripts/solve.py anh1.jpg anh2.jpg --out ket_qua --stages
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sudoku_vision import paths  # noqa: E402
from sudoku_vision.classifier import DigitClassifier  # noqa: E402
from sudoku_vision.config import PipelineParams  # noqa: E402
from sudoku_vision.imageio import imread, imwrite  # noqa: E402
from sudoku_vision.pipeline import SudokuPipeline  # noqa: E402


def format_grid(grid) -> str:
    lines = []
    for r in range(9):
        if r and r % 3 == 0:
            lines.append("------+-------+------")
        row = [str(v) if v > 0 else "." for v in grid[r]]
        lines.append(" | ".join(" ".join(row[c:c + 3]) for c in (0, 3, 6)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", help="mặc định: models/digit_model.joblib")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--stages", action="store_true", help="lưu ảnh của mọi bước")
    ap.add_argument("--no-correct", action="store_true", help="tắt bước tự sửa bằng luật Sudoku")
    args = ap.parse_args()

    model = Path(args.model) if args.model else paths.find_model()
    if model is None:
        sys.exit("Không thấy model. Chạy scripts/train.py trước hoặc truyền --model.")
    pipe = SudokuPipeline(DigitClassifier.load(model), PipelineParams(auto_correct=not args.no_correct))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in map(Path, args.images):
        image = imread(path)
        if image is None:
            print(f"{path}: không đọc được ảnh")
            failed += 1
            continue
        res = pipe.run(image)
        print(f"== {path.name}: {res.message}")
        if res.grid is not None:
            print(format_grid(res.grid))
        if res.solve and res.solve.ok:
            print("Lời giải:")
            print(format_grid(res.solve.solution))
        else:
            failed += 1
        stages = res.stages if args.stages else {k: v for k, v in res.stages.items() if k == "solution"}
        for name, img in stages.items():
            imwrite(out / f"{path.stem}_{name}.png", img)
    print(f"Ảnh kết quả lưu trong {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
