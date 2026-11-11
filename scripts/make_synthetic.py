"""Tạo bộ ảnh Sudoku viết tay tổng hợp.

Lấy đề bài từ nhãn của dataset, viết từng số bằng một chữ số MNIST,
vẽ lên lưới giấy với độ đậm kiểu bút chì/bút bi, vị trí và cỡ chữ lệch ngẫu nhiên, rồi phối cảnh,
làm mờ, thêm nhiễu và nén JPEG như ảnh chụp điện thoại.

    python scripts/make_synthetic.py                 # bộ đánh giá: 60 ảnh, chữ số MNIST t10k, đề tập test
    python scripts/make_synthetic.py --split train   # bộ train: 200 ảnh, chữ số MNIST train, đề tập train
    python scripts/evaluate.py --folder data/synthetic_hw

Hai bộ dùng chữ số MNIST và đề bài khác nhau nên đánh giá trên bộ test vẫn độc lập với dữ liệu train.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sudoku_vision.dataset import load_split  # noqa: E402
from sudoku_vision.digits_data import _read_idx  # noqa: E402
from sudoku_vision.imageio import imwrite  # noqa: E402


def render(grid, mnist_imgs, mnist_by_label, rng, size=900, cell=80):
    paper = np.full((size, size), rng.uniform(215, 245), np.float32)
    # Ánh sáng không đều
    yy, xx = np.mgrid[0:size, 0:size] / size
    paper += rng.uniform(-25, 25) * xx + rng.uniform(-25, 25) * yy
    off = (size - 9 * cell) // 2
    line = rng.uniform(20, 70)
    for i in range(10):
        t = 4 if i % 3 == 0 else 2
        p = off + i * cell
        cv2.line(paper, (off, p), (off + 9 * cell, p), line, t)
        cv2.line(paper, (p, off), (p, off + 9 * cell), line, t)

    ink = rng.uniform(30, 120)  # bút bi đậm ... bút chì nhạt
    strength = rng.uniform(0.45, 1.0)
    for r in range(9):
        for c in range(9):
            v = int(grid[r, c])
            if not v:
                continue
            d = mnist_imgs[rng.choice(mnist_by_label[v])].astype(np.float32) / 255
            h = int(cell * rng.uniform(0.6, 1.0))
            d = cv2.resize(d, (h, h), interpolation=cv2.INTER_LINEAR)
            m = cv2.getRotationMatrix2D((h / 2, h / 2), rng.uniform(-12, 12), 1.0)
            d = cv2.warpAffine(d, m, (h, h))
            cx = off + c * cell + cell / 2 + rng.uniform(-0.1, 0.1) * cell
            cy = off + r * cell + cell / 2 + rng.uniform(-0.1, 0.1) * cell
            x0, y0 = int(cx - h / 2), int(cy - h / 2)
            region = paper[y0:y0 + h, x0:x0 + h]
            alpha = np.clip(d * strength, 0, 1)
            paper[y0:y0 + h, x0:x0 + h] = region * (1 - alpha) + ink * alpha

    # Phối cảnh như khi chụp nghiêng
    src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    dst = src + rng.uniform(-0.08, 0.08, size=(4, 2)).astype(np.float32) * size
    m = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(paper, m, (size, size), borderValue=float(rng.uniform(120, 200)))
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.8))
    img += rng.normal(0, rng.uniform(2, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 90))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--mnist", default=str(ROOT / "data" / "mnist"))
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", help="mặc định data/synthetic_hw (test) hoặc data/synthetic_hw_train (train)")
    ap.add_argument("-n", type=int, help="mặc định 60 (test) hoặc 200 (train)")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    train = args.split == "train"
    n = args.n or (200 if train else 60)
    rng = np.random.default_rng(args.seed if args.seed is not None else (11 if train else 7))
    prefix = "train" if train else "t10k"
    images = _read_idx(Path(args.mnist) / f"{prefix}-images-idx3-ubyte.gz")
    labels = _read_idx(Path(args.mnist) / f"{prefix}-labels-idx1-ubyte.gz")
    by_label = {v: np.flatnonzero(labels == v) for v in range(1, 10)}
    puzzles = [s.grid for s in load_split(args.data, args.split)]
    out = Path(args.out or ROOT / "data" / ("synthetic_hw_train" if train else "synthetic_hw"))
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        grid = puzzles[i % len(puzzles)]
        img = render(grid, images, by_label, rng)
        name = f"synthetic_{i + 1:03d}"
        imwrite(out / f"{name}.jpg", img)
        rows = "\n".join(" ".join(str(int(v)) for v in row) for row in grid)
        (out / f"{name}.dat").write_text(f"synthetic handwritten (MNIST {prefix})\n900x900 JPG\n{rows}\n",
                                         encoding="utf-8")
    print(f"Đã tạo {n} ảnh trong {out}")


if __name__ == "__main__":
    main()
