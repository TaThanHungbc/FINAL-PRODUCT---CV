"""Tải dữ liệu về data/:
- data/raw:      wichtounet/sudoku_dataset bản V2 (ảnh Sudoku + nhãn)
- data/chars74k: Chars74K EnglishFnt (chữ số in từ font máy tính)
- data/mnist:    MNIST (chữ số viết tay)

    python scripts/download_data.py              # tải tất cả
    python scripts/download_data.py --sudoku-only
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/wichtounet/sudoku_dataset/master/"
FILES = ["datasets/v2_train.tar.bz2", "datasets/v2_test.tar.bz2", "outlines_sorted.csv", "README.rst"]
CHARS74K = "https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"
MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  đã có {dest.name}")
        return dest
    print(f"  tải {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def extract(archive: Path, dest: Path) -> None:
    print(f"  giải nén {archive.name}")
    with tarfile.open(archive) as t:
        t.extractall(dest, filter="data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--sudoku-only", action="store_true", help="chỉ tải dataset Sudoku")
    args = ap.parse_args()

    raw = Path(args.out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print("Sudoku dataset (CC-BY-4.0, github.com/wichtounet/sudoku_dataset):")
    for f in FILES:
        path = fetch(BASE + f, raw / Path(f).name)
        split_dir = raw / path.name.replace(".tar.bz2", "")
        if path.name.endswith(".tar.bz2") and not split_dir.is_dir():
            extract(path, raw)

    if not args.sudoku_only:
        print("Chars74K EnglishFnt (~50 MB):")
        dest = Path(args.out) / "chars74k"
        dest.mkdir(parents=True, exist_ok=True)
        path = fetch(CHARS74K, dest / "EnglishFnt.tgz")
        if not (dest / "English").is_dir():
            extract(path, dest)
        print("MNIST (~11 MB):")
        dest = Path(args.out) / "mnist"
        dest.mkdir(parents=True, exist_ok=True)
        for f in MNIST_FILES:
            fetch(MNIST_BASE + f, dest / f)
    print("Xong.")


if __name__ == "__main__":
    main()
