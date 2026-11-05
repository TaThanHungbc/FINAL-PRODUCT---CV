# Sudoku Vision (CS231)

Giải Sudoku từ ảnh chụp bằng xử lý ảnh cổ điển: OpenCV để tìm lưới và cắt ô, HOG + SVM (scikit-learn) để nhận dạng chữ số in lẫn viết tay, backtracking để giải. Khi nhận dạng sai, chính luật Sudoku được dùng để tự sửa.

**Input:** ảnh chụp một ô Sudoku trên báo/giấy (chữ in hoặc viết tay). **Output:** lưới 9×9 nhận dạng được, lời giải, và ảnh gốc có đáp án vẽ đè lên các ô trống.

## Kết quả (40 ảnh test của dataset)

| Độ đo | v1: k-NN, chỉ chữ in | v2: SVM, in + viết tay, có tự sửa |
|---|---|---|
| Nhận dạng đúng trên ô có số | 98,95% | 99,31% |
| Đúng trên toàn bộ 81 ô | 98,95% | 99,60% |
| Ảnh đúng cả 81 ô (trước tự sửa) | 60,0% | 75,0% |
| Ảnh đúng cả 81 ô (sau tự sửa) | – | 85,0% |
| **Ảnh có đáp án đúng** | – | **92,5%** (77,5% nếu tắt tự sửa) |

Validation của bộ phân loại: 99,25% trên ô chữ in từ ảnh Sudoku, 98,11% trên chữ viết tay MNIST. Ảnh viết tay bút chì trong `samples/` được giải đúng 100%. Lưu ý: ảnh này cũng đã được dùng khi chọn ngưỡng tách ô, nên chưa phải tập kiểm tra độc lập.

Chi tiết trong `reports/` (`reports/v1_knn_print/` là kết quả của bản đầu tiên).

## Pipeline

| Bước | Module | Kỹ thuật |
|---|---|---|
| 1. Tiền xử lý | `preprocess.py` | Thu nhỏ, ảnh xám, Gaussian blur, adaptive threshold (hoặc Otsu) |
| 2. Tìm lưới | `grid.py` | Giãn nét, contour lớn nhất, `approxPolyDP` lấy 4 góc |
| 3. Nắn phẳng | `grid.py` | `getPerspectiveTransform` + `warpPerspective` về 450×450 |
| 4. Cắt ô | `cells.py` | Chia 81 ô, xoá viền, connected components, gộp các mảnh nét bị đứt, chuẩn hoá 28×28 |
| 5. Nhận dạng | `features.py`, `classifier.py` | HOG (324 chiều), SVM 10 lớp (0 = nhiễu, 1–9) |
| 6. Giải | `solver.py` | Backtracking + MRV, bitmask |
| 6b. Tự sửa | `correction.py` | Đề trùng số / vô nghiệm: thử thay các ô kém tin cậy bằng lựa chọn thứ 2, 3 của SVM, ưu tiên tổ hợp có xác suất cao nhất |
| 7. Hiển thị | `overlay.py` | Vẽ đáp án trên ảnh phẳng, warp ngược về ảnh gốc |

Tham số của mọi bước nằm trong `sudoku_vision/config.py`.

**Dữ liệu train bộ phân loại:** ô cắt từ 160 ảnh train của dataset Sudoku (chữ in thật, kèm mẫu nhiễu làm lớp 0), Chars74K EnglishFnt (chữ in nhiều font) và MNIST (viết tay).

## Cài đặt

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\download_data.py
```

Nên dùng `.venv` cho cả train lẫn đóng gói exe: model `.joblib` phải được nạp bằng đúng phiên bản scikit-learn đã train ra nó.

## Train và đánh giá

```bat
.venv\Scripts\python scripts\train.py                     :: so sánh k-NN và SVM, lưu models\digit_model.joblib (~4 phút)
.venv\Scripts\python scripts\evaluate.py --save-vis 10    :: 40 ảnh test, kết quả vào reports\
.venv\Scripts\python scripts\evaluate.py --no-correct     :: tắt bước tự sửa (để so sánh)
.venv\Scripts\python scripts\evaluate.py --folder samples :: ảnh tự chụp kèm nhãn .dat
.venv\Scripts\python scripts\evaluate.py --no-model       :: chỉ đánh giá tìm lưới + phát hiện ô trống
```

Tuỳ chọn khi train: `--model svm|knn`, `--feature raw`, `--augment 0`, `--no-mnist`, `--no-chars74k`, `--mnist-per-class 2000`.

**Thêm ảnh tự chụp để kiểm tra:** đặt ảnh vào `samples/` và tạo file `.dat` cùng tên theo định dạng của dataset (2 dòng mô tả, rồi 9 dòng × 9 số, 0 = ô trống), xem `samples/handwritten_01.dat`.

Độ đo trong `reports\eval_*.json`:
- `grid_found_rate`, `grid_iou_mean`: bước tìm lưới (so với `outlines_sorted.csv`)
- `empty_detection`: tỉ lệ phát hiện đúng ô trống và ô có số
- `digit_accuracy`, `cell_accuracy`: độ chính xác nhận dạng (trước bước tự sửa)
- `grid_accuracy`, `grid_accuracy_after_correction`: tỉ lệ ảnh đúng cả 81 ô
- `solution_correct_rate`: tỉ lệ ảnh có đáp án trùng với đáp án của lưới nhãn thật

## Ứng dụng

```bat
.venv\Scripts\python app.py   :: chạy từ mã nguồn
build_exe.bat                  :: đóng gói thành dist\SudokuVision.exe (kèm model nếu đã train)
```

- **Mở ảnh…** hoặc **Ảnh mẫu ngẫu nhiên**. Có thể kéo thả ảnh lên file exe.
- Mỗi tab hiển thị một bước của pipeline: Ảnh gốc, Nhị phân, Tìm lưới, Nắn phẳng, Các ô, Nhận dạng, Lời giải.
- Lưới bên phải: cam = độ tin cậy thấp, tím = ô đã tự sửa, xanh lá = đáp án. Có thể sửa tay rồi bấm **Giải**.
- Exe chạy độc lập vì model được đóng gói sẵn bên trong. Nếu đặt `models\digit_model.joblib` cạnh exe thì model đó được ưu tiên dùng.

## Phát hành trên GitHub

Repo không chứa dữ liệu, model và file exe (xem `.gitignore`). Khi đẩy một tag, workflow `.github/workflows/release.yml` sẽ tự tải dữ liệu, train, đánh giá, đóng gói exe, rồi đính kèm `SudokuVision.exe` cùng model vào trang Releases:

```bat
git tag v1.0.0
git push origin v1.0.0
```

## Cấu trúc

```
sudoku_vision/   thư viện pipeline
scripts/         download_data.py, train.py, evaluate.py
app.py           giao diện Tkinter
build_exe.bat    đóng gói PyInstaller
samples/         ảnh tự chụp kèm nhãn
reports/         kết quả train và đánh giá
data/            dữ liệu tải về (không đưa lên repo)
models/          model sau khi train (không đưa lên repo)
```

## Lưu ý về dataset

Có 9 ảnh mang thẻ xoay EXIF (ví dụ `image1031.jpg`), trong khi nhãn `.dat` ứng với hướng pixel gốc. Vì vậy code đọc ảnh dataset bằng `imread(..., ignore_orientation=True)`. Ảnh người dùng mở trong app vẫn được xoay theo EXIF như bình thường.

## Nguồn dữ liệu

- [wichtounet/sudoku_dataset](https://github.com/wichtounet/sudoku_dataset), CC-BY-4.0
- [Chars74K](https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/), T. de Campos, B. R. Babu, M. Varma (2009)
- [MNIST](https://yann.lecun.com/exdb/mnist/), Y. LeCun, C. Cortes, C. Burges
