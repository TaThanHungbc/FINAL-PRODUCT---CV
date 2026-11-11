# Sudoku Vision (CS231)

Giải Sudoku từ ảnh chụp bằng xử lý ảnh cổ điển: OpenCV để tìm lưới và cắt ô, HOG + SVM (scikit-learn) để nhận dạng chữ số in lẫn viết tay, backtracking để giải. Khi nhận dạng sai, chính luật Sudoku được dùng để tự sửa.

**Input:** ảnh chụp một ô Sudoku trên báo/giấy (chữ in hoặc viết tay). **Output:** lưới 9×9 nhận dạng được, lời giải, và ảnh gốc có đáp án vẽ đè lên các ô trống.

## Kết quả

Tỉ lệ ảnh giải ra **đáp án đúng** (pipeline đầy đủ, có bước tự sửa):

| Bộ ảnh | v2 (MNIST, chưa xử lý riêng chữ viết tay) | v3 (hiện tại) |
|---|---|---|
| 40 ảnh test của dataset (chữ in, ảnh báo thật) | 100% | 100% |
| 60 ảnh Sudoku viết tay tổng hợp (`data/synthetic_hw`) | 18,3% | **86,7%** |
| Ảnh viết tay bút chì tự chụp (`samples/`) | 100% | 100% |

Độ chính xác trên từng ô chữ số (validation): chữ in 99,4%, MNIST 98,5%, ô viết tay tổng hợp 97,6%.

Những thay đổi ở v3 để nhận dạng chữ viết tay tốt hơn:
- **Ảnh Sudoku viết tay tổng hợp để train:** `make_synthetic.py --split train` viết các đề bài bằng chữ số MNIST lên lưới giấy, mô phỏng bút chì/bút bi, chụp nghiêng, mờ, nhiễu, nén JPEG. Các ô được cắt bằng chính pipeline nên dữ liệu train giống hệt những gì bộ phân loại nhìn thấy khi chạy thật.
- **Xoá đoạn kẻ lưới còn sót trong ô:** trước đây các mảnh hình chữ L của đường kẻ bị lấy làm mẫu "nhiễu" (lớp 0), khiến số 2, 7, 1 viết tay hay bị nhận thành ô trống. Số mẫu nhiễu giả giảm từ 280 xuống 46.
- **Chỉ coi một ô là trống khi chắc chắn** (xác suất lớp nhiễu ≥ 0,7). Bỏ sót một số đề bài làm hỏng cả đề, còn một số thừa vẫn có thể được bước tự sửa loại bỏ.
- **Tự sửa cả khi đề có nhiều lời giải**, không chỉ khi đề vô nghiệm: đó là dấu hiệu một số đề bài bị đọc thành ô trống.
- **Học nét chữ của người dùng:** nút **Lưu làm dữ liệu train** trong app (xem phần Ứng dụng).

Bộ viết tay tổng hợp dùng để đánh giá được tạo từ chữ số MNIST t10k và đề bài của tập test, còn bộ để train dùng MNIST train và đề bài của tập train, nên hai bộ độc lập với nhau. Dù vậy đây vẫn là ảnh mô phỏng; với chữ viết tay thật, cách đánh giá tốt nhất là thêm ảnh tự chụp vào `samples/`.

Bảng so sánh các cấu hình (k-NN/SVM, HOG/pixel thô, nguồn dữ liệu, tăng cường, tự sửa, adaptive/Otsu) ở `reports/experiments.md`. Kết quả của bản đầu tiên ở `reports/v1_knn_print/`.

## Pipeline

| Bước | Module | Kỹ thuật |
|---|---|---|
| 1. Tiền xử lý | `preprocess.py` | Thu nhỏ, ảnh xám, Gaussian blur, adaptive threshold (hoặc Otsu) |
| 2. Tìm lưới | `grid.py` | Giãn nét, contour lớn nhất, `approxPolyDP` lấy 4 góc |
| 3. Nắn phẳng | `grid.py` | `getPerspectiveTransform` + `warpPerspective` về 450×450 |
| 4. Cắt ô | `cells.py` | Chia 81 ô, xoá viền và đoạn kẻ lưới còn sót, connected components, gộp các mảnh nét bị đứt, chuẩn hoá 28×28 |
| 5. Nhận dạng | `features.py`, `classifier.py` | HOG (324 chiều), SVM 10 lớp (0 = nhiễu, 1–9) |
| 6. Giải | `solver.py` | Backtracking + MRV, bitmask |
| 6b. Tự sửa | `correction.py` | Đề trùng số / vô nghiệm / nhiều lời giải: thử thay các ô kém tin cậy bằng lựa chọn thứ 2, 3 của SVM, ưu tiên tổ hợp có xác suất cao nhất và cho ra đề có đúng một lời giải |
| 7. Hiển thị | `overlay.py` | Vẽ đáp án trên ảnh phẳng, warp ngược về ảnh gốc |

Tham số của mọi bước nằm trong `sudoku_vision/config.py`.

**Dữ liệu train bộ phân loại:** ô cắt từ 160 ảnh train của dataset Sudoku (chữ in thật, kèm mẫu nhiễu làm lớp 0), Chars74K EnglishFnt (chữ in nhiều font), MNIST (viết tay, 1.500 mẫu/lớp), ô cắt từ 200 ảnh Sudoku viết tay tổng hợp, và ô người dùng lưu từ app (nếu có).

## Cài đặt

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\download_data.py
.venv\Scripts\python scripts\make_synthetic.py --split train   :: ảnh viết tay tổng hợp để train
.venv\Scripts\python scripts\make_synthetic.py                 :: ảnh viết tay tổng hợp để đánh giá
```

Nên dùng `.venv` cho cả train lẫn đóng gói exe: model `.joblib` phải được nạp bằng đúng phiên bản scikit-learn đã train ra nó.

## Train và đánh giá

```bat
.venv\Scripts\python scripts\train.py                     :: so sánh k-NN và SVM, lưu models\digit_model.joblib (~4 phút)
.venv\Scripts\python scripts\evaluate.py --save-vis 10    :: 40 ảnh test, kết quả vào reports\
.venv\Scripts\python scripts\evaluate.py --no-correct     :: tắt bước tự sửa (để so sánh)
.venv\Scripts\python scripts\evaluate.py --folder samples :: ảnh tự chụp kèm nhãn .dat
.venv\Scripts\python scripts\evaluate.py --folder data\synthetic_hw :: ảnh viết tay tổng hợp
.venv\Scripts\python scripts\evaluate.py --no-model       :: chỉ đánh giá tìm lưới + phát hiện ô trống
.venv\Scripts\python scripts\experiments.py                :: bảng so sánh cho báo cáo -> reports\experiments.md (~15 phút)
.venv\Scripts\python scripts\solve.py anh.jpg --stages     :: giải ảnh bất kỳ từ dòng lệnh, ảnh kết quả vào outputs\
.venv\Scripts\python -m unittest discover tests            :: kiểm thử
```

Tuỳ chọn khi train: `--model svm|knn`, `--feature raw`, `--augment 0`, `--no-mnist`, `--no-chars74k`, `--no-synthetic`, `--no-user-digits`, `--mnist-per-class 3000`.

**Thêm ảnh tự chụp để kiểm tra:** đặt ảnh vào `samples/` và tạo file `.dat` cùng tên theo định dạng của dataset (2 dòng mô tả, rồi 9 dòng × 9 số, 0 = ô trống), xem `samples/handwritten_01.dat`.

Độ đo trong `reports\eval_*.json`:
- `grid_found_rate`, `grid_iou_mean`: bước tìm lưới (so với `outlines_sorted.csv`)
- `empty_detection`: tỉ lệ phát hiện đúng ô trống và ô có số
- `digit_accuracy`, `cell_accuracy`: độ chính xác nhận dạng (trước bước tự sửa)
- `grid_accuracy`, `grid_accuracy_after_correction`: tỉ lệ ảnh đúng cả 81 ô
- `solution_correct_rate`: tỉ lệ ảnh có đáp án trùng với đáp án của lưới nhãn thật
- `solved_not_unique`: số ảnh giải được nhưng đề nhận dạng có nhiều lời giải (thường do bỏ sót số đề bài)

## Ứng dụng

```bat
.venv\Scripts\python app.py   :: chạy từ mã nguồn
build_exe.bat                  :: đóng gói thành dist\SudokuVision.exe (kèm model nếu đã train)
```

- **Mở ảnh…**, **Ảnh mẫu ngẫu nhiên**, hoặc **Webcam…**: xem trước có vẽ khung lưới tìm được, bấm Space để chụp. Có thể kéo thả ảnh lên file exe.
- Mỗi tab hiển thị một bước của pipeline: Ảnh gốc, Nhị phân, Tìm lưới, Nắn phẳng, Các ô, Nhận dạng, Lời giải.
- Lưới bên phải: cam = độ tin cậy thấp, tím = ô đã tự sửa, xanh lá = đáp án. Có thể sửa tay rồi bấm **Giải**.
- Nếu đề nhận dạng được có nhiều hơn một lời giải, app cảnh báo (chữ cam) vì nhiều khả năng đã bỏ sót một số đề bài.
- **Lưu làm dữ liệu train:** khi model đọc sai nét chữ của bạn, sửa lưới cho đúng rồi bấm nút này. Các ô được lưu vào `data\user_digits\<số>\` (ô để trống trên lưới được lưu là nhiễu). Chạy lại `scripts\train.py` để model học thêm; các ô này được nhân bản tăng cường 6 lần vì thường rất ít.
- Exe chạy độc lập vì model được đóng gói sẵn bên trong. Nếu đặt `models\digit_model.joblib` cạnh exe thì model đó được ưu tiên dùng.

## Phát hành trên GitHub

Repo không chứa dữ liệu, model và file exe (xem `.gitignore`). Workflow `tests.yml` chạy kiểm thử mỗi lần đẩy code. Khi đẩy một tag, workflow `.github/workflows/release.yml` sẽ tự tải dữ liệu, train, đánh giá, đóng gói exe, rồi đính kèm `SudokuVision.exe` cùng model vào trang Releases:

```bat
git tag v1.0.0
git push origin v1.0.0
```

## Cấu trúc

```
sudoku_vision/   thư viện pipeline
scripts/         download_data.py, make_synthetic.py, train.py, evaluate.py, experiments.py, solve.py
tests/           kiểm thử (unittest)
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
