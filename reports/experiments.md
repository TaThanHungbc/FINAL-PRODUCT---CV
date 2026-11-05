# Kết quả thực nghiệm

Tập test: 40 ảnh của wichtounet/sudoku_dataset V2. Ảnh tự chụp: 1 ảnh trong samples/.

## Bộ phân loại

| Cấu hình | Mẫu train | Val chữ in | MNIST viết tay | Test: đúng theo ô | Test: ảnh đúng 81 ô | Test: đáp án đúng | + tự sửa | Ảnh tự chụp: đáp án đúng |
|---|---|---|---|---|---|---|---|---|
| k-NN + HOG, chỉ ô Sudoku | 11919 | 98.1% | 55.2% | 99.4% | 70.0% | 70.0% | 87.5% | 100.0% |
| SVM + HOG, chỉ ô Sudoku | 11919 | 99.5% | 39.6% | 99.8% | 87.5% | 87.5% | 100.0% | 100.0% |
| SVM + pixel thô, đủ dữ liệu | 30063 | 97.3% | 94.7% | 99.2% | 67.5% | 70.0% | 82.5% | 0.0% |
| k-NN + HOG, đủ dữ liệu | 30063 | 97.9% | 95.9% | 99.4% | 70.0% | 70.0% | 87.5% | 100.0% |
| SVM + HOG, đủ dữ liệu, không tăng cường | 22117 | 99.0% | 98.1% | 99.7% | 82.5% | 82.5% | 95.0% | 100.0% |
| SVM + HOG, đủ dữ liệu (model chính) | 30063 | 99.1% | 98.1% | 99.8% | 87.5% | 87.5% | 100.0% | 100.0% |

## Tìm lưới

| Nhị phân hoá | Tìm thấy lưới | IoU trung bình | IoU > 0,9 |
|---|---|---|---|
| adaptive | 100.0% | 0.949 | 95.0% |
| otsu | 100.0% | 0.732 | 15.0% |
