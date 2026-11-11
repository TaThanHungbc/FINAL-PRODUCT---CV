# Kết quả thực nghiệm

Tập test: 40 ảnh của wichtounet/sudoku_dataset V2. Viết tay tổng hợp: 60 ảnh (data/synthetic_hw). Ảnh tự chụp: 1 ảnh trong samples/.

## Bộ phân loại

Độ chính xác trên ô: validation chữ in, MNIST, ô viết tay tổng hợp. Đáp án đúng: tỉ lệ ảnh giải đúng.

| Cấu hình | Mẫu train | Ô chữ in | Ô MNIST | Ô viết tay tổng hợp | Test: ảnh đúng 81 ô | Test: đáp án đúng | + tự sửa | Viết tay tổng hợp: đáp án đúng | Ảnh tự chụp: đáp án đúng |
|---|---|---|---|---|---|---|---|---|---|
| k-NN + HOG, chỉ ô Sudoku | 11310 | 99.7% | 59.1% | 58.6% | 92.5% | 92.5% | 100.0% | 0.0% | 100.0% |
| SVM + HOG, chỉ ô Sudoku | 11310 | 99.5% | 53.4% | 52.9% | 92.5% | 92.5% | 100.0% | 0.0% | 100.0% |
| SVM + pixel thô, đủ dữ liệu | 39726 | 99.3% | 96.4% | 94.2% | 90.0% | 92.5% | 100.0% | 65.0% | 100.0% |
| k-NN + HOG, đủ dữ liệu | 39726 | 99.3% | 96.3% | 93.5% | 90.0% | 90.0% | 100.0% | 60.0% | 100.0% |
| SVM + HOG, không có ảnh viết tay tổng hợp | 33954 | 99.2% | 98.3% | 92.8% | 90.0% | 90.0% | 100.0% | 50.0% | 100.0% |
| SVM + HOG, đủ dữ liệu, không tăng cường | 32186 | 99.2% | 98.5% | 97.2% | 90.0% | 90.0% | 100.0% | 83.3% | 100.0% |
| SVM + HOG, đủ dữ liệu (model chính) | 39726 | 99.3% | 98.5% | 97.6% | 90.0% | 90.0% | 100.0% | 86.7% | 100.0% |

## Tìm lưới

| Nhị phân hoá | Tìm thấy lưới | IoU trung bình | IoU > 0,9 |
|---|---|---|---|
| adaptive | 100.0% | 0.949 | 95.0% |
| otsu | 100.0% | 0.732 | 15.0% |
