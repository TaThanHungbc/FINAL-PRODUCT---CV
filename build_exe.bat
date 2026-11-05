@echo off
rem Dong goi app.py thanh dist\SudokuVision.exe (PyInstaller, 1 file, khong cua so console)
rem Neu da co models\digit_model.joblib thi model duoc dong goi luon vao exe.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Tao moi truong ao .venv ...
    python -m venv .venv || goto :error
)

echo [2/3] Cai thu vien ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error

set "ADD_MODEL="
if exist "models\digit_model.joblib" (
    set "ADD_MODEL=--add-data models\digit_model.joblib;models"
    echo     Dong goi kem models\digit_model.joblib
) else (
    echo     Chua co model: exe se chay o che do chi tim luoi va cat o
)

echo [3/3] Dong goi exe ...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name SudokuVision %ADD_MODEL% ^
    --exclude-module matplotlib --exclude-module pandas --exclude-module IPython ^
    app.py || goto :error

echo.
echo Xong: dist\SudokuVision.exe
echo Model dat canh exe (models\digit_model.joblib) se duoc uu tien hon model dong goi san.
exit /b 0

:error
echo Dong goi that bai.
exit /b 1
