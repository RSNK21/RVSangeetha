@echo off
rem Build RVSangeetha.exe (run on Windows with Python 3.10+ installed)
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --windowed --name RVSangeetha ^
    --collect-all pyqtgraph ^
    --collect-binaries soundfile ^
    --collect-binaries sounddevice ^
    rvsangeetha\__main__.py
echo.
echo Done. The application is in dist\RVSangeetha\RVSangeetha.exe
pause
