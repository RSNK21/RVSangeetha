@echo off
rem Run RVSangeetha from source (run on Windows with Python 3.10+ installed)
if not exist .venv (
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
python -m rvsangeetha %*
