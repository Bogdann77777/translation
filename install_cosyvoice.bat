@echo off
REM ============================================
REM CosyVoice3 Installation Script for Windows
REM ============================================

echo.
echo ========================================
echo CosyVoice3 Installation
echo ========================================
echo.

REM Check if conda environment is activated
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please activate your conda environment first.
    pause
    exit /b 1
)

echo [1/5] Installing Python dependencies...
pip install modelscope soundfile

echo.
echo [2/5] Creating third_party directory...
if not exist "third_party" mkdir third_party

echo.
echo [3/5] Cloning CosyVoice repository...
cd third_party
if exist "CosyVoice" (
    echo CosyVoice already exists, skipping clone...
) else (
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
)
cd ..

echo.
echo [4/5] Installing CosyVoice dependencies...
pip install -r third_party\CosyVoice\requirements.txt

echo.
echo [5/5] Downloading CosyVoice3-0.5B model...
if not exist "models" mkdir models

python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='models/CosyVoice3-0.5B')"

echo.
echo ========================================
echo CosyVoice3 Installation Complete!
echo ========================================
echo.
echo Model location: models\CosyVoice3-0.5B
echo Voice samples: voice_samples\
echo.
echo To use CosyVoice3:
echo 1. Make sure config.yaml has: provider: "cosyvoice"
echo 2. Start server: python app/main.py
echo.

pause
