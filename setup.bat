@echo off
title JARVIS V3 Setup
echo.
echo  ============================================
echo   J.A.R.V.I.S  V3  Setup  (Windows + RTX)
echo  ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)
echo [OK] Python found

echo.
echo [1/6] Installing PyTorch with CUDA 12.4...
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 -q
echo [OK] PyTorch installed

echo.
echo [2/6] Installing core dependencies...
pip install faster-whisper kokoro sounddevice soundfile pytz resemblyzer numpy -q
echo [OK] Core deps done

echo.
echo [3/6] Installing LLM and memory deps...
pip install ollama requests -q
echo [OK] LLM deps done

echo.
echo [4/6] Installing fine-tuning deps...
pip install unsloth trl datasets transformers accelerate bitsandbytes schedule -q
echo [OK] Fine-tuning deps done

echo.
echo [5/6] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama not installed.
    echo        Download: https://ollama.com/download
    echo        Then run: ollama pull llama3.2
) else (
    echo [OK] Ollama found — pulling llama3.2...
    ollama pull llama3.2
)

echo.
echo [6/6] Verifying GPU...
python -c "import torch; g=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'; m=torch.cuda.get_device_properties(0).total_memory//1024**3 if torch.cuda.is_available() else 0; print(f'  GPU: {g} ({m}GB VRAM)')"

echo.
echo  ============================================
echo  JARVIS V3 Setup Complete!
echo.
echo  Before starting:
echo    1. Copy your_voice_sample.wav from V2
echo    2. Run in separate terminal: ollama serve
echo.
echo  Commands:
echo    Start Jarvis:    python main.py
echo    Fine-tune:       run_finetune.bat
echo    Auto-retrain:    python finetune\auto_train.py
echo  ============================================
echo.
pause