@echo off
title JARVIS V3 Fine-tuning
cd /d "%~dp0"

echo.
echo  ============================================
echo   JARVIS V3 Fine-tuning - Incremental
echo  ============================================
echo.

echo [1/4] Checking environment...
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
if errorlevel 1 (
    echo [ERROR] Python or Torch not found. Run setup.bat first.
    pause & exit /b 1
)

python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
if errorlevel 1 (
    echo [ERROR] Unsloth not found. Run setup.bat first.
    pause & exit /b 1
)
echo [OK] Environment ready

echo.
if exist "finetune\jarvis-finetuned\adapter_config.json" (
    echo [OK] Previous model found - incremental training
) else (
    echo [INFO] First run - starting from base llama3.2
)

echo.
echo [2/4] Building dataset...
python finetune\clean_memory.py
if errorlevel 1 (
    echo [ERROR] Dataset generation failed
    pause & exit /b 1
)
echo [OK] Dataset ready

echo.
echo [3/4] Training... (10-30 mins, watch GPU temp)
python -u finetune\train.py
if errorlevel 1 (
    echo [ERROR] Training failed
    pause & exit /b 1
)

if not exist "finetune\jarvis-finetuned-gguf\unsloth.Q4_K_M.gguf" (
    echo [ERROR] GGUF not created - training crashed
    pause & exit /b 1
)
echo [OK] Training complete

echo.
echo [4/4] Loading into Ollama...
ollama create jarvis -f Modelfile
if errorlevel 1 (
    echo [ERROR] Ollama failed
    pause & exit /b 1
)
echo [OK] Jarvis model loaded

echo.
echo  ============================================
echo  Done! Now update config.py:
echo    OLLAMA_MODEL = "jarvis"
echo  Then run: python main.py
echo  ============================================
pause