rmdir /s /q "%~dp0venv" 2>nul

python -m venv venv

call venv\Scripts\activate

nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    pip install --editable "%~dp0[dev,ncnn,gpt4all_cuda,bitsandbytes,triton_windows,console_ui_opengl]" --extra-index-url https://download.pytorch.org/whl/cu128/
) else (
    xpu-smi >nul 2>&1
    if %errorlevel% equ 0 (
        pip install --editable "%~dp0[dev,ncnn,gpt4all,bitsandbytes,console_ui_opengl]" --extra-index-url https://download.pytorch.org/whl/xpu/
    ) else (
        pip install --editable "%~dp0[dev,ncnn,gpt4all,console_ui_opengl]"
    )
)