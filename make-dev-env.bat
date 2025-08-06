rmdir /s /q "%~dp0venv" 2>nul

python -m venv venv

call venv\Scripts\activate

pip install --editable "%~dp0[dev,ncnn,gpt4all_cuda,bitsandbytes,triton_windows,console_ui_opengl]" --extra-index-url https://download.pytorch.org/whl/cu128/