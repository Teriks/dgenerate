rmdir /s /q "%~dp0venv" 2>nul

python -m venv venv

call venv\Scripts\activate

pip install --editable "%~dp0[dev,ncnn]" --extra-index-url https://download.pytorch.org/whl/cu124/