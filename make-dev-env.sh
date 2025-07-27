#!/bin/bash

rm -rf "$(dirname "$0")/venv"

python3 -m venv venv

source venv/bin/activate

if which rocminfo &> /dev/null; then
  pip3 install --editable "$(dirname "$0")[dev,ncnn,gpt4all,console_ui_opengl]" --extra-index-url https://download.pytorch.org/whl/rocm6.3/
else
  pip3 install --editable "$(dirname "$0")[dev,ncnn,gpt4all_cuda,quant,console_ui_opengl]" --extra-index-url https://download.pytorch.org/whl/cu128/
fi