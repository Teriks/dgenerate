#!/bin/bash

rm -rf "$(dirname "$0")/venv"

python3 -m venv venv

source venv/bin/activate

if which rocminfo &> /dev/null; then
  pip3 install --editable "$(dirname "$0")[dev,ncnn,gpt4all_cuda]" --extra-index-url https://download.pytorch.org/whl/rocm6.2.4/
else
  pip3 install --editable "$(dirname "$0")[dev,ncnn,gpt4all_cuda]" --extra-index-url https://download.pytorch.org/whl/cu124/
fi