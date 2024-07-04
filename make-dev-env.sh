#!/bin/bash

rm -rf "$(dirname "$0")/venv"

python3 -m venv venv

source venv/bin/activate

pip3 install --editable "$(dirname "$0")[dev]" --extra-index-url https://download.pytorch.org/whl/cu121/