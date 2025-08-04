# Nix Environment Setup Guide

Very basic environment for testing nix features on Windows, this should also work on *nix platforms.

If running from windows, make sure you have the latest version of WSL, Docker For Windows, and your graphics driver.

## Usage

To start a shell use:
```bash
python run.py
```

To run a command in a fresh environment use for example:

```bash
python run.py -e CIVIT_AI_TOKEN=tokenhere -e HF_TOKEN=tokenhere "python3 examples/run.py --short-animations --subprocess-only --skip ncnn &> examples/examples-docker.log"

python run.py -e CIVIT_AI_TOKEN=tokenhere -e HF_TOKEN=tokenhere "python3 run_tests.py --clean --examples --skip ncnn"

python run.py -e CIVIT_AI_TOKEN=tokenhere -e HF_TOKEN=tokenhere "python3 run_tests.py --clean --examples --subprocess-only --skip ncnn"
```

## Options

You can also use the `--dev` option of `run.py` to indicate that you want a development install from the current directory instead of building and then installing a wheel.

To install and use the ROCm torch backend (AMD Cards), use the `--amd` option of `run.py`. This only works from a linux host.

## Important Notes

- The initial working directory of the environment is the top level directory of the dgenerate project
- It is a linux environment, therefore the python executable is named `python3`
- The value of `DISPLAY` in the container is set to `host.docker.internal:0.0`

## Cross Platform Testing

You may run `dgenerate --console` (the Console UI) on Windows for cross-platform testing using Xming (X Server for windows), see: http://www.straightrunning.com/XmingNotes/

(Yes, that website has no certificate, it is fine)