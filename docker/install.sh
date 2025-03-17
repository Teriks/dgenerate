export DGENERATE_POETRY_LOCKFILE_PATH="$(pwd)/poetry/poetry.lock"
export DGENERATE_POETRY_PYPROJECT_PATH="$(pwd)/poetry/pyproject.toml"

mkdir -p ~/dgenerate_venv
python3 -m venv ~/dgenerate_venv
source ~/dgenerate_venv/bin/activate

if command -v nvidia-smi &> /dev/null; then
    extras="[quant,ncnn,gpt4all_cuda]"
else
    extras="[ncnn,gpt4all]"
fi

if [ "$DGENERATE_INSTALL_DEV" = "1" ]; then
    pip3 install --editable ".${extras}" --extra-index-url "$DGENERATE_INSTALL_INDEX"
else
    pip3 install build
    rm -rf docker/dist
    python3 -m build --outdir docker/dist

    latest_whl=$(find docker/dist -name "*.whl" -print0 | xargs -r -0 ls -t | head -n 1)

    if [ -n "$latest_whl" ]; then
        pip3 install "${latest_whl}${extras}" --extra-index-url "$DGENERATE_INSTALL_INDEX"
    else
        echo "No .whl file found in docker/dist"
        exit 1
    fi
fi
