export DGENERATE_POETRY_LOCKFILE_PATH="$(pwd)/poetry/poetry.lock"
export DGENERATE_POETRY_PYPROJECT_PATH="$(pwd)/poetry/pyproject.toml"

mkdir ~/dgenerate_venv
python3 -m venv ~/dgenerate_venv
source ~/dgenerate_venv/bin/activate

if [ "$DGENERATE_INSTALL_DEV" = "1" ]; then
  pip3 install --editable . --extra-index-url $DGENERATE_INSTALL_INDEX
else
  pip3 install build
  rm -rf docker/dist
  python3 -m build --outdir docker/dist
  latest_whl=$(realpath $(ls -t docker/dist/*.whl | head -n 1))


  pip3 install "${latest_whl}" --extra-index-url $DGENERATE_INSTALL_INDEX
fi

