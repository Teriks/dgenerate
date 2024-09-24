export DGENERATE_POETRY_LOCKFILE_PATH="$(pwd)/poetry/poetry.lock"
export DGENERATE_POETRY_PYPROJECT_PATH="$(pwd)/poetry/pyproject.toml"

if [ "$DGENERATE_INSTALL_DEV" = "1" ]; then
  pip3 install --editable . --extra-index-url $DGENERATE_INSTALL_INDEX
else
  pip install build
  rm -rf docker/dist
  python3 -m build --outdir docker/dist
  latest_whl=$(realpath $(ls -t docker/dist/*.whl | head -n 1))


  pip3 install "${latest_whl}" --extra-index-url $DGENERATE_INSTALL_INDEX
fi

