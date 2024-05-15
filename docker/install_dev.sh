export DGENERATE_POETRY_LOCKFILE_PATH="$(pwd)/poetry/poetry.lock"
export DGENERATE_POETRY_PYPROJECT_PATH="$(pwd)/poetry/pyproject.toml"

pip3 install --editable .[flax] --extra-index-url https://download.pytorch.org/whl/cu121/ \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html