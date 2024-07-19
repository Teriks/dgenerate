export DGENERATE_POETRY_LOCKFILE_PATH="$(pwd)/poetry/poetry.lock"
export DGENERATE_POETRY_PYPROJECT_PATH="$(pwd)/poetry/pyproject.toml"

pip install build
rm -rf docker/dist
python3 -m build --outdir docker/dist
latest_whl=$(realpath $(ls -t docker/dist/*.whl | head -n 1))


pip3 install "${latest_whl}[flax, ncnn]" --extra-index-url https://download.pytorch.org/whl/cu121/ \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html