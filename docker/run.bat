@echo off

mkdir "%~dp0..\docker_cache" 2> NUL
mkdir "%~dp0..\docker_cache\huggingface" 2> NUL
mkdir "%~dp0..\docker_cache\dgenerate" 2> NUL

docker rm -f dgenerate
docker run --gpus all --name dgenerate -v "%~dp0..\:/opt/dgenerate" -v "%~dp0..\docker_cache\huggingface:/home/dgenerate/.cache/huggingface" -v "%~dp0..\docker_cache\dgenerate:/home/dgenerate/.cache/dgenerate" -it teriks/dgenerate:3.4.3 bash -c "source docker/install.sh; bash"