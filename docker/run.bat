docker rm -f dgenerate
docker run --gpus all --name dgenerate -v "%~dp0..\:/opt/dgenerate" -it teriks/dgenerate:3.3.0 bash -c "source docker/install.sh; bash"