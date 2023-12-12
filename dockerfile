FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
RUN apt update && apt install -y python3.10 python3-pip python3.10-venv python3-wheel
COPY setup.py README.rst /opt/dgenerate/
COPY dgenerate /opt/dgenerate/dgenerate
COPY poetry /opt/dgenerate/poetry
WORKDIR "/opt/dgenerate"


