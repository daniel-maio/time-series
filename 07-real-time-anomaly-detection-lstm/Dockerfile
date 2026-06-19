FROM python:3.14-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        curl \
        git \
        vim \
        nano \
        unzip \
        rsync \
        build-essential \
        netcat-openbsd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project/requirements

WORKDIR /workspace/project/requirements

COPY workdir/requirements/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu126

WORKDIR /workspace/project

ENV PYTHONPATH=/workspace/python:$PYTHONPATH

EXPOSE 9999

CMD ["bash"]