ARG BASE_IMAGE="debian12-cuda12.9-py3.12:latest"
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps (keep minimal; add more only when needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv for isolation
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch (CUDA 12.9 / cu129) for Python 3.12
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.8.0+cu129

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Miles (install without deps; we pin deps above)
ARG MILES_COMMIT="be6cc286b7483fe55e7bd53a03d8fbbcf8c73826"
RUN python -m pip install --no-cache-dir --no-deps \
    "git+https://github.com/radixark/miles.git@${MILES_COMMIT}"

# Repo code
COPY async_rl /app/async_rl
COPY sitecustomize.py /app/sitecustomize.py
COPY entrypoint.sh /app/entrypoint.sh
COPY configs /app/configs

RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH="/app"

# Reasonable defaults; can be overridden by the platform.
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ENTRYPOINT ["/app/entrypoint.sh"]
