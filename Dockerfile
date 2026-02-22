ARG BASE_IMAGE="debian12-cuda12.9-py3.12:latest"
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps. We include build tools because some Python deps
# (e.g. flash-attn / apex / ring_flash_attn) may need compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv for isolation
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# Torch: keep CUDA wheel channel configurable for cu128/cu129/cu130 bases.
ARG TORCH_VERSION="2.8.0"
ARG TORCH_CUDA_TAG="cu129"
RUN python -m pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
    "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}"

# Shared Python deps used by this repo + Miles runtime.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Optional Megatron stack for unified FP8 on bare CUDA+Python bases.
# Enable with: --build-arg INSTALL_MEGATRON_STACK=1
ARG INSTALL_MEGATRON_STACK="0"
ARG ENABLE_CUDA_13="0"
ARG MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"
ARG MBRIDGE_REF="89eb10887887bc74853f89a4de258c0702932a1c"
ARG TORCH_MEMORY_SAVER_REF="dc6876905830430b5054325fa4211ff302169c6b"
ARG MEGATRON_BRIDGE_REF="dev_rl"

RUN if [ "${INSTALL_MEGATRON_STACK}" = "1" ]; then \
      MAX_JOBS=64 python -m pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1 && \
      python -m pip install --no-cache-dir "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_REF}" --no-deps && \
      if [ "${ENABLE_CUDA_13}" = "1" ]; then \
        python -m pip install --no-cache-dir nvidia-mathdx==26.6.0 && \
        python -m pip install --no-cache-dir --no-build-isolation "git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10"; \
      else \
        python -m pip install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]==2.10.0"; \
      fi && \
      python -m pip install --no-cache-dir flash-linear-attention==0.4.0 && \
      python -m pip install --no-cache-dir tilelang -f https://tile-ai.github.io/whl/nightly/cu128/ && \
      NVCC_APPEND_FLAGS="--threads 4" \
      python -m pip install --no-cache-dir --disable-pip-version-check --no-build-isolation \
      --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
      "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4" && \
      git clone --recursive https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM && \
      cd /opt/Megatron-LM && \
      git checkout "${MEGATRON_COMMIT}" && \
      python -m pip install --no-cache-dir -e . && \
      python -m pip install --no-cache-dir --force-reinstall \
      "git+https://github.com/fzyzcjy/torch_memory_saver.git@${TORCH_MEMORY_SAVER_REF}" && \
      python -m pip install --no-cache-dir --no-build-isolation \
      "git+https://github.com/fzyzcjy/Megatron-Bridge.git@${MEGATRON_BRIDGE_REF}" && \
      python -m pip install --no-cache-dir --no-build-isolation "nvidia-modelopt[torch]>=0.37.0"; \
    else \
      echo "Skipping Megatron stack install (INSTALL_MEGATRON_STACK=${INSTALL_MEGATRON_STACK})"; \
    fi

# Miles (install without deps; we pin deps above)
# Defaults point to the fork commit that includes GPT-OSS unified-FP8 bridge fixes.
ARG MILES_REPO="https://github.com/hongyi-zhang/miles.git"
ARG MILES_REF="3cba08a9a8a4f53f1d128a85efb86364cf39589b"
RUN python -m pip install --no-cache-dir --no-deps \
    "git+${MILES_REPO}@${MILES_REF}"

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
