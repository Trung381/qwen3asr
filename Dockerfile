# Based on official Dockerfile-qwen3-asr-cu128
ARG CUDA_VERSION=12.8.0
ARG from=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
FROM ${from} as base

ARG DEBIAN_FRONTEND=noninteractive

# System deps — python3-dev is CRITICAL so Triton can compile its C extension (cuda_utils.c)
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    vim \
    libsndfile1 \
    ccache \
    software-properties-common \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install a recent cmake (required by some vLLM build deps)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN git lfs install

WORKDIR /app

ENV MAX_JOBS=32
ENV NVCC_THREADS=2
ENV CCACHE_DIR=/root/.cache/ccache

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -U pip setuptools wheel

RUN apt remove python3-blinker -y || true

# Install qwen-asr with vLLM backend
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -U "qwen-asr[vllm]"

# Optional: Flash Attention — makes forced_aligner much faster (can be slow to build)
ARG BUNDLE_FLASH_ATTENTION=false
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
        pip3 install -U flash-attn --no-build-isolation; \
    fi

# Install FastAPI server dependencies + gRPC
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install fastapi uvicorn python-multipart grpcio grpcio-tools

# Fix: Triton links against -lcuda at compile time, and vLLM spawns child processes
# using Python's 'spawn' method (fresh env). We need libcuda.so visible system-wide
# so ALL processes (including spawned subprocesses) can find it.
#
# Step 1: Add /usr/local/cuda/lib64/stubs to system linker (has unversioned libcuda.so)
RUN echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig

# Step 2: Also symlink into Triton's own lib dir (belt-and-suspenders)
RUN TRITON_LIB=$(python3 -c "import triton, os; print(os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/lib'))") \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "$TRITON_LIB/libcuda.so" \
    && echo "Symlinked /usr/local/cuda/lib64/stubs/libcuda.so -> $TRITON_LIB/libcuda.so"

RUN rm -rf /root/.cache/pip

# Copy proto definition and compile to Python stubs
COPY proto/ ./proto/
RUN mkdir -p app/proto \
    && python3 -m grpc_tools.protoc \
        -I proto \
        --python_out=app/proto \
        --grpc_python_out=app/proto \
        proto/asr.proto \
    && touch app/proto/__init__.py \
    && sed -i 's/^import asr_pb2/from . import asr_pb2/' app/proto/asr_pb2_grpc.py \
    && echo "Proto stubs generated and import patched."

# Copy the server application
COPY app/ ./app/

EXPOSE 8000
EXPOSE 50051

CMD ["python3", "-m", "app.server"]
