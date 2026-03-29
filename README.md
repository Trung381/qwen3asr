# Qwen3-ASR — Dockerized Speech-to-Text API

A production-ready HTTP API for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) running on vLLM, containerized with Docker for one-command deployment on any GPU machine.

## Project Structure

```
├── Dockerfile              # CUDA 12.8 image with all env fixes baked in
├── docker-compose.yml      # GPU passthrough, volume caching, config via env
├── .env.example            # Copy to .env and tune for your GPU
├── proto/
│   └── asr.proto           # gRPC service definition
├── app/
│   ├── server.py           # FastAPI (HTTP) + gRPC server
│   ├── grpc_servicer.py    # gRPC bidirectional streaming handler
│   └── proto/              # Python stubs (generated during Docker build)
├── client/
│   ├── main.go             # Go gRPC client: mic streaming + file test
│   ├── go.mod
│   └── proto/              # Go stubs (generated via `make proto`)
└── Makefile                # proto generation + build shortcuts
```

---

## Prerequisites

> The `Dockerfile`, `docker-compose.yml`, and `app/server.py` are identical across platforms. Only the one-time setup differs.

| | Windows + Docker Desktop | Native Ubuntu/Linux |
|---|---|---|
| Docker commands | PowerShell ✅ | Terminal ✅ |
| NVIDIA Toolkit | Installed via WSL | Installed directly on host |
| WSL required? | Only to install NVIDIA toolkit | ❌ Not needed |

### Windows (Docker Desktop + WSL2)

**1. Install Docker Desktop** — https://www.docker.com/products/docker-desktop/
- Choose **"Use WSL 2 instead of Hyper-V"** during setup
- After install: Docker Desktop → **Settings → Resources → WSL Integration** → enable for Ubuntu

**2. Install NVIDIA Container Toolkit** (inside WSL terminal):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```
Then **restart Docker Desktop** from the system tray.

**3. Verify GPU passthrough works** (PowerShell):
```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
✅ Expected: your GPU name and driver version appear.

---

### Native Ubuntu/Linux

**1. Install Docker Engine:**
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in
```

**2. Install NVIDIA Container Toolkit:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**3. Verify:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Setup & Run

**1. Clone and configure:**

```bash
# Windows
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit `.env` based on your GPU VRAM:

| GPU VRAM | `GPU_MEMORY_UTILIZATION` | `MAX_MODEL_LEN` |
|---|---|---|
| 6 GB (RTX 3060) | `0.80` | `2048` |
| 8 GB (RTX 3070) | `0.85` | `4096` |
| 12 GB (RTX 3080 Ti) | `0.90` | `8192` |
| 16 GB+ | `0.90` | `16384` |

**2. Build and start:**

```bash
# First time — builds image (~5-15 min, downloads CUDA base + deps)
docker compose build

# Start the server
docker compose up
```

> On first run, the model weights (~4 GB) are downloaded into `./models/`. Subsequent starts reuse the cache.

**3. Verify it's running:**
```bash
curl http://localhost:8000/health
# {"status":"ok","model":"Qwen/Qwen3-ASR-1.7B","ready":true}
```

---

## API Usage

### `POST /transcribe`

Upload an audio file (WAV, MP3, FLAC, etc.) to get a transcription.

```bash
# Basic
curl -X POST http://localhost:8000/transcribe -F "file=@audio.wav"

# With language hint
curl -X POST "http://localhost:8000/transcribe?language=English" -F "file=@audio.wav"
```

**Response:**
```json
{"language": "English", "text": "Hello, this is a test transcription."}
```

> On **Windows PowerShell**, use `curl.exe` instead of `curl` to avoid the PowerShell alias.

### `GET /health`
Returns `{"status":"ok","model":"...","ready":true,"grpc_port":50051}` when the model is loaded and ready.

### Swagger UI
Open **http://localhost:8000/docs** in your browser for an interactive API explorer.

---

## gRPC Streaming (Go Client)

For real-time, low-latency transcription, use the gRPC streaming API with the Go client.

### Prerequisites
```bash
# Install Go: https://go.dev/dl/
# Install protoc: https://grpc.io/docs/protoc-installation/
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# On Windows/Linux for PortAudio (mic input):
# Ubuntu/WSL: sudo apt-get install portaudio19-dev
# Windows: vcpkg install portaudio
```

### Setup
```bash
# Generate Go proto stubs
make proto

# Install Go dependencies
cd client && go mod tidy
```

### Stream a WAV file (smoke test)
```bash
make test-client
# or:
cd client && go run . --file ../test_en.wav --language English
```

### Live microphone streaming
```bash
make run-client
# or:
cd client && go run . --language English

# List available mic devices
cd client && go run . --list-devices

# Full options
cd client && go run . --host localhost --port 50051 --language English --chunk-ms 500 --silence-ms 2000
```

**Output while speaking:**
```
[partial] "Hello, this is"
[partial] "Hello, this is a test"
[final]   lang="English"  text="Hello, this is a test transcription."
```

---

## Useful Commands

```bash
# View live logs
docker compose logs -f

# Stop
docker compose down

# Rebuild after editing app/server.py (fast — only last layer rebuilds)
docker compose up --build

# Override VRAM settings on the fly without editing .env
GPU_MEMORY_UTILIZATION=0.75 MAX_MODEL_LEN=2048 docker compose up
```

---

## Notes

- **Model weights** persist in `./models/` — not re-downloaded on container restart.
- **Compiled GPU kernels** (Triton/PyTorch) persist in Docker volumes (`triton_cache`, `torch_compile_cache`). First cold start: ~2–3 min. Restarts: ~30 sec.
- The `libcuda.so` symlink fix for Triton's CUDA compilation is baked into the `Dockerfile` — no manual host-level hacks needed.
- GPU memory is tunable via `.env` — lower `GPU_MEMORY_UTILIZATION` if the container crashes on startup with an OOM error.
