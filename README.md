# Qwen3-ASR — Dockerized Speech-to-Text API with gRPC Streaming

Real-time speech transcription powered by [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) running on vLLM.
Exposes both an **HTTP REST API** (file upload) and a **bidirectional gRPC streaming API** (live microphone).

## Project Structure

```
├── Dockerfile              # CUDA 12.8 image — grpcio install + proto compile baked in
├── docker-compose.yml      # GPU passthrough, ports 8000 (HTTP) + 50051 (gRPC)
├── .env.example            # Copy to .env and tune for your GPU
├── proto/
│   └── asr.proto           # gRPC service definition (shared by server + client)
├── app/
│   ├── server.py           # FastAPI (HTTP) + gRPC server, runs both concurrently
│   ├── grpc_servicer.py    # Bidirectional streaming ASR servicer
│   └── proto/              # Python stubs (auto-generated during Docker build)
├── client/
│   ├── main.go             # Go gRPC client: mic streaming + file test (uses ffmpeg)
│   ├── go.mod
│   └── proto/              # Go stubs (generated via `make proto`)
└── Makefile                # Shortcuts: make proto, make test-client, make run-client
```

---

## Server Setup (Docker)

### Prerequisites

| | Windows + Docker Desktop | Native Ubuntu/Linux |
|---|---|---|
| Docker | Docker Desktop (WSL2 backend) | `curl -fsSL https://get.docker.com \| sh` |
| NVIDIA Toolkit | Install inside WSL terminal | Install directly on host |
| WSL required? | Yes — NVIDIA toolkit must be installed in WSL | No |

> **Windows tip:** Use **Ubuntu-24.04** as your primary WSL distro. Install the NVIDIA toolkit inside it, not in another distro.

#### Install NVIDIA Container Toolkit (inside WSL / Ubuntu):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
# Windows: restart Docker Desktop from system tray
# Linux:   sudo systemctl restart docker
```

Verify GPU passthrough:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

### Configure & Start

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / WSL
cp .env.example .env
```

Edit `.env` for your GPU VRAM:

| GPU VRAM | `GPU_MEMORY_UTILIZATION` | `MAX_MODEL_LEN` |
|---|---|---|
| 6 GB (RTX 3060) | `0.80` | `2048` |
| 8 GB (RTX 3070) | `0.85` | `4096` |
| 12 GB (RTX 3080 Ti) | `0.90` | `8192` |
| 16 GB+ | `0.90` | `16384` |

```bash
# Always run from the PROJECT ROOT, not a subdirectory
docker compose build   # ~5-15 min on first run (downloads CUDA base + deps)
docker compose up
```

> ⚠️ **Always run `docker compose` from the project root.** Running from a subdirectory (e.g. `client/`) causes Docker credential errors even if the compose file is found.

Model weights (~4 GB) download to `./models/` on first start. Subsequent starts: ~30 sec.

Wait for both lines in the logs:
```
[startup] gRPC server listening on 0.0.0.0:50051
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Check health:
```bash
curl http://localhost:8088/health
# {"status":"ok","model":"Qwen/Qwen3-ASR-1.7B","ready":true,"grpc_port":50051}
```

---

## HTTP API Usage

### `POST /transcribe`

```bash
# Basic
curl.exe -X POST http://localhost:8088/transcribe -F "file=@audio.wav"

# With language hint
curl.exe -X POST "http://localhost:8088/transcribe?language=English" -F "file=@audio.wav"
```

**Response:**
```json
{"language": "English", "text": "Hello, this is a test transcription."}
```

> **Windows PowerShell:** use `curl.exe` (not `curl`, which is an alias for `Invoke-WebRequest`).

### `GET /health`
Returns `{"status":"ok","model":"...","ready":true,"grpc_port":50051}` when ready.

### Swagger UI
Open **http://localhost:8088/docs** for an interactive API explorer.

---

## gRPC Streaming Client (Go) — Live Microphone

For real-time, low-latency transcription from a microphone.

### Prerequisites

**1. Go for Windows** — https://go.dev/dl/

**2. FFmpeg** (handles mic capture — no native audio library needed):
```powershell
winget install Gyan.FFmpeg
```
> ⚠️ After `winget install`, the PATH is updated but **not in the current terminal session**.
> Run this to reload PATH without restarting:
> ```powershell
> $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
> ```
> Or use `refreshenv` if you have Chocolatey. Verify with `ffmpeg -version`.

**3. Generate Go proto stubs** (one-time, in Ubuntu-24.04 WSL):
```bash
# Inside Ubuntu-24.04 WSL terminal
sudo apt-get install -y golang-go protobuf-compiler

go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
export PATH=$PATH:$(go env GOPATH)/bin   # add to ~/.bashrc to make permanent

cd /mnt/g/Code/vllm-qwen   # adjust path to your project
mkdir -p client/proto
protoc \
  --go_out=client/proto \
  --go_opt=paths=source_relative \
  --go-grpc_out=client/proto \
  --go-grpc_opt=paths=source_relative \
  -I proto \
  proto/asr.proto
```

> ⚠️ `go install` downloads packages silently — it may take 1-2 minutes with no output. Wait for the prompt to return.
> After install, run `export PATH=$PATH:$(go env GOPATH)/bin` before running `protoc`, otherwise it won't find `protoc-gen-go`.

**4. Download Go dependencies** (PowerShell, from `client/` directory):
```powershell
cd G:\Code\vllm-qwen\client
go mod tidy
```

---

### Find Your Microphone Device Name

Run this in PowerShell (ffmpeg must be in PATH):
```powershell
ffmpeg -list_devices true -f dshow -i dummy 2>&1
```

Look for the `(audio)` entries, e.g.:
```
"Microphone (Parsec Virtual Audio)" (audio)
  Alternative name "@device_cm_{33D9A762-...}\wave_{19AAA777-...}"
```

Use either the friendly name or the alternative name (more reliable).

---

### Run the Client

**Test with a file first** (confirms gRPC connection):
```powershell
cd G:\Code\vllm-qwen\client
go run . --file ..\test_en.wav --language English
```

**Live microphone streaming:**
```powershell
# Using friendly name
go run . --device "Microphone (Parsec Virtual Audio)" --language English

# Using hardware ID (most reliable, no name-matching issues)
go run . --device "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{19AAA777-1EFA-4C87-856E-F629322190AC}" --language English
```

**All flags:**
```
--host        gRPC server host (default: localhost)
--port        gRPC server port (default: 50051)
--language    Language hint: "English", "Chinese", etc. (default: auto-detect)
--chunk-ms    Audio chunk size in ms (default: 500; lower = faster partials)
--device      Audio input device name (from --list-devices)
--file        Path to audio file to stream instead of microphone
--list-devices  List available audio input devices and exit
```

**Expected output while speaking:**
```
[partial] "Hello"
[partial] "Hello this is a test"
[final]   lang="English"  text="Hello this is a test transcription."
```

Press `Ctrl+C` to stop.

---

## Useful Commands

```bash
# View live server logs
docker compose logs -f

# Stop server
docker compose down

# Rebuild after editing app/ (fast — only the COPY layer rebuilds)
docker compose up --build

# Override VRAM without editing .env
GPU_MEMORY_UTILIZATION=0.75 MAX_MODEL_LEN=2048 docker compose up
```

---

## Gotchas & Known Issues

| Issue | Cause | Fix |
|---|---|---|
| `curl: Empty reply from server` | Model still loading | Wait ~60 sec after startup for model to initialise |
| `ffmpeg: not found` after winget | PATH not refreshed in current session | Run `refreshenv` or the `$env:PATH = ...` command above |
| `no default input device` in WSL | WSL has no access to Windows audio hardware | Run the Go client from **Windows PowerShell**, not WSL |
| Docker build fails with credential error | Running `docker compose` from a subdirectory | Always run from the **project root** |
| `ModuleNotFoundError: No module named 'app'` | Old Docker image | Rebuild with `docker compose build` |
| gRPC stream exits instantly | Wrong device name | Use `ffmpeg -list_devices true -f dshow -i dummy 2>&1` to get exact name |
| Transcription in wrong language | Auto-detect can be wrong for short clips | Always pass `--language English` (or the correct language) |
| Proto stubs missing after clone | Stubs are generated, not committed | Run the `protoc` command in the Prerequisites section |

---

## Notes

- **Model weights** persist in `./models/` — not re-downloaded on container restart.
- **GPU kernel cache** (Triton/PyTorch) persists in Docker volumes. Cold start: ~2-3 min. Warm restart: ~30 sec.
- The `libcuda.so` symlink fix for Triton is baked into the `Dockerfile`.
- HTTP endpoint runs on internal port `8000` (mapped to `8088` by default in `.env`).
- gRPC endpoint runs on port `50051` (mapped 1:1).
