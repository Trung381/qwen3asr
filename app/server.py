import io
import os
import threading
import logging
import numpy as np
import soundfile as sf
import grpc
from concurrent import futures
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from qwen_asr import Qwen3ASRModel
import uvicorn

import app.proto.asr_pb2_grpc as asr_pb2_grpc
from app.grpc_servicer import ASRServicer

logger = logging.getLogger(__name__)

# --- Config from environment ---
MODEL_NAME             = os.getenv("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
MAX_MODEL_LEN          = int(os.getenv("MAX_MODEL_LEN", "4096"))
HOST                   = os.getenv("HOST", "0.0.0.0")
PORT                   = int(os.getenv("PORT", "8000"))
GRPC_PORT              = int(os.getenv("GRPC_PORT", "50051"))

# Global references — populated in lifespan
asr = None
_grpc_server = None


def _start_grpc_server(asr_model):
    """Start the gRPC server in a daemon thread."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    asr_pb2_grpc.add_ASRServiceServicer_to_server(ASRServicer(asr_model), server)
    listen_addr = f"{HOST}:{GRPC_PORT}"
    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"[startup] gRPC server listening on {listen_addr}")
    print(f"[startup] gRPC server listening on {listen_addr}")
    return server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Model is loaded HERE (not at module level) so that when vLLM spawns child
    processes via Python 'spawn' multiprocessing, re-importing this module does
    NOT trigger another model load and cause the 'bootstrapping phase' RuntimeError.
    """
    global asr, _grpc_server

    print(f"[startup] Loading model: {MODEL_NAME}")
    print(f"[startup] GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    print(f"[startup] Max model len: {MAX_MODEL_LEN}")

    asr = Qwen3ASRModel.LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        disable_log_stats=True,
    )
    print("[startup] Model loaded.")

    # Start gRPC server alongside uvicorn
    _grpc_server = _start_grpc_server(asr)
    print(f"[startup] HTTP server on port {PORT}, gRPC on port {GRPC_PORT}. Ready!")

    yield

    # Graceful shutdown
    if _grpc_server:
        _grpc_server.stop(grace=5)
    asr = None


app = FastAPI(
    title="Qwen3-ASR API",
    description="Speech-to-text API powered by Qwen3-ASR and vLLM. HTTP + gRPC streaming supported.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Health check — use this to verify the container is ready."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "ready": asr is not None,
        "http_port": PORT,
        "grpc_port": GRPC_PORT,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, etc.)"),
    language: str | None = None,
):
    """
    Transcribe an uploaded audio file (non-streaming HTTP endpoint).

    - **file**: Audio file to transcribe (WAV recommended)
    - **language**: Optional language hint (e.g. 'English', 'Chinese'). Leave empty for auto-detect.

    For real-time streaming, use the gRPC `TranscribeStream` RPC instead.
    """
    if asr is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        audio_bytes = await file.read()
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
        results = asr.transcribe(audio=(wav, sr), language=language)
        result = results[0]
        return {"language": result.language, "text": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
