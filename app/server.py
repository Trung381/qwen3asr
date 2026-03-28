import io
import os
import numpy as np
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from qwen_asr import Qwen3ASRModel
import uvicorn

# --- Config from environment ---
MODEL_NAME             = os.getenv("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
MAX_MODEL_LEN          = int(os.getenv("MAX_MODEL_LEN", "4096"))
HOST                   = os.getenv("HOST", "0.0.0.0")
PORT                   = int(os.getenv("PORT", "8000"))

# Global model reference — populated in lifespan
asr = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Model is loaded HERE (not at module level) so that when vLLM spawns child
    processes via Python 'spawn' multiprocessing, re-importing this module does
    NOT trigger another model load and cause the 'bootstrapping phase' RuntimeError.
    """
    global asr
    print(f"[startup] Loading model: {MODEL_NAME}")
    print(f"[startup] GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    print(f"[startup] Max model len: {MAX_MODEL_LEN}")

    asr = Qwen3ASRModel.LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        disable_log_stats=True,
    )
    print("[startup] Model loaded. Server ready.")
    yield
    # Cleanup on shutdown (if needed)
    asr = None


app = FastAPI(
    title="Qwen3-ASR API",
    description="Speech-to-text API powered by Qwen3-ASR and vLLM",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Health check — use this to verify the container is ready."""
    return {"status": "ok", "model": MODEL_NAME, "ready": asr is not None}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, etc.)"),
    language: str | None = None,
):
    """
    Transcribe an uploaded audio file.

    - **file**: Audio file to transcribe (WAV recommended)
    - **language**: Optional language hint (e.g. 'English', 'Chinese'). Leave empty for auto-detect.

    Returns: `{ "language": "...", "text": "..." }`
    """
    if asr is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        audio_bytes = await file.read()
        # Decode audio bytes to (np.ndarray, sample_rate) — the format qwen_asr accepts
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
        results = asr.transcribe(audio=(wav, sr), language=language)
        result = results[0]
        return {"language": result.language, "text": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
