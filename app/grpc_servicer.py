"""
gRPC ASR Servicer — bidirectional streaming transcription.
Runs alongside the FastAPI HTTP server, sharing the same `asr` model instance.
"""
import io
import logging
import numpy as np
import soundfile as sf
import grpc

import app.proto.asr_pb2 as asr_pb2
import app.proto.asr_pb2_grpc as asr_pb2_grpc

logger = logging.getLogger(__name__)


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample audio to 16 kHz using linear interpolation."""
    if sr == 16000:
        return wav.astype(np.float32, copy=False)
    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


class ASRServicer(asr_pb2_grpc.ASRServiceServicer):
    """
    gRPC servicer for bidirectional streaming ASR.

    The client streams AudioChunk messages containing raw PCM float32 audio.
    After each chunk the server responds with a partial TranscriptChunk.
    When the client sends is_final=True, the server flushes the streaming
    state and returns the consolidated final transcript.
    """

    def __init__(self, asr_model):
        self.asr = asr_model

    def TranscribeStream(self, request_iterator, context):
        """
        Bidirectional streaming RPC.
        Reads AudioChunk messages, runs streaming inference, yields TranscriptChunk responses.
        """
        state = None
        language = None

        try:
            for chunk in request_iterator:
                # Initialize state on first chunk
                if state is None:
                    language = chunk.language or None
                    state = self.asr.init_streaming_state(
                        unfixed_chunk_num=2,
                        unfixed_token_num=5,
                        chunk_size_sec=2.0,
                    )

                # Decode raw PCM bytes → numpy float32 array
                # Client sends float32 LE bytes at the declared sample_rate
                wav = np.frombuffer(chunk.audio_data, dtype=np.float32)

                # Resample to 16 kHz if needed
                sr = chunk.sample_rate if chunk.sample_rate > 0 else 16000
                if sr != 16000:
                    wav = _resample_to_16k(wav, sr)

                # Feed to streaming model
                self.asr.streaming_transcribe(wav, state)

                # Yield partial result after each chunk
                yield asr_pb2.TranscriptChunk(
                    text=state.text or "",
                    language=state.language or "",
                    is_final=False,
                )

                # If client signals end of stream, flush and finalize
                if chunk.is_final:
                    break

            # Flush streaming state to get the final consolidated transcript
            if state is not None:
                self.asr.finish_streaming_transcribe(state)
                yield asr_pb2.TranscriptChunk(
                    text=state.text or "",
                    language=state.language or "",
                    is_final=True,
                )

        except Exception as e:
            logger.exception("Error in TranscribeStream")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return
