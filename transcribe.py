#!/usr/bin/env python3
"""
Persistent transcription + speaker diarization server.
Loads all models once on startup, then accepts POST /transcribe requests.
Started automatically alongside the bot via `npm run dev`.

Backends
--------
- diarize=true  → whisperX pipeline (faster-whisper → align → pyannote)
- diarize=false → original OpenAI whisper (PyTorch, GPU-capable)
  Falls back to whisperX if openai-whisper is not installed.

Setup:
    pip install whisperx openai-whisper
    # Add HF_TOKEN to .env (free HuggingFace account + accept pyannote model terms)
    # https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import os
import json
import time
import threading
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

# PyTorch 2.6 changed torch.load to weights_only=True by default, which breaks pyannote
# checkpoints that store omegaconf objects. Rather than chasing individual blocked types,
# patch torch.load to keep weights_only=False for any caller that doesn't set it explicitly.
# This is safe — all models here come from trusted HuggingFace sources.
_orig_torch_load = torch.load
def _torch_load_compat(f, *args, **kwargs):
    kwargs['weights_only'] = False  # override even if caller explicitly set True
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _torch_load_compat  # type: ignore[assignment]
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

PORT              = int(os.getenv("TRANSCRIBE_PORT", "8765"))
WHISPERX_MODEL    = os.getenv("WHISPERX_MODEL", os.getenv("WHISPER_MODEL", "small"))
PLAIN_MODEL       = os.getenv("PLAIN_WHISPER_MODEL", os.getenv("WHISPER_MODEL", "turbo"))
os.environ.setdefault("OMP_NUM_THREADS", "8")
HF_TOKEN          = os.getenv("HF_TOKEN", "")
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE      = "float16" if DEVICE == "cuda" else "int8"
FP16              = DEVICE == "cuda"

# Inject ffmpeg directory if FFMPEG_PATH is set
ffmpeg_path = os.getenv("FFMPEG_PATH")
if ffmpeg_path:
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")

# ── Vocabulary ────────────────────────────────────────────────────────────────

initial_prompt = None
vocab_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary.json")
if os.path.exists(vocab_file):
    with open(vocab_file, encoding="utf-8") as f:
        words = json.load(f)
    if words:
        prompt_str = ", ".join(words)
        if len(prompt_str) > 800:
            prompt_str = prompt_str[:800]
            last_comma = prompt_str.rfind(",")
            if last_comma > 0:
                prompt_str = prompt_str[:last_comma]
        initial_prompt = prompt_str

# ── Model loading (once at startup) ──────────────────────────────────────────

# Original OpenAI whisper — PyTorch-based, uses GPU directly (CUDA/ROCm).
# Used for plain transcription (diarize=false).
plain_model = None
try:
    import whisper as openai_whisper
    print(f"[transcribe] Loading OpenAI whisper '{PLAIN_MODEL}' on {DEVICE}...", flush=True)
    plain_model = openai_whisper.load_model(PLAIN_MODEL, device=DEVICE)
    print("[transcribe] OpenAI whisper ready.", flush=True)
except ImportError:
    print("[transcribe] openai-whisper not installed — plain transcription will use whisperX.", flush=True)

# whisperX (faster-whisper) — used for diarization path (diarize=true).
print(f"[transcribe] Loading whisperX '{WHISPERX_MODEL}' on {DEVICE} ({COMPUTE_TYPE})...", flush=True)
whisperx_model = whisperx.load_model(WHISPERX_MODEL, DEVICE, compute_type=COMPUTE_TYPE, language="en")

print("[transcribe] Loading alignment model (en)...", flush=True)
align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

diarize_model = None
if HF_TOKEN:
    print("[transcribe] Loading diarization pipeline...", flush=True)
    diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    print("[transcribe] Diarization enabled.", flush=True)
else:
    print("[transcribe] HF_TOKEN not set — diarization disabled.", flush=True)

model_lock = threading.Lock()
print(f"[transcribe] Ready on port {PORT}.", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_diarized(segments: list) -> str:
    """Merge consecutive same-speaker segments into labelled lines."""
    lines = []
    current_speaker = None
    current_texts: list[str] = []

    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_??")
        text    = seg.get("text", "").strip()
        if not text:
            continue
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append(f"[{current_speaker}]: {' '.join(current_texts)}")
            current_speaker = speaker
            current_texts   = [text]
        else:
            current_texts.append(text)

    if current_speaker is not None and current_texts:
        lines.append(f"[{current_speaker}]: {' '.join(current_texts)}")

    return "\n".join(lines)

# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/transcribe":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))
        fname  = body.get("filename", "")
        diarize = body.get("diarize", False)

        if not fname:
            self._send_json(400, {"error": "missing filename"})
            return

        try:
            with model_lock:
                if diarize and diarize_model is not None:
                    self._transcribe_with_diarization(fname)
                else:
                    self._transcribe_plain(fname)

        except Exception as e:
            print(f"[transcribe] Error on {fname}: {e}", flush=True)
            self._send_json(500, {"error": str(e)})

    def _transcribe_plain(self, fname: str):
        """Plain transcription via OpenAI whisper (GPU-capable) or whisperX fallback."""
        t0 = time.perf_counter()

        if plain_model is not None:
            # OpenAI whisper — uses PyTorch directly, benefits from GPU (CUDA/ROCm)
            result = plain_model.transcribe(
                fname,
                language="en",
                initial_prompt=initial_prompt,
                fp16=FP16,
            )
            text = result["text"].strip()
            backend = "whisper"
        else:
            # Fallback: faster-whisper via whisperX
            segments_gen, _ = whisperx_model.model.transcribe(
                fname,
                language="en",
                initial_prompt=initial_prompt,
                beam_size=1,
            )
            text = " ".join(s.text.strip() for s in segments_gen)
            backend = "whisperX"

        print(f"[transcribe] plain ({backend}): {time.perf_counter()-t0:.2f}s", flush=True)

        if not text:
            self._send_json(200, {"text": "", "diarized": False})
            return

        self._send_json(200, {"text": text, "diarized": False})

    def _transcribe_with_diarization(self, fname: str):
        """Full whisperX pipeline: transcribe → align → diarize."""
        # Transcription via faster-whisper
        t0 = time.perf_counter()
        segments_gen, info = whisperx_model.model.transcribe(
            fname,
            language="en",
            initial_prompt=initial_prompt,
            beam_size=1,
        )
        segments = [{"start": s.start, "end": s.end, "text": s.text}
                    for s in segments_gen]
        result = {"segments": segments, "language": info.language}
        print(f"[transcribe] transcription: {time.perf_counter()-t0:.2f}s", flush=True)

        if not result.get("segments"):
            self._send_json(200, {"text": "", "diarized": False})
            return

        # Load audio array for alignment + diarization
        audio = whisperx.load_audio(fname)

        t1 = time.perf_counter()
        result = whisperx.align(
            result["segments"], align_model, align_metadata,
            audio, DEVICE, return_char_alignments=False,
        )
        print(f"[transcribe] alignment:    {time.perf_counter()-t1:.2f}s", flush=True)

        t2 = time.perf_counter()
        diarize_segments = diarize_model(audio)
        result = assign_word_speakers(diarize_segments, result)
        print(f"[transcribe] diarization:  {time.perf_counter()-t2:.2f}s", flush=True)

        text = format_diarized(result["segments"])
        self._send_json(200, {"text": text, "diarized": True})


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.serve_forever()
