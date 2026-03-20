# Transcription Server

Python HTTP server that loads Whisper models once at startup and accepts transcription requests.

## Starting the server

```bash
python transcribe.py
```

Listens on `127.0.0.1:8765` by default.

**Environment variables:**
- `TRANSCRIBE_PORT` — port (default: `8765`)
- `WHISPERX_MODEL` or `WHISPER_MODEL` — WhisperX model size (default: `small`)
- `HF_TOKEN` — HuggingFace token, required for diarization

## POST /transcribe

**Plain transcription:**
```bash
curl -X POST http://127.0.0.1:8765/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"filename": "/path/to/audio.wav"}'
```
```json
{"text": "Hello everyone welcome to the session.", "diarized": false}
```

**With speaker diarization:**
```bash
curl -X POST http://127.0.0.1:8765/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"filename": "/path/to/audio.wav", "diarize": true}'
```
```json
{"text": "[SPEAKER_00]: Hello everyone.\n[SPEAKER_01]: Let's begin.", "diarized": true}
```

**Error response:**
```json
{"error": "file not found"}
```

The server uses OpenAI Whisper (GPU) with WhisperX as fallback. If `vocabulary.json` exists in the project root, its terms are fed as an initial prompt to improve proper noun recognition.

---

## Summarizer

Summarizes a transcript using Ollama (gemma3:12b). Outputs a bullet-point D&D session summary.

```bash
python summarize.py session.txt
python summarize.py session.txt --url http://localhost:11434 --vocab vocabulary.json
```

If `vocabulary.json` is provided, the LLM is guided to correct misspelled proper nouns in the summary.

---

## Benchmark

Tests the transcription server against a suite of audio files.

```bash
python benchmark.py              # run all files in ./benchmark/
python benchmark.py audio.wav    # run a single file
```

Results are printed as a table and saved to `benchmark_results/`.

```
File         Size    Mode           Time   Preview
hqyok.wav    0.7MB   plain          4.32s  Hello everyone...
hqyok.wav    0.7MB   diarization    6.18s  [SPEAKER_00]: Hello...
```

---

## Vocabulary

`vocabulary.json` is a list of proper nouns and campaign-specific terms. It is:
- Loaded by the transcription server as a Whisper initial prompt
- Used by the summarizer to fix STT mishearings
- Auto-updated by the Discord bot from the `#wiki` channel via Ollama extraction

---

## Discord Bot

```bash
npm run dev   # starts both the TypeScript bot and transcribe.py
```

**Commands:**

| Command | Description |
|---|---|
| `!join` | Join your voice channel |
| `!leave` | Leave and clean up |
| `!listen` | Record all users, flush to transcription every 60s |
| `!listen nodiarize` | Record without speaker labels |
| `!listen nodiarize 30` | Record without diarization, flush every 30s |
| `!stop` | Flush remaining audio and stop |

Transcripts are posted to `#session-transcripts`. Vocabulary is rebuilt from `#wiki` on startup.
