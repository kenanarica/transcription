#!/usr/bin/env python3
"""
Summarize a D&D session transcript using Ollama (gemma3:12b).

Usage:
    python summarize.py transcript.txt
    python summarize.py transcript.txt --url http://localhost:11434
    python summarize.py transcript.txt --vocab vocabulary.json
"""

import argparse
import json
import os
import sys
import urllib.request

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL       = "gemma3:12b"
VOCAB_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary.json")


def load_vocab(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt(vocab: list[str]) -> str:
    prompt = (
        "You summarize D&D session transcripts as a bullet-point list. "
        "Output ONLY the bullet points — no introduction, no conclusion, no commentary. "
        "Each bullet is one sentence. Past tense. "
        "Cover only what actually happened: events, decisions, NPCs met, locations, items, cliffhangers."
    )
    if vocab:
        terms = ", ".join(vocab)
        prompt += (
            f"\n\nThe transcript was produced by speech-to-text and may contain mishearings, "
            f"especially for proper nouns. When you encounter an unfamiliar or garbled word, "
            f"check whether it sounds like one of the known campaign terms below and treat it as that term. "
            f"For example, a word like 'quarsh' or 'cores' is likely 'Qours'. "
            f"Always use the correct spelling from this list in your summary:\n{terms}"
        )
    return prompt


def summarize(transcript: str, system: str, ollama_url: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "system": system,
        "prompt": f"Please summarize the following D&D session transcript:\n\n{transcript}",
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
        return data["response"].strip()


def main():
    parser = argparse.ArgumentParser(description="Summarize a D&D session transcript via Ollama.")
    parser.add_argument("transcript", help="Path to the transcript text file")
    parser.add_argument("--url",   default=OLLAMA_URL, help=f"Ollama base URL (default: {OLLAMA_URL})")
    parser.add_argument("--vocab", default=VOCAB_FILE,  help=f"Path to vocabulary.json (default: {VOCAB_FILE})")
    args = parser.parse_args()

    if not os.path.exists(args.transcript):
        print(f"Error: transcript file not found: {args.transcript}", file=sys.stderr)
        sys.exit(1)

    with open(args.transcript, encoding="utf-8") as f:
        transcript = f.read().strip()

    if not transcript:
        print("Error: transcript file is empty.", file=sys.stderr)
        sys.exit(1)

    vocab = load_vocab(args.vocab)
    system = build_system_prompt(vocab)

    print(f"[summarize] Model:      {MODEL}", flush=True)
    print(f"[summarize] Ollama URL: {args.url}", flush=True)
    print(f"[summarize] Vocab terms loaded: {len(vocab)}", flush=True)
    print(f"[summarize] Transcript length:  {len(transcript)} chars", flush=True)
    print("[summarize] Sending to Ollama...\n", flush=True)

    summary = summarize(transcript, system, args.url)
    print(summary)


if __name__ == "__main__":
    main()
