"""Transcribe the full scored dev set with faster-whisper, resumably.

Caches per part to data/derived/transcripts_{part}_{model}.json — the same files the
notebooks read. Already-transcribed utterances are skipped, so the script can be
interrupted and re-run; it only ever does the remaining work.

Usage:
    python scripts/transcribe_dev.py                 # all scored parts, P3/P4 first
    python scripts/transcribe_dev.py --parts P3 P4   # subset
"""

import argparse
from pathlib import Path

from speechlab.asr import load_or_transcribe
from speechlab.data import load_responses

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "sandi-corpus-2025"
DERIVED = ROOT / "data" / "derived"

# Long single turns (one score per file) first — they unblock notebook 03;
# the many short P1/P5 responses follow.
DEFAULT_PARTS = ["P3", "P4", "P5", "P1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", nargs="+", default=DEFAULT_PARTS, choices=["P1", "P3", "P4", "P5"])
    ap.add_argument("--model", default="small")
    args = ap.parse_args()

    DERIVED.mkdir(exist_ok=True)
    for part in args.parts:
        df = load_responses(CORPUS, part)
        items = {r.utt: CORPUS / r.flac for r in df.itertuples()}
        cache_path = DERIVED / f"transcripts_{part}_{args.model}.json"
        print(f"== {part}: {len(items)} responses -> {cache_path.name}", flush=True)
        cache = load_or_transcribe(cache_path, items, model_name=args.model)
        print(f"== {part} done: {len(cache)} cached", flush=True)


if __name__ == "__main__":
    main()
