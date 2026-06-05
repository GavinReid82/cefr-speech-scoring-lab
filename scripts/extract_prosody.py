"""Extract prosody features for the scored dev set, resumably.

Caches per part to data/derived/prosody_{part}_full.json. Re-running skips
already-extracted utterances.

Usage:
    python scripts/extract_prosody.py --parts P3 P4
"""

import argparse
import json
from pathlib import Path

from speechlab.acoustic_features import prosody_features
from speechlab.data import load_responses

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "sandi-corpus-2025"
DERIVED = ROOT / "data" / "derived"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", nargs="+", default=["P3", "P4"], choices=["P1", "P3", "P4", "P5"])
    args = ap.parse_args()

    for part in args.parts:
        df = load_responses(CORPUS, part)
        cache_path = DERIVED / f"prosody_{part}_full.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        todo = [r for r in df.itertuples() if r.utt not in cache]
        print(f"== {part}: {len(todo)} of {len(df)} to extract", flush=True)
        for i, r in enumerate(todo, 1):
            cache[r.utt] = prosody_features(CORPUS / r.flac)
            if i % 50 == 0 or i == len(todo):
                cache_path.write_text(json.dumps(cache))
                print(f"{i}/{len(todo)} extracted", flush=True)
        print(f"== {part} done: {len(cache)} cached", flush=True)


if __name__ == "__main__":
    main()
