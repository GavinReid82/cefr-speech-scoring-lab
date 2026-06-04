"""Download and unzip the Speak & Improve Corpus 2025 dev-set audio.

Audio is distributed separately from the base corpus package (see
data/sandi-corpus-2025/README-install-dataset.txt). This script fetches the
two dev zips (~2 GB total) and unzips them so flac files land in
data/sandi-corpus-2025/data/flac/dev/, the layout the corpus tooling expects.

Usage: python3 scripts/download_dev_audio.py
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "data" / "sandi-corpus-2025"
STATUS_FILE = CORPUS_ROOT / ".download-status.json"
BASE_URL = "https://speak-and-improve-corpus-2025.s3.eu-west-1.amazonaws.com/audio/"
DEV_ZIPS = ["data.flac.dev.01.zip", "data.flac.dev.02.zip"]
CHUNK = 1 << 20  # 1 MiB


def set_status(status: dict) -> None:
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def download(name: str, status: dict) -> None:
    dest = CORPUS_ROOT / name
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        status["files"][name] = "already-present"
        set_status(status)
        return
    with urllib.request.urlopen(BASE_URL + name) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        last_pct = -10
        with open(part, "wb") as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                pct = int(done / total * 100) if total else 0
                if pct >= last_pct + 5:
                    last_pct = pct
                    status["files"][name] = f"downloading {pct}%"
                    set_status(status)
    part.rename(dest)
    status["files"][name] = "downloaded"
    set_status(status)


def main() -> int:
    status = {"state": "downloading", "files": {}, "error": None}
    set_status(status)
    try:
        for name in DEV_ZIPS:
            download(name, status)
        status["state"] = "unzipping"
        set_status(status)
        for name in DEV_ZIPS:
            subprocess.run(
                ["unzip", "-q", "-o", str(CORPUS_ROOT / name), "-d", str(CORPUS_ROOT)],
                check=True,
                capture_output=True,
            )
            status["files"][name] = "unzipped"
            set_status(status)
        flac_count = sum(1 for _ in (CORPUS_ROOT / "data" / "flac" / "dev").rglob("*.flac"))
        status["state"] = "done"
        status["flac_count"] = flac_count
        set_status(status)
        print(f"DONE: {flac_count} flac files in data/flac/dev")
        return 0
    except Exception as exc:  # noqa: BLE001 — report any failure into the status file
        status["state"] = "error"
        status["error"] = str(exc)
        set_status(status)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
