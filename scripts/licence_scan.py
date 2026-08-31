"""Pre-push licence scan: no corpus content in anything that reaches git.

The corpus is research-licensed and the repository is public, so three things must never
appear in a committed file: **utterance ids**, **speaker ids**, and **verbatim runs of
transcript text**. Notebooks 01–05 are committed *with* outputs, so the convention is not
"no outputs" but "no corpus content in the outputs", and that is a property of the output
text rather than of the code — it has to be measured against the corpus itself, not
reasoned about.

Three checks, each against the real corpus rather than a pattern:

1. **Utterance ids** — every `utt` in the dev flists, matched as a literal substring.
2. **Speaker ids** — likewise. Checked separately because a speaker id is a prefix of its
   utterance ids, so an utterance-id leak is also a speaker-id leak and the counts should
   be read together.
3. **Verbatim five-word runs** — every 5-gram of every cached Whisper transcript, matched
   against the target's own 5-grams after the same normalisation (lowercase, letters and
   apostrophes only). Five is the threshold the project has used since notebook 05: short
   enough to catch a quoted fragment, long enough that ordinary English prose does not
   collide with spontaneous speech by accident.

**A 5-gram hit is a signal to inspect, not proof of a leak.** Learner speech is ordinary
English, so a sufficiently plain sentence in a report could in principle collide with
something a candidate said. Every hit is printed in full so it can be judged.

Requires the corpus locally, so this is a pre-push check and not something CI can run.

Usage:
    python scripts/licence_scan.py                 # git-tracked files in the worktree
    python scripts/licence_scan.py --history       # every blob reachable from any ref
    python scripts/licence_scan.py notebooks/05_finetuned_scorer.ipynb reports/
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from speechlab.data import load_responses

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "sandi-corpus-2025"
DERIVED = ROOT / "data" / "derived"
PARTS = ["P1", "P3", "P4", "P5"]
NGRAM = 5
# Five-word runs adjudicated as coincidence, with the reason each was cleared. Learner
# speech is ordinary English, so a plain enough sentence in the project's own prose will
# occasionally collide with something a candidate said; the collision is the scan working
# rather than failing. Entries apply to the n-gram check only. **An id hit is never
# adjudicated** — there is no innocent reason for a real utterance or speaker id to appear
# in a committed file, so ids have no allowlist and cannot be suppressed here.
ADJUDICATED = {
    "this is the most important":
        "notebook 01's own editorial aside, '*This is the most important comparison in "
        "the notebook.*' — the notebook's voice, not a candidate's",
}

# Text formats that can carry corpus content. Binary and data files are excluded because
# `data/` is gitignored from commit zero — if one appears in git at all, that is a
# different and larger problem than this script is checking for.
SUFFIXES = {".ipynb", ".md", ".py", ".txt", ".toml", ".cfg", ".yml", ".yaml", ".json"}


def corpus_index() -> tuple[set[str], set[str], set[str]]:
    """Utterance ids, speaker ids, and transcript 5-grams — the things to search for."""
    utts, speakers = set(), set()
    for part in PARTS:
        try:
            df = load_responses(CORPUS, part)
        except FileNotFoundError:
            continue                                  # not every part is transcribed yet
        utts |= set(df.utt)
        speakers |= set(df.speaker)

    grams = set()
    for cache in sorted(DERIVED.glob("transcripts_*.json")):
        for words in json.loads(cache.read_text()).values():
            toks = [t for t in (normalise(w["w"]) for w in words) if t]
            grams |= {" ".join(toks[i:i + NGRAM]) for i in range(len(toks) - NGRAM + 1)}
    return utts, speakers, grams


def normalise(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def ngrams(text: str) -> set[str]:
    toks = [t for t in (normalise(w) for w in text.split()) if t]
    return {" ".join(toks[i:i + NGRAM]) for i in range(len(toks) - NGRAM + 1)}


def visible_text(path: str, blob: str) -> str:
    """The reader-facing text of a file — for a notebook, source *and* outputs.

    An early version scanned notebook outputs only, on the reasoning that `utt` and
    `speaker` are used freely inside cell source. That was a mistake and it cost a real
    finding: those are *column names*, whereas this scan looks for literal id values,
    which have no legitimate reason to appear in source either. Scanning source first
    caught an example utterance id sitting in a comment in `src/speechlab/data.py` and
    in two cells of notebook 01, committed since the `src/` refactor. Markdown cells are
    rendered prose in any case, so excluding them was never defensible.

    JSON is flattened rather than parsed structurally: the goal is to look at every
    string a reader could see, and a notebook that fails to parse should be scanned
    whole rather than skipped.
    """
    if not path.endswith(".ipynb"):
        return blob
    try:
        nb = json.loads(blob)
    except json.JSONDecodeError:
        return blob                                   # malformed: scan it whole, loudly
    out = []
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        out.append("".join(source) if isinstance(source, list) else str(source))
        for o in cell.get("outputs", []):
            text = o.get("text") or o.get("data", {}).get("text/plain", "")
            out.append("".join(text) if isinstance(text, list) else str(text))
    return "\n".join(out)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=True).stdout


def worktree_targets(paths: list[str]) -> list[tuple[str, str]]:
    tracked = git("ls-files", *paths).split()
    return [(p, (ROOT / p).read_text(errors="replace"))
            for p in tracked if Path(p).suffix in SUFFIXES and (ROOT / p).exists()]


def history_targets() -> list[tuple[str, str]]:
    """Every blob reachable from any ref, labelled by the path it was stored under.

    Deduplicated by object id, so a notebook committed twenty times is scanned once per
    distinct content rather than once per commit.
    """
    seen, out = set(), []
    for line in git("rev-list", "--objects", "--all").splitlines():
        oid, _, path = line.partition(" ")
        if not path or oid in seen or Path(path).suffix not in SUFFIXES:
            continue
        seen.add(oid)
        out.append((f"{path} @ {oid[:8]}", git("cat-file", "-p", oid)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[], help="limit to these paths")
    ap.add_argument("--history", action="store_true",
                    help="scan every blob in the object graph, not just the worktree")
    ap.add_argument("--quiet", action="store_true", help="print only failures")
    args = ap.parse_args()

    assert CORPUS.exists(), f"corpus not found at {CORPUS} — this check needs it locally"
    utts, speakers, grams = corpus_index()
    print(f"corpus: {len(utts)} utterance ids, {len(speakers)} speaker ids, "
          f"{len(grams)} distinct {NGRAM}-grams")

    targets = history_targets() if args.history else worktree_targets(args.paths)
    print(f"scanning {len(targets)} {'blobs' if args.history else 'tracked files'}\n")

    failures = 0
    for name, blob in targets:
        text = visible_text(name.split(" @ ")[0], blob)
        hits = {
            "utterance id": sorted(u for u in utts if u in text),
            "speaker id": sorted(s for s in speakers if s in text),
            f"verbatim {NGRAM}-word run": sorted((ngrams(text) & grams) - set(ADJUDICATED)),
        }
        cleared = sorted(ngrams(text) & grams & set(ADJUDICATED))
        n = sum(len(v) for v in hits.values())
        if n:
            failures += n
            print(f"FAIL {name}")
            for kind, found in hits.items():
                for h in found:
                    print(f"       {kind}: {h!r}")
        elif not args.quiet:
            note = f"  ({len(cleared)} adjudicated)" if cleared else ""
            print(f"ok   {name}{note}")

    print(f"\n{'CLEAN' if not failures else f'*** {failures} HITS — do not push ***'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
