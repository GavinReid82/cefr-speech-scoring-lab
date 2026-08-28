"""Fine-tune a LoRA band scorer per CV fold and collect out-of-fold predictions.

The fifth scoring arm. Every other arm in this repo is content-blind — fluency,
prosody, word counts — so `reports/evaluation_report.md` closes by noting that its
ASR-propagation null is *conditional* on that blindness. This script produces the
predictions that let notebook 05 test the same question with a scorer that reads what
the candidate actually said.

Protocol is deliberately unchanged from notebook 03: speaker-grouped `GroupKFold(5)`
over the same 876 long-turn dev responses, scored by `speechlab.evaluation`. A fold's
model never sees any response from a test-fold speaker, including that speaker's other
test part.

Caches out-of-fold predictions to data/derived/lora_oof_{tag}.json and skips folds that
are already in the cache, so an interrupted run resumes where it stopped — the same
contract as transcribe_dev.py. Training is delegated to the `mlx_lm lora` CLI in a
subprocess per fold: its trainer is not worth reimplementing, and process isolation
sidesteps mlx-lm's per-process LoRA state (`linear_to_lora_layers` raises on an
already-converted layer, and its RNG seeding is global).

Usage:
    python scripts/finetune_lora.py                        # Whisper transcripts, 5 folds
    python scripts/finetune_lora.py --transcripts fluent   # gold-transcript ablation
    python scripts/finetune_lora.py --folds 0 --iters 100  # one short fold, for timing
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from speechlab.asr import parse_stm
from speechlab.data import load_responses
from speechlab.llm_scorer import (
    DEFAULT_MODEL,
    LoRABandScorer,
    assert_prompt_parity,
    build_prompt,
    load_prompt_tokenizer,
    transcript_text,
    write_fold_data,
)
from speechlab.ordinal import BandCodec

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "sandi-corpus-2025"
DERIVED = ROOT / "data" / "derived"
WORK = DERIVED / "lora"

# The long single-turn parts: one human score per response, which is what makes them
# usable as supervision. P1/P5 are many short responses sharing a score.
PARTS = ["P3", "P4"]


def load_scored_transcripts(source: str, whisper_model: str) -> pd.DataFrame:
    """One row per scored response: utt, speaker, part, score, transcript.

    `source="whisper"` reads the ASR caches (all 876 responses); the gold sources read
    reference-materials/stms/dev-{asr,fluent,gec}.stm, which cover ~68% of them. The
    gap is the point of the ablation, not an accident — but it means gold-source runs
    are NOT comparable to the published RF QWK 0.558 without re-running RF on the same
    subset. Notebook 05 does that comparison on matched rows.
    """
    frames = []
    for part in PARTS:
        df = load_responses(CORPUS, part)
        df["part"] = part
        if source == "whisper":
            cache_path = DERIVED / f"transcripts_{part}_{whisper_model}.json"
            if not cache_path.exists():
                raise SystemExit(
                    f"missing {cache_path.name} — run:\n"
                    f"  python scripts/download_dev_audio.py\n"
                    f"  python scripts/transcribe_dev.py --parts {' '.join(PARTS)}"
                )
            cache = json.loads(cache_path.read_text())
            # absent from the cache = not yet transcribed (drop); present but empty =
            # Whisper genuinely heard nothing (keep — silence is evidence about the
            # candidate, and the prompt builder gives it an explicit placeholder)
            df = df[df.utt.isin(cache)].copy()
            df["transcript"] = [transcript_text(cache[u]) for u in df.utt]
        else:
            stm = parse_stm(CORPUS / "reference-materials" / "stms" / f"dev-{source}.stm")
            df = df[df.utt.isin(stm)].copy()
            df["transcript"] = [" ".join(stm[u]) for u in df.utt]
        frames.append(df[["utt", "speaker", "part", "score", "transcript"]])
    return pd.concat(frames, ignore_index=True)


def fold_config(data_dir: Path, adapter_dir: Path, args, seed: int) -> dict:
    """The YAML mlx-lm reads. Rank/scale/dropout have no CLI flags, and a *partial*
    `lora_parameters` dict raises KeyError rather than filling in defaults — so all
    three are always written out."""
    return {
        "model": args.model,
        "train": True,
        "data": str(data_dir),
        "adapter_path": str(adapter_dir),      # per fold: mlx-lm overwrites
        "fine_tune_type": "lora",              # adapter_config.json unconditionally
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "steps_per_report": 25,
        "steps_per_eval": max(args.iters // 4, 1),
        "save_every": args.iters,              # one checkpoint: the final adapter
        "seed": seed,                          # varied per fold — mlx/np seeding is
        "mask_prompt": True,                   # process-global, so a constant seed
        "lora_parameters": {                   # would give every fold identical init
            "rank": args.rank,                 # AND identical batch order
            "scale": args.scale,
            "dropout": args.dropout,
        },
    }


def train_fold(cfg: dict, config_path: Path) -> None:
    """Run one fold's training in a subprocess, failing loudly if it does."""
    import yaml

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)],
                   check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--transcripts", default="whisper", choices=["whisper", "asr", "fluent", "gec"],
                    help="whisper = ASR caches (all responses); the rest are gold STMs (~68%%)")
    ap.add_argument("--whisper-model", default="small", help="which transcript cache to read")
    ap.add_argument("--n-splits", type=int, default=5, help="must match notebook 03")
    ap.add_argument("--folds", type=int, nargs="+", help="subset of fold indices, for testing")
    ap.add_argument("--valid-frac", type=float, default=0.1,
                    help="speakers held out of each training fold to watch overfitting")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--num-layers", type=int, default=8, help="topmost layers to adapt; -1 = all")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=1.0, help="softmax dial for the decode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", help="cache name; defaults to <transcripts>_<model slug>")
    ap.add_argument("--force", action="store_true", help="retrain folds already cached")
    args = ap.parse_args()

    tag = args.tag or f"{args.transcripts}_{args.model.split('/')[-1]}"
    oof_path = DERIVED / f"lora_oof_{tag}.json"
    WORK.mkdir(parents=True, exist_ok=True)

    df = load_scored_transcripts(args.transcripts, args.whisper_model)
    codec = BandCodec.from_scores(df.score)
    print(f"{len(df)} responses / {df.speaker.nunique()} speakers / "
          f"{len(codec)} bands {codec.labels[0]}–{codec.labels[-1]} "
          f"({codec.lo}–{codec.hi})", flush=True)

    # Check prompt parity once, on the tokenizer, before spending hours on training:
    # a template that renders the training and inference paths differently costs QWK
    # silently, and this is the only place it can be caught cheaply.
    assert_prompt_parity(load_prompt_tokenizer(args.model), codec,
                         build_prompt(df.transcript.iloc[0], df.part.iloc[0], codec),
                         codec.encode([df.score.iloc[0]])[0])
    print("prompt parity OK", flush=True)

    cache = json.loads(oof_path.read_text()) if oof_path.exists() else {"meta": {}, "folds": {}}
    cache["meta"] = {k: v for k, v in vars(args).items() if k != "force"} | {
        "tag": tag, "n_responses": len(df), "bands": codec.labels,
        "lo": codec.lo, "hi": codec.hi,
    }

    splitter = GroupKFold(n_splits=args.n_splits)
    for k, (tr_idx, te_idx) in enumerate(splitter.split(df, groups=df.speaker)):
        if args.folds is not None and k not in args.folds:
            continue
        if str(k) in cache["folds"] and not args.force:
            print(f"== fold {k}: cached, skipping", flush=True)
            continue

        train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
        # hold out whole speakers, not rows: the two parts of one speaker are two
        # measurements of one proficiency, so splitting them would leak
        speakers = train_df.speaker.drop_duplicates().sample(frac=1, random_state=args.seed + k)
        n_val = max(int(len(speakers) * args.valid_frac), 1)
        val_mask = train_df.speaker.isin(set(speakers[:n_val]))

        rows = lambda d: list(zip(d.transcript, d.part, d.score))   # noqa: E731
        fold_dir = WORK / tag / f"fold{k}"
        data_dir, adapter_dir = fold_dir / "data", fold_dir / "adapter"
        counts = write_fold_data(data_dir, rows(train_df[~val_mask]),
                                 rows(train_df[val_mask]), codec)
        print(f"== fold {k}: train {counts['train']} / valid {counts['valid']} / "
              f"test {len(test_df)} ({test_df.speaker.nunique()} speakers)", flush=True)

        train_fold(fold_config(data_dir, adapter_dir, args, args.seed + k),
                   fold_dir / "config.yaml")

        scorer = LoRABandScorer(codec, model_path=args.model, adapter_path=str(adapter_dir),
                                temperature=args.temperature)
        out = scorer.score_batch([build_prompt(t, p, codec)
                                  for t, p in zip(test_df.transcript, test_df.part)])
        cache["folds"][str(k)] = {
            "utt": list(test_df.utt),
            "speaker": list(test_df.speaker),
            "part": list(test_df.part),
            "y_true": [float(s) for s in test_df.score],
            "expected": [float(s) for s in out["expected"]],
            "argmax": [float(s) for s in out["argmax"]],
            # the full distribution, so notebook 05 can re-decode at another temperature
            # without retraining — the compression trade is a decode question, not a
            # training question
            "probs": [[round(float(p), 6) for p in row] for row in out["probs"]],
        }
        oof_path.write_text(json.dumps(cache))
        print(f"== fold {k} done: {len(cache['folds'])}/{args.n_splits} folds cached", flush=True)

    print(f"wrote {oof_path}", flush=True)


if __name__ == "__main__":
    main()
