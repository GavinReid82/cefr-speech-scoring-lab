"""Regenerate the aggregate tables in reports/error_analysis.md.

The error-analysis material is spread across three notebooks — WER and disfluency
erasure in 02, scorer error anatomy and the propagation null in 04, the replication
under a content-reading scorer in 05 — and the report consolidates them. Everything
this script prints beyond that consolidation is new: §6 of the evaluation report tested
whether *overall* WER propagates into scoring error, and this asks the sharper question
it left open — whether a particular **kind** of ASR error does. Deletions are the
interesting case, because `n_words` is the Random Forest's dominant feature, so a
transcript that loses words loses the feature the scorer leans on hardest.

Output is aggregates only: counts, means and correlations by class. No utterance ids,
no speaker ids, no transcript text — the corpus licence forbids all three in anything
that reaches git, and this script's stdout is quoted directly into the report.

Usage:
    python scripts/error_analysis.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from speechlab.acoustic_features import FLUENCY_FEATURES, PROSODY_FEATURES, fluency_features
from speechlab.asr import parse_stm, parse_stm_tags, wer_pair
from speechlab.data import load_responses
from speechlab.evaluation import evaluate, to_grid
from speechlab.text_features import mtld, normalise_tokens

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "sandi-corpus-2025"
REF = CORPUS / "reference-materials"
DERIVED = ROOT / "data" / "derived"
PARTS = ["P3", "P4"]
SEED = 42
LORA_OOF = DERIVED / "lora_oof_whisper_Qwen2.5-1.5B-Instruct-4bit.json"


def residualise(v: np.ndarray, on: np.ndarray) -> np.ndarray:
    """Linear residual of `v` after regressing it on `on` — the partial-correlation step."""
    return v - np.polyval(np.polyfit(on, v, 1), on)


def partial_r(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    return pearsonr(residualise(x, control), residualise(y, control))


def build_features() -> pd.DataFrame:
    """Notebook 04's feature frame, rebuilt from the transcript and prosody caches."""
    frames = []
    for part in PARTS:
        df = load_responses(CORPUS, part)
        trans = json.load(open(DERIVED / f"transcripts_{part}_small.json"))
        pros = json.load(open(DERIVED / f"prosody_{part}_full.json"))
        rows = []
        for r in df.itertuples():
            words = trans[r.utt]
            duration = sf.info(str(CORPUS / r.flac)).duration
            rows.append({"utt": r.utt, "speaker": r.speaker, "part": part, "score": r.score,
                         **fluency_features(words, duration),
                         "mtld": mtld(normalise_tokens([w["w"] for w in words])),
                         **pros[r.utt]})
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    raw = build_features()
    n_silent = int((raw.n_words == 0).sum())
    data = raw[raw.n_words > 0].reset_index(drop=True)
    data["part_p4"] = (data.part == "P4").astype(float)

    features = ([f if f != "ttr" else "mtld" for f in FLUENCY_FEATURES]
                + PROSODY_FEATURES + ["part_p4"])
    y = data.score.to_numpy()
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                               random_state=SEED, n_jobs=-1)
    data["pred"] = cross_val_predict(rf, data[features].to_numpy(), y,
                                     cv=GroupKFold(n_splits=5),
                                     groups=data.speaker.to_numpy())
    data["err"] = data.pred - y                      # signed: positive = over-scored
    data["abs_err"] = data.err.abs()

    print("== 1. Rows and the scorer being analysed")
    print(f"responses scored by humans:      {len(raw)}")
    print(f"empty Whisper transcript:        {n_silent}   (excluded from modelling)")
    print(f"responses entering the model:    {len(data)}")
    print("Random Forest, speaker-grouped GroupKFold(5): "
          f"{ {k: round(v, 3) for k, v in evaluate(y, data.pred).items()} }")

    # ---- ASR error composition, on the gold-covered subset -------------------------
    gold = parse_stm(REF / "stms" / "dev-asr.stm")
    tags = parse_stm_tags(REF / "stms" / "dev-asr.stm")
    trans_all = {}
    for part in PARTS:
        trans_all |= json.load(open(DERIVED / f"transcripts_{part}_small.json"))

    rows = []
    for utt in data.utt:
        if utt not in gold:
            continue
        hyp = " ".join(w["w"] for w in trans_all[utt])
        if not hyp.strip():
            continue
        w = wer_pair(gold[utt], hyp)
        if w["n_gold"] == 0:
            continue
        rows.append({"utt": utt, "quality": tags[utt]["quality"], "n_hyp": len(hyp.split()),
                     **{k: w[k] for k in
                        ("wer_lenient", "n_gold", "subs", "dels", "ins")}})
    cov = data.merge(pd.DataFrame(rows), on="utt")

    # Rates share one denominator — gold words — so the three are directly comparable
    # and sum to the lenient WER.
    for kind in ("subs", "dels", "ins"):
        cov[f"{kind}_rate"] = cov[kind] / cov.n_gold
    cov["len_ratio"] = cov.n_hyp / cov.n_gold

    print("\n== 2. What the ASR errors are, on the gold-covered subset")
    print(f"gold-covered responses: {len(cov)} of {len(data)}")
    total = cov[["subs", "dels", "ins"]].sum()
    # Corpus WER is total errors over total gold words, not the mean of per-response WERs
    # — notebook 02 reports it over the 65-response notebook-01 sample, this is all 600.
    print(f"corpus WER (lenient): {total.sum() / cov.n_gold.sum():.1%}")
    print(f"error composition (share of all errors): "
          + ", ".join(f"{k} {v / total.sum():.1%}" for k, v in total.items()))
    print(f"insertion rate is the skewed one: max {cov.ins_rate.max():.2f} "
          f"against a median of {cov.ins_rate.median():.3f} — Whisper looping, not noise")
    dist = cov[["wer_lenient", "subs_rate", "dels_rate", "ins_rate", "len_ratio"]]
    print(dist.agg(["mean", "median",
                    lambda c: c.quantile(0.9)]).set_axis(["mean", "median", "p90"])
          .round(3).to_string())

    print("\n== 3. Does error *type* propagate, where overall WER did not?")
    print(f"{'variable':<14}{'raw r vs signed err':>22}{'partial (score held)':>24}")
    for col in ("wer_lenient", "subs_rate", "dels_rate", "ins_rate", "len_ratio"):
        r, p = pearsonr(cov[col], cov.err)
        rp, pp = partial_r(cov[col].to_numpy(), cov.err.to_numpy(), cov.score.to_numpy())
        print(f"{col:<14}{f'{r:+.3f} (p={p:.4f})':>22}{f'{rp:+.3f} (p={pp:.3f})':>24}")

    # ---- the case classes ----------------------------------------------------------
    # Thresholds are the sample's own deciles rather than round numbers: a fixed cut
    # would define the classes by the ASR system's absolute error level, and every
    # figure here is already specific to whisper-small.
    d90, s90, i90 = (cov[f"{k}_rate"].quantile(0.9) for k in ("dels", "subs", "ins"))
    classes = {
        "truncation (top-decile deletions)": cov.dels_rate >= d90,
        "corruption (top-decile substitutions)": cov.subs_rate >= s90,
        "hallucination (top-decile insertions)": cov.ins_rate >= i90,
        "truncation and corruption together": (cov.dels_rate >= d90) & (cov.subs_rate >= s90),
        "clean (bottom-quartile WER)": cov.wer_lenient <= cov.wer_lenient.quantile(0.25),
    }
    print("\n== 4. Case classes: what the scorer does with each kind of failure")
    print(f"{'class':<40}{'n':>5}{'WER':>8}{'human':>8}{'signed err':>12}{'MAE':>8}")
    for name, mask in classes.items():
        s = cov[mask]
        print(f"{name:<40}{len(s):>5}{s.wer_lenient.mean():>8.3f}{s.score.mean():>8.2f}"
              f"{s.err.mean():>+12.3f}{s.abs_err.mean():>8.3f}")
    print(f"{'all gold-covered':<40}{len(cov):>5}{cov.wer_lenient.mean():>8.3f}"
          f"{cov.score.mean():>8.2f}{cov.err.mean():>+12.3f}{cov.abs_err.mean():>8.3f}")

    # ---- the same classes under a scorer that reads the transcript -----------------
    if LORA_OOF.exists():
        oof = json.load(open(LORA_OOF))
        lora = pd.DataFrame([
            {"utt": u, "lora_pred": f["expected"][i]}
            for f in oof["folds"].values() for i, u in enumerate(f["utt"])])
        lcov = cov.merge(lora, on="utt")
        lcov["lora_err"] = lcov.lora_pred - lcov.score
        print(f"\n== 5. The same classes under the content-reading arm (n={len(lcov)})")
        print(f"{'class':<40}{'n':>5}{'signed err':>12}{'MAE':>8}")
        for name, mask in classes.items():
            s = lcov[lcov.utt.isin(cov.loc[mask, "utt"])]
            print(f"{name:<40}{len(s):>5}{s.lora_err.mean():>+12.3f}"
                  f"{s.lora_err.abs().mean():>8.3f}")
        r, p = pearsonr(lcov.dels_rate, lcov.lora_err)
        rp, pp = partial_r(lcov.dels_rate.to_numpy(), lcov.lora_err.to_numpy(),
                           lcov.score.to_numpy())
        print(f"deletion rate vs signed error: raw {r:+.3f} (p={p:.4f}), "
              f"partial {rp:+.3f} (p={pp:.3f})")

    # ---- the erasure that no WER figure charges for --------------------------------
    ann = json.load(open(REF / "annotations" / "dev-trans-ref.json"))
    covered = set(cov.utt)
    hes = sum(sum(1 for t in f["Transcript"] if t.get("tag") == "hesitation")
              for f in ann["files"] if f["File-id"] in covered)
    dis = sum(sum(1 for t in f["Transcript"] if "disfluency" in t.get("marks", []))
              for f in ann["files"] if f["File-id"] in covered)
    n_ann = sum(1 for f in ann["files"] if f["File-id"] in covered)
    filled = re.compile(r"^(um+|uh+|er+m?|eh+|h?mm+)$")
    whisper_fp = sum(
        bool(filled.match(re.sub(r"[^a-z\']", "", w["w"].lower())))
        for utt in covered for w in trans_all[utt])
    print(f"\n== 6. Disfluency erasure over the gold-covered set ({n_ann} annotated)")
    print(f"gold hesitation tags: {hes}; gold disfluency-marked words: {dis}")
    print(f"filled pauses in Whisper output: {whisper_fp}")


if __name__ == "__main__":
    main()
