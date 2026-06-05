"""Synthetic TSVs and DataFrames standing in for corpus metadata."""

import numpy as np
import pandas as pd

from speechlab.data import load_responses, stratified_sample


def _write_corpus(tmp_path, n=6):
    ref = tmp_path / "reference-materials"
    (ref / "sla-marks").mkdir(parents=True)
    (ref / "flists.flac").mkdir(parents=True)
    speakers = [f"SI000{i}-0000{i}" for i in range(n)]
    pd.DataFrame({"speaker": speakers, "score": np.linspace(2.0, 5.0, n)}).to_csv(
        ref / "sla-marks" / "dev-sla-P3.tsv", sep="\t", header=False, index=False)
    pd.DataFrame({"utt": [f"{s}-P3000{i}" for i, s in enumerate(speakers)],
                  "flac": [f"data/flac/dev/file{i}.flac" for i in range(n)]}).to_csv(
        ref / "flists.flac" / "dev-sla-P3.tsv", sep="\t", header=False, index=False)
    return tmp_path


def test_load_responses_joins_scores_to_files(tmp_path):
    corpus = _write_corpus(tmp_path)
    df = load_responses(corpus, "P3")
    assert len(df) == 6
    assert set(df.columns) == {"utt", "flac", "speaker", "score"}
    assert df.score.between(2.0, 5.0).all()


def test_stratified_sample_is_deterministic_and_sized():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "utt": [f"u{i}" for i in range(200)],
        "score": rng.uniform(1.0, 6.0, 200).round(1),
    })
    a = stratified_sample(df, n=50, seed=42)
    b = stratified_sample(df, n=50, seed=42)
    # per-band rounding may undershoot by a few rows; never overshoots
    assert 45 <= len(a) <= 50
    assert a.utt.tolist() == b.utt.tolist()          # same seed -> same rows, same order
    assert stratified_sample(df, n=50, seed=7).utt.tolist() != a.utt.tolist()
