"""Corpus loading and sampling for the Speak & Improve Corpus 2025."""

from pathlib import Path

import pandas as pd


def load_responses(corpus: Path, part: str, split: str = "dev") -> pd.DataFrame:
    """Join the audio file list with human scores for one test part.

    Returns one row per response: utt, flac (relative path), speaker, score.
    """
    ref = corpus / "reference-materials"
    scores = pd.read_csv(ref / "sla-marks" / f"{split}-sla-{part}.tsv",
                         sep="\t", names=["speaker", "score"])
    flist = pd.read_csv(ref / "flists.flac" / f"{split}-sla-{part}.tsv",
                        sep="\t", names=["utt", "flac"])
    # utteranceID 'SI114J-00011-P30017' -> speakerID 'SI114J-00011'
    flist["speaker"] = flist["utt"].str.rsplit("-", n=1).str[0]
    return flist.merge(scores, on="speaker", how="inner")


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Reproduce the notebook-01 band-stratified sample, byte-for-byte.

    Row order matters downstream: it determines the K-fold split, so this must
    stay identical to the notebook-01 code (same seed -> same folds).
    """
    df = df.copy()
    df["band"] = df.score.clip(1, 5.99).astype(int)  # integer CEFR-ish band: 1=A1 .. 5=C1
    sample = (
        df.groupby("band", group_keys=False)
          .apply(lambda g: g.sample(max(1, round(len(g) * n / len(df))), random_state=seed),
                 include_groups=False)
    )
    return sample.sample(min(n, len(sample)), random_state=seed).reset_index(drop=True)
