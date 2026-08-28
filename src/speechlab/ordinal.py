"""Ordinal decoding of the 0.5-step score grid for language-model scorers.

A language model asked for a band emits a *token*, and cross-entropy treats every
wrong token as equally wrong: predicting A1 for a C1 candidate costs exactly what
predicting B2 costs. The score scale is ordinal, so that is the wrong geometry —
and QWK, the metric this project reports, penalises distant disagreements
quadratically. This module restores the ordering at decode time:

- `BandCodec` maps the 0.5-step grid to **single-character class labels**, so one
  forward pass yields a full distribution over every band at a single position;
- `expected_score` returns sum_i p_i * band_i over that distribution — a continuous
  score on the original scale, which drops straight into `evaluation.evaluate`
  next to the Ridge and Random Forest predictions.

The expectation is not free: averaging pulls predictions towards the distribution's
centre, which is the same shrinkage that produced the band compression documented
in `reports/evaluation_report.md` §5. `argmax_score` is therefore kept alongside it
so notebook 05 can measure the trade rather than assume it, and `temperature` (<1
sharpens) exposes the dial between them.

Deliberately numpy-only — no mlx, no tokenizer — so it is testable on synthetic
arrays. The model-facing half lives in `speechlab.llm_scorer`.
"""

import numpy as np

from speechlab.evaluation import to_grid

# Single uppercase letters: one BPE token each in every vocab we target, which is what
# lets a single logit row carry the whole band distribution. Verified against the real
# tokenizer by llm_scorer.band_token_ids — this module never sees a tokenizer.
LABEL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

STEP = 0.5      # the corpus's manual-score interval (data_gates.md, gate 1)


class BandCodec:
    """Bidirectional map between scores on the 0.5-step grid and class labels.

    Snapping is delegated to `evaluation.to_grid`, so a training label and the
    gold class used in the QWK confusion matrix can never disagree.
    """

    def __init__(self, lo: float, hi: float):
        if hi <= lo:
            raise ValueError(f"hi must exceed lo, got lo={lo} hi={hi}")
        self.lo, self.hi = float(lo), float(hi)
        self.grid = np.arange(self.lo, self.hi + STEP / 2, STEP)
        if len(self.grid) > len(LABEL_ALPHABET):
            raise ValueError(f"{len(self.grid)} bands exceeds the single-token label alphabet")
        self.labels = list(LABEL_ALPHABET[:len(self.grid)])

    @classmethod
    def from_scores(cls, y) -> "BandCodec":
        """Build the grid from observed human scores — the same bounds `evaluate` uses."""
        y = np.asarray(y, dtype=float)
        return cls(float(y.min()), float(y.max()))

    def __len__(self) -> int:
        return len(self.grid)

    def encode(self, scores) -> list[str]:
        """Scores -> class labels, snapped and clipped exactly as `to_grid` does."""
        idx = to_grid(np.atleast_1d(scores), self.lo, self.hi) - int(round(self.lo * 2))
        return [self.labels[i] for i in idx]

    def decode(self, labels) -> np.ndarray:
        """Class labels -> the score at the centre of each band."""
        pos = {lab: i for i, lab in enumerate(self.labels)}
        return np.array([self.grid[pos[lab]] for lab in np.atleast_1d(labels)])


def band_probabilities(band_logits, temperature: float = 1.0) -> np.ndarray:
    """Softmax restricted to the band-label logits, row-wise and numerically stable.

    band_logits: (n_items, n_bands) — logits already gathered at the label token ids,
    i.e. the rest of the vocabulary is discarded rather than competing for mass. That
    renormalisation is what makes the output a distribution over *bands*.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    z = np.atleast_2d(np.asarray(band_logits, dtype=float)) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_score(probs, grid) -> np.ndarray:
    """sum_i p_i * band_i — the ordinal decode. Continuous, on the human score scale."""
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    return probs @ np.asarray(grid, dtype=float)


def argmax_score(probs, grid) -> np.ndarray:
    """The single most likely band — the decode you get for free from generation."""
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    return np.asarray(grid, dtype=float)[probs.argmax(axis=1)]
