# CEFR Spoken Response Scoring Lab

ASR, fluency features, reliability and fairness evaluation for L2 English speaking assessment.

A small Python/PyTorch evaluation lab that scores short learner spoken responses from the
[Speak & Improve Corpus 2025](https://englishlanguageitutoring.com/datasets/speak-and-improve-corpus-2025)
against real human proficiency ratings, and — more importantly — evaluates the scorers themselves:
correlation, quadratic weighted kappa, scorer consistency, fairness slices, and ASR-error impact analysis.

> The point is not to build a state-of-the-art scorer. The point is to show the
> evaluation science behind one.

## Status

Early setup. Design doc approved 2026-06-04; gate questions (score scale, label
structure, gold transcripts) answered from corpus metadata — see `reports/` as they land.

## Data and licence

This repo contains **no corpus data**. The Speak & Improve Corpus 2025 is
research-licensed (Cambridge University Press & Assessment / ELiT) and must be
obtained directly from the distributor. `data/` is gitignored from commit zero;
all committed reports contain aggregate figures only. See `data/README.md`
(local-only) for layout.

## Planned structure

```
notebooks/   01_vertical_slice → 02_feature_extraction → 03_model_training → 04_evaluation
src/         asr, acoustic_features, text_features, scoring_models, evaluation, fairness
reports/     model_card, evaluation_report, error_analysis
tests/       pytest on synthetic fixtures (no corpus data)
```
