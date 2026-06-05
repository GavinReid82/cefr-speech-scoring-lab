# CEFR Spoken Response Scoring Lab

ASR, fluency features, reliability and fairness evaluation for L2 English speaking assessment.

A small Python/PyTorch evaluation lab that scores short learner spoken responses from the
[Speak & Improve Corpus 2025](https://englishlanguageitutoring.com/datasets/speak-and-improve-corpus-2025)
against real human proficiency ratings, and — more importantly — evaluates the scorers themselves:
correlation, quadratic weighted kappa, scorer consistency, fairness slices, and ASR-error impact analysis.

> The point is not to build a state-of-the-art scorer. The point is to show the
> evaluation science behind one.

## Status

Four notebooks in: vertical slice → WER/prosody → full-scale models → error
analysis & fairness. Honest headline on the full long-turn dev set (876
responses, **speaker-grouped CV**): Random Forest **QWK 0.56** / r 0.62 — the
earlier n=100 figure (0.65) was small-sample optimism, corrected here
deliberately. Key findings: the scorer compresses the scale (over-scores weak
candidates +0.96, under-scores strong ones −0.70); ASR error does *not*
independently propagate into scoring error (partial correlation ≈ 0 controlling
for proficiency); 81.5% of predictions within half a band. Notebooks: `01`–`04`. **Deliverables:
`reports/model_card.md` and `reports/evaluation_report.md`.**
Design: `docs/design.md`. Session log: `JOURNAL.md`.

## Data and licence

This repo contains **no corpus data**. The Speak & Improve Corpus 2025 is
research-licensed (Cambridge University Press & Assessment / ELiT) and must be
obtained directly from the distributor. `data/` is gitignored from commit zero;
all committed reports contain aggregate figures only. See `data/README.md`
(local-only) for layout.

## Structure

```
notebooks/   01–04 (frozen session records)
src/         speechlab: data, asr, acoustic_features, text_features, scoring_models, evaluation
scripts/     download_dev_audio, transcribe_dev, extract_prosody (resumable)
tests/       pytest on synthetic fixtures (no corpus data)
reports/     data_gates, model_card, evaluation_report
```
