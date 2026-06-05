# CEFR Spoken Response Scoring Lab

ASR, fluency features, reliability and fairness evaluation for L2 English speaking assessment.

A small Python/PyTorch evaluation lab that scores short learner spoken responses from the
[Speak & Improve Corpus 2025](https://englishlanguageitutoring.com/datasets/speak-and-improve-corpus-2025)
against real human proficiency ratings, and — more importantly — evaluates the scorers themselves:
correlation, quadratic weighted kappa, scorer consistency, fairness slices, and ASR-error impact analysis.

> The point is not to build a state-of-the-art scorer. The point is to show the
> evaluation science behind one.

## Status

Vertical slice complete, ASR error analysis done: 100 dev responses end-to-end
(Whisper → fluency + prosody features → Ridge vs baselines), best result
**QWK 0.65 / r 0.66** with MTLD lexical diversity
(`notebooks/02_wer_and_prosody.ipynb`); whisper-small WER 16.8% vs gold,
rising for weaker speakers — the seed of the fairness analysis. Code now lives in
an installable `speechlab` package (`src/`, 23 synthetic-fixture tests); full
dev-set transcription in progress. Slice results:
`notebooks/01_vertical_slice.ipynb`. Design and roadmap: `docs/design.md`.
Data verification: `reports/data_gates.md`. Session log: `JOURNAL.md`.

## Data and licence

This repo contains **no corpus data**. The Speak & Improve Corpus 2025 is
research-licensed (Cambridge University Press & Assessment / ELiT) and must be
obtained directly from the distributor. `data/` is gitignored from commit zero;
all committed reports contain aggregate figures only. See `data/README.md`
(local-only) for layout.

## Structure

```
notebooks/   01_vertical_slice, 02_wer_and_prosody (frozen session records) → 03_models next
src/         speechlab: data, asr, acoustic_features, text_features, evaluation
scripts/     download_dev_audio, transcribe_dev (resumable)
tests/       pytest on synthetic fixtures (no corpus data)
reports/     data_gates → model_card, evaluation_report, error_analysis to come
```
