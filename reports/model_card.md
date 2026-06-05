# Model card — CEFR spoken-response scorer (Random Forest, v0.1)

Following the structure of Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model details

| | |
|---|---|
| System | ASR → feature extraction → regression pipeline scoring L2 English spoken responses on the 0–6 (≈CEFR) scale |
| ASR | `whisper-small` via faster-whisper (CTranslate2, int8), word timestamps, English forced |
| Features (17) | 11 fluency (speech/articulation rate, pause statistics, length of run, lexical diversity as MTLD, filled pauses), 5 prosody (pitch spread in semitones, voiced ratio, intensity spread via Praat/parselmouth), test-part indicator |
| Model | `RandomForestRegressor` — 500 trees, `min_samples_leaf=5`, seed 42 |
| Version / date | v0.1, 5 June 2026 |
| Code | `src/speechlab/`, notebooks 01–04 in this repository |

## Intended use

**Primary:** a research artefact demonstrating evaluation methodology for automated speaking
assessment — agreement analysis, error analysis, fairness slicing, ASR-impact analysis.

**Out of scope:** any operational scoring decision about a real candidate. The error
characteristics below (especially band compression) make stakes-bearing use inappropriate.

## Data

Speak & Improve Corpus 2025 (Cambridge University Press & Assessment / ELiT), research
licence. Working set: the full scored long-turn dev set — Parts 3 and 4, **876 responses
from 438 speakers**, one human score each (0–6 at 0.5 intervals, single consensus label per
response). No corpus data is distributed with this repository.

## Evaluation protocol

5-fold cross-validation **grouped by speaker** (each speaker contributes a P3 and a P4
response; random folds leak speaker identity — measured inflation +0.015 QWK for Ridge,
none for RF). QWK computed on the 0.5-step grid. All figures are out-of-fold predictions.

## Metrics

| Model | Pearson r | MAE | QWK |
|---|---|---|---|
| Mean predictor | — | 0.580 | 0.000 |
| Word-count Ridge | 0.589 | 0.474 | 0.500 |
| Ridge (17 features) | 0.609 | 0.464 | 0.548 |
| **Random Forest (this model)** | **0.622** | **0.456** | **0.558** |
| MLP (PyTorch, default settings) | 0.483 | 0.580 | 0.475 |

Agreement with human scores: **34.1% exact** (same 0.5 step), **81.5% within half a band**,
**97.7% within one band**. Per part: QWK 0.604 (P3) vs 0.496 (P4).

## Known failure modes

1. **Band compression.** Mean signed error +0.96 at band 2 (every band-2 response
   over-scored), −0.70 at band 5. The model flatters the weakest candidates and penalises
   the strongest; it rarely predicts below 3 or above 5.
2. **Per-speaker tail.** Median per-speaker MAE 0.39, but the 90th percentile is 0.83 and
   the worst-served speaker averages 1.71 bands of error.
3. **Disfluency blindness.** Whisper erases 100% of filled pauses (333 spoken → 0
   transcribed in the gold-covered sample), removing a proficiency cue that is predictive
   in the gold annotations (r = −0.25).
4. **Word count dominates.** A word-count-only baseline reaches QWK 0.500 of the model's
   0.558 — most of the signal is quantity of speech, a known property of fluency features.

## Fairness

Slices the corpus supports (no L1 metadata exists, so **L1 fairness cannot be assessed** —
the most important slice in the assessment literature is a stated limitation):

- **Audio quality** (STM labels, 600 gold-covered responses): MAE flat within 0.08 across
  Q3/Q4/Q5; mild signed-error gradient tracks group proficiency, not audio quality. Thin
  evidence: 22 Q3 responses, no Q2/QX present.
- **Proficiency band:** see band compression above — error is not uniform across bands.
- **Test part:** comparable error size (MAE 0.43 vs 0.48), weaker *ranking* on P4
  (QWK 0.50 vs 0.60).
- **ASR-error propagation:** none independent of proficiency (raw r = +0.15 collapses to
  r = −0.01 controlling for score).

## Caveats and recommendations

- Trained and evaluated on dev-set long turns only; P1/P5 short responses and the eval
  split are untouched.
- Single consensus labels mean model–human agreement cannot be compared against a
  human–human ceiling.
- Small-sample warning, demonstrated: the same pipeline scored QWK 0.647 at n=100 vs 0.604
  at n=438 on P3. Treat sub-1000-response evaluation figures as upper bounds.
- Before any operational consideration: calibration correction for band compression,
  text/content features beyond fluency, L1-annotated data for the missing fairness slice.
