# Model card — CEFR spoken-response scorer (Random Forest, v0.1)

Following the structure of Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model details

| | |
|---|---|
| System | ASR → feature extraction → regression pipeline scoring L2 English spoken responses on the 0–6 (≈CEFR) scale |
| ASR | `whisper-small` via faster-whisper (CTranslate2, int8), word timestamps, English forced |
| Features (17) | 11 fluency (speech/articulation rate, pause statistics, length of run, lexical diversity as MTLD, filled pauses), 5 prosody (pitch spread in semitones, voiced ratio, intensity spread via Praat/parselmouth), test-part indicator |
| Model | `RandomForestRegressor` — 500 trees, `min_samples_leaf=5`, seed 42 |
| Version / date | v0.1, 5 June 2026 (metrics unchanged; alternatives section added 27 August 2026, its untrained floor 28 August 2026) |
| Code | `src/speechlab/`, notebooks 01–05 in this repository |

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
| LoRA transcript scorer (argmax) | 0.505 | 0.511 | 0.446 |
| LoRA transcript scorer (expected) | 0.550 | 0.523 | 0.415 |
| *same transcript scorer, untrained* | 0.228 | 0.886 | 0.144 |

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
   The obvious remedy was tested and did not work: see *Alternatives considered*.

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
  r = −0.01 controlling for score). Re-tested on a scorer that reads only the transcript
  and replicates (+0.161 → −0.036), so this is not an artefact of content-blind features.

## Alternatives considered

**A content-aware scorer was built, evaluated and rejected** — recorded here because a
negative result about the obvious next step is part of what this card should carry.
`Qwen2.5-1.5B-Instruct-4bit` was LoRA-fine-tuned (rank 8, 8 layers, 400 iterations, trained
locally under MLX as the licence requires) to emit a proficiency band from the transcript
alone, then decoded ordinally over the band distribution. Under the identical
speaker-grouped protocol and on identical held-out rows it reaches **QWK 0.446** (argmax) /
**0.415** (expectation) — last of the four real arms and below the word-count floor.

**The alternative was not undertrained.** Scored with the adapter removed — same prompt,
same rows, no fitting — the base model reaches QWK 0.144, so the fine-tuning is worth
+0.302 and the arm's shortfall is a ceiling rather than a failed run. Untrained it places
797 of 876 responses in the top band; naming the CEFR anchors in the prompt spreads the
predictions over five bands and improves MAE without improving QWK, so the floor belongs
to the model rather than to the label scheme.

Three properties of that result bear on this model rather than only on the alternative:

- **It does not fix band compression; it worsens it.** +1.028 / −1.012 at the band edges
  against this model's +0.96 / −0.70, and 27.6% exact agreement against 34.1%.
- **Its two decodes disagree about which is better**, and both are reported. Expectation
  ranks better (r 0.550 vs 0.505) and compresses harder (predicted sd 0.39 vs 0.49, against
  a human sd of 0.73); QWK is computed on the snapped grid and charges for compression that
  Pearson r ignores.
- **Reliability was worse, not only accuracy.** One fold in five collapsed to two output
  classes, which no aggregate metric in the table above would have revealed.

Scope: this is a result at 1.5B and rank 8 on 876 responses, not a general finding about
content features. Fine-tuned adapters are corpus derivatives and are not distributed.
Full analysis in notebook 05 and §4/§6 of `evaluation_report.md`.

## Caveats and recommendations

- Trained and evaluated on dev-set long turns only; P1/P5 short responses and the eval
  split are untouched.
- Single consensus labels mean model–human agreement cannot be compared against a
  human–human ceiling.
- Small-sample warning, demonstrated: the same pipeline scored QWK 0.647 at n=100 vs 0.604
  at n=438 on P3. Treat sub-1000-response evaluation figures as upper bounds.
- Before any operational consideration: calibration correction for band compression,
  L1-annotated data for the missing fairness slice, and a content-feature approach that
  actually clears the word-count baseline — the one tested did not.
