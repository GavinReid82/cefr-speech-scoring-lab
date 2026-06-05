# Evaluation report — automated CEFR scoring of L2 English spoken responses

**System:** whisper-small ASR → 17 fluency/prosody features → Random Forest
**Data:** Speak & Improve Corpus 2025, scored long-turn dev set (876 responses, 438 speakers)
**Companion:** `model_card.md` (summary), notebooks 01–04 (full analysis), `data_gates.md` (data verification)

---

## 1. Executive summary

A feature-based scorer reaches **QWK 0.558** against human consensus scores under
speaker-grouped cross-validation, with **81.5% of predictions within half a band**. The
evaluation surfaced five findings that matter more than the headline:

1. The scorer **compresses the scoring scale**: it over-scores the weakest candidates by
   nearly a full band (+0.96 at band 2) and under-scores the strongest (−0.70 at band 5).
2. **ASR error does not independently bias scores.** The apparent WER→error relationship
   (r = +0.15) vanishes when proficiency is controlled (r = −0.01) — a mediation effect
   that a naive analysis would have misreported as ASR-induced unfairness.
3. **Small-sample evaluation inflates results**: the identical pipeline scored QWK 0.647 on
   a 100-response sample vs 0.604 on the full 438-response part.
4. **Word count alone captures ~90% of the achievable QWK** (0.500 of 0.558) — the model is
   mostly measuring how much candidates say.
5. **Whisper erases the disfluency signal entirely** (100% of filled pauses), removing a
   cue that gold annotations show is predictive (r = −0.25).

## 2. System under evaluation

Audio (FLAC, ~60 s long turns) → faster-whisper `small` (int8, word timestamps) → 11
fluency features from the timestamps (rates, pause statistics, mean length of run, MTLD
lexical diversity, filled pauses) + 5 prosody features from the raw audio via Praat
(pitch spread in semitones, voiced ratio, intensity spread) + a part indicator → Random
Forest (500 trees, `min_samples_leaf=5`).

Design choices with evaluation consequences:

- **MTLD replaces TTR** after notebook 01 showed TTR correlating *negatively* with
  proficiency (r = −0.23) — the textbook length-sensitivity artefact. MTLD correlates
  +0.34 and lifted QWK by +0.026 at n=100.
- **Intensity means are excluded** (recording gain varies by device); only spread is used.
- **Pitch statistics are in semitones**, making them comparable across speaker registers.

## 3. Data and protocol

876 responses (438 speakers × Parts 3 and 4), human-scored 0–6 at 0.5 intervals, single
consensus label per response. Verification of scale, labels and transcript availability:
`reports/data_gates.md`.

**Speaker-grouped 5-fold cross-validation** throughout: each speaker appears in exactly one
fold, since their two responses share underlying proficiency. The leakage this prevents was
measured, not assumed: naive random folds inflate Ridge QWK by +0.015 and Random Forest by
nothing. The near-null is itself evidence that these handcrafted features barely encode
speaker identity beyond proficiency.

Because the corpus provides single consensus labels, **no human–human agreement ceiling is
available**; model–human agreement below cannot be benchmarked against inter-rater QWK.

## 4. Results

| Model | Pearson r | Spearman ρ | MAE | RMSE | QWK |
|---|---|---|---|---|---|
| Mean predictor | — | — | 0.580 | 0.733 | 0.000 |
| Word-count Ridge | 0.589 | 0.599 | 0.474 | 0.593 | 0.500 |
| Ridge (17 features) | 0.609 | 0.610 | 0.464 | 0.582 | 0.548 |
| **Random Forest** | **0.622** | **0.625** | **0.456** | **0.575** | **0.558** |
| MLP (PyTorch) | 0.483 | 0.486 | 0.580 | 0.727 | 0.475 |

- The word-count baseline is the most important row: features beyond quantity-of-speech buy
  +0.058 QWK. Any claimed improvement to this system should be benchmarked against word
  count, not the mean predictor.
- The MLP is an honest null: 876 rows is too few for a neural model at default settings.
  It remains in the table as the baseline for when features or data grow.
- Per part: QWK 0.604 (P3) vs 0.496 (P4) with the same features and speakers — the model
  *ranks* P4 responses less well, while error size and bias are comparable
  (MAE 0.43 vs 0.48, signed error ≈ 0 for both).

## 5. Agreement and calibration

| Tolerance | Agreement |
|---|---|
| Exact (same 0.5 step) | 34.1% |
| Within half a band | 81.5% |
| Within one band | 97.7% |

The calibration curve bends at both extremes: the model rarely predicts below 3 or above 5.
Mean signed error by band:

| Band | n | MAE | Signed error |
|---|---|---|---|
| 2 | 42 | 0.964 | **+0.964** |
| 3 | 290 | 0.455 | +0.361 |
| 4 | 412 | 0.327 | −0.120 |
| 5 | 132 | 0.701 | **−0.701** |

At band 2, signed error equals MAE: **every band-2 response is over-scored.** This is
regression to the mean operating on an imbalanced score distribution, and it is the
finding with operational teeth — the scorer is most wrong, and wrong in the candidate's
favour, exactly where a pass/fail decision would sit. Per speaker (mean of two responses):
median MAE 0.39, 90th percentile 0.83, maximum 1.71.

## 6. ASR error impact

whisper-small scores **16.8% corpus WER** against gold disfluent transcripts (lenient
normalisation; +4.1 points when disfluencies must also be transcribed). WER is heavily
right-skewed (per-response mean 22.2%, median 12.1%) and rises as proficiency falls
(r = −0.28).

The propagation analysis proceeded in two steps across notebooks 02 and 04:

1. **The alarm:** WER correlates with *signed* scoring error (r = +0.15, p = .0002) —
   apparently, badly-transcribed responses get over-scored.
2. **The resolution:** controlling for proficiency, the partial correlation is **−0.01
   (p = .75)**. High-WER speakers are low-proficiency speakers, and low-proficiency
   speakers are over-scored by band compression. The ASR effect was proficiency in
   disguise.

Caveat: this null holds for a system whose features barely use transcript *content*.
Content-sensitive features (lexical, grammatical, semantic relevance) would re-open the
question, as would the separately-demonstrated disfluency erasure: Whisper transcribed
**0 of 333** gold-annotated filled pauses, and the erased signal is predictive of score
(gold disfluency rate, r = −0.25, p = .046).

## 7. Fairness

The corpus contains **no L1 metadata**, so the most important fairness slice in the
language-assessment literature cannot be assessed. This is a limitation of the evaluation,
stated rather than silently skipped. The supported slices:

| Slice | Finding |
|---|---|
| Audio quality (Q3/Q4/Q5 STM labels, 600 gold-covered responses) | MAE flat within 0.08; mild signed-error gradient tracks group proficiency, not audio. Thin: n=22 at Q3, no Q2/QX present. |
| Proficiency band | Not uniform — see band compression (§5). |
| Test part | Error parity on size, not ranking (QWK 0.60 vs 0.50). |

## 8. Threats to validity

- **Single consensus labels** — no inter-rater ceiling; QWK 0.558 cannot be positioned
  against human agreement.
- **Dev set only** — the eval split is untouched; no held-out confirmation of these figures.
- **Long turns only** — P1/P5 short responses excluded from modelling so far.
- **Small data, demonstrated risk** — the n=100 → n=438 QWK drop (0.647 → 0.604) is direct
  evidence that figures at this scale carry optimism; 876 responses is still small.
- **One ASR system, one size** — all ASR-dependent findings are specific to whisper-small.

## 9. Recommendations

1. Any future model change must beat the **word-count baseline**, reported alongside.
2. Apply and evaluate a **calibration correction** for band compression before any
   stakes-adjacent use.
3. Add **content features** (lexical/grammatical/semantic), then re-run the ASR-propagation
   analysis — the current null is conditional on content-blind features.
4. Re-run the audio-quality slice once P1/P5 transcription completes (more Q2/QX coverage).
5. Report per-candidate reliability (per-speaker error distribution), not only aggregate QWK.
