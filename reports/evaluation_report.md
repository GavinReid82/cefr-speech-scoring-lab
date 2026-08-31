# Evaluation report — automated CEFR scoring of L2 English spoken responses

**System:** whisper-small ASR → 17 fluency/prosody features → Random Forest
**Data:** Speak & Improve Corpus 2025, scored long-turn dev set (876 responses, 438 speakers)
**Companion:** `model_card.md` (summary), `error_analysis.md` (failure anatomy),
notebooks 01–05 (full analysis), `data_gates.md` (data verification)

---

## 1. Executive summary

A feature-based scorer reaches **QWK 0.558** against human consensus scores under
speaker-grouped cross-validation, with **81.5% of predictions within half a band**. The
evaluation surfaced six findings that matter more than the headline:

1. The scorer **compresses the scoring scale**: it over-scores the weakest candidates by
   nearly a full band (+0.96 at band 2) and under-scores the strongest (−0.70 at band 5).
2. **ASR error does not independently bias scores.** The apparent WER→error relationship
   (r = +0.15) vanishes when proficiency is controlled (r = −0.01) — a mediation effect
   that a naive analysis would have misreported as ASR-induced unfairness. This was
   re-tested against its own stated caveat and **replicates** on a scorer that reads
   nothing but the transcript (§6). Broken down by *error type* rather than by WER, one
   small residual does survive, in the opposite direction to the alarm: deleted words
   lead to **under**-scoring (partial r = −0.12), about 1% of the error variance
   (`error_analysis.md` §4).
3. **Small-sample evaluation inflates results**: the identical pipeline scored QWK 0.647 on
   a 100-response sample vs 0.604 on the full 438-response part.
4. **Word count alone captures ~90% of the achievable QWK** (0.500 of 0.558) — the model is
   mostly measuring how much candidates say.
5. **Whisper erases the disfluency signal entirely** (100% of filled pauses), removing a
   cue that gold annotations show is predictive (r = −0.25).
6. **Reading content did not help — but the training did.** A 1.5B instruct model
   LoRA-fine-tuned on the transcripts reaches QWK 0.446 on identical rows — below the
   word-count baseline, and with *worse* scale compression than the feature-based scorer.
   Untrained, the same model scores 0.144, so fine-tuning is worth +0.302 and the arm's
   weakness is its ceiling, not a failed training run (§4).

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
| LoRA transcript scorer (argmax) | 0.505 | 0.529 | 0.511 | 0.670 | 0.446 |
| LoRA transcript scorer (expected) | 0.550 | 0.559 | 0.523 | 0.647 | 0.415 |
| *same model untrained* (expected) | 0.228 | 0.233 | 0.886 | 1.076 | 0.144 |
| *same model untrained* (argmax) | 0.151 | 0.145 | 1.525 | 1.689 | 0.072 |

- The word-count baseline is the most important row: features beyond quantity-of-speech buy
  +0.058 QWK. Any claimed improvement to this system should be benchmarked against word
  count, not the mean predictor.
- The MLP is an honest null: 876 rows is too few for a neural model at default settings.
  It remains in the table as the baseline for when features or data grow.
- **The content-aware arm is a reported negative result.** A `Qwen2.5-1.5B-Instruct-4bit`
  model LoRA-fine-tuned (rank 8, 8 layers) to emit a proficiency band from the transcript
  alone finishes last of the four real arms and below the word-count floor, on exactly the
  same held-out rows — notebook 05 reads the fold assignment out of the LoRA cache rather
  than re-splitting, so the comparison is not a fold artefact. Its two ordinal decodes
  disagree about which is better and are both reported: expectation decoding ranks better
  (r 0.550 vs 0.505) and compresses harder (predicted sd 0.39 vs 0.49, against human 0.73),
  and QWK — computed on the snapped 0.5 grid — charges for the compression that Pearson r
  ignores. Prediction range tells the story: 1.72–2.00 against 2.38–2.93 for the linear and
  forest arms. The claim is scoped to this scale: 1.5B at rank 8 on 876 responses is the
  smallest credible version of the experiment, not a result about content features
  in general.
- **Every arm needs its own floor, and this one was published without it.** The two rows
  above were added after the fine-tuned arm had already been written up, when the obvious
  reader's question — did the fine-tuning help? — turned out to be unanswerable from the
  table. It was not: 0.446 had only ever been compared against a *different pipeline*.
  Scoring the same model with the adapter removed puts the untrained floor at QWK 0.144,
  so the LoRA is worth **+0.302** and the arm's problem is its ceiling rather than its
  training. Untrained, the model assigns 797 of 876 responses to the top band and the
  remaining 79 to the bottom one; naming the CEFR anchors in the prompt (rather than the
  deliberately blank `A`–`H` labels that fine-tuning wants) spreads the predictions across
  five bands and improves MAE to 0.728 without improving QWK, which confirms the floor is
  the model's and not the labelling's. Two cautions for reading the table above: prediction
  range is uninformative when a decode uses two of eight bands (the untrained argmax spans
  the full 3.5), and predicted spread is not calibration — the untrained expectation decode
  has sd 0.594 against the fine-tuned 0.391 and the human 0.733, while ranking three times
  worse.
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

Compression is not an artefact of the feature set. The content-aware arm compresses
*harder* at both edges — +1.028 at band 2 and −1.012 at band 5 (expectation decoding),
against the forest's +0.964 / −0.701 — with exact agreement 27.6% against 34.1%. Decoding
the band distribution at a lower temperature recovers only ~+0.03 QWK before running out
(predicted sd still 0.40 against the human 0.73), which locates the fix in calibration
rather than in decoding: sharpening moves scale, not ranking. Any such calibration must be
fitted on an inner split of the training folds; the temperature that maximises QWK on the
evaluation set is not a figure that can stand beside 0.558.

### Intra-scorer consistency

The design specifies consistency as QWK between two independent LLM scoring passes, with
deterministic models exempt. **Every arm in this report turns out to be exempt**, including
the LLM one: `LoRABandScorer` takes a single forward pass and reads eight logits at fixed
token ids, with no sampling anywhere, so a repeat pass is bit-identical and pass-to-pass
QWK is 1.0 by construction. That number measures the decode's determinism, not the
scorer's reliability, and is not reported as a consistency figure.

The measurable version is agreement across a benign **rewording** of the prompt — the two
zero-shot passes of §4, blank band letters versus the same letters anchored to CEFR names:

| | expected | argmax |
|---|---|---|
| pass-to-pass QWK | 0.512 | 0.244 |
| pass-to-pass r | 0.717 | 0.420 |
| exact agreement | 39.6% | 18.5% |
| *the same passes vs human, best of the two* | *0.144* | *0.072* |

Two readings, and both matter. Renaming the bands moves 60% of the expectation decode's
band assignments, which is substantial instability for a change that adds information and
removes none. But self-agreement is a **ceiling** on agreement with humans, and the
untrained arm sits at 0.144 under a ceiling of 0.512 — a quarter of the way up. Its floor
is therefore a validity problem, not a reliability one, which rules out the reading that
§4's zero-shot number is low merely because the prompt was arbitrary. No equivalent figure
exists for the fine-tuned arm: measuring it needs a second fine-tune under a reworded
prompt, which is a training run, not a decode.

## 6. ASR error impact

whisper-small scores **16.8% corpus WER** against gold disfluent transcripts (lenient
normalisation; +4.1 points when disfluencies must also be transcribed). WER is heavily
right-skewed (per-response mean 22.2%, median 12.1%) and rises as proficiency falls
(r = −0.28).

The propagation analysis proceeded in three steps across notebooks 02, 04 and 05:

1. **The alarm:** WER correlates with *signed* scoring error (r = +0.15, p = .0002) —
   apparently, badly-transcribed responses get over-scored.
2. **The resolution:** controlling for proficiency, the partial correlation is **−0.01
   (p = .75)**. High-WER speakers are low-proficiency speakers, and low-proficiency
   speakers are over-scored by band compression. The ASR effect was proficiency in
   disguise.
3. **The caveat, tested.** This null was originally qualified: it held for a system whose
   features barely use transcript *content*, and content-sensitive features would re-open
   the question. That test has now been run. A LoRA-fine-tuned scorer reading nothing but
   the transcript (§4) shows the same pattern — raw r = **+0.161** (p = .0001), partial
   r = **−0.036** (p = .379) — and the Random Forest reproduces its published figures to
   three decimals on the same rows and the same code path (+0.151 → −0.014). **The null is
   not an artefact of content-blind features.**

| Scorer | WER vs signed error | Partial (proficiency controlled) |
|---|---|---|
| Random Forest | +0.151 (p = .0002) | −0.014 (p = .733) |
| LoRA (expected) | +0.161 (p = .0001) | −0.036 (p = .379) |
| LoRA (argmax) | +0.117 (p = .0040) | −0.068 (p = .095) |

**One refinement, from `error_analysis.md` §4.** Steps 1 and 2 test *overall* WER. Broken
into its components — substitution, deletion and insertion rates, all over the same
denominator — deletion rate retains a partial correlation with signed error of **−0.119**
(p = .004) where WER retains −0.014 (p = .73), and it **replicates on the LoRA arm**
(−0.114, p = .005). The sign is the interesting part: it is *negative*, so once
proficiency is held constant, a transcript with words missing is **under**-scored — the
opposite direction to the raw alarm in step 1, and mechanically what `n_words` being the
dominant feature predicts. The effect is about 1% of the variance in signed error against
a band-compression term of nearly a full band, so the operational conclusion is unchanged;
what changes is the strength of the phrasing. The null holds for aggregate WER and not
quite for deletion. Insertion rate, including the responses where Whisper loops and emits
several times the spoken length, shows nothing at all (+0.017, p = .68).

The *signed* qualifier is load-bearing and easy to lose: signed error asks whether bad
transcription pushes scores in a **direction**, absolute error asks whether it makes them
less accurate either way. Only the first is the claim above, and the two do not agree
(WER vs |error| is −0.031 for the forest, −0.083 for LoRA expected). A draft of this
analysis measured |error| and would have reported a non-replication that was really a
different question.

What remains untested is the *interventional* form of the question — retraining on gold
transcripts and comparing. That was attempted and is **confounded**: the gold subset is 589
of 876 responses, so the gold arm trains on 425 rows per fold against 630 and, at a fixed
iteration budget, sees ~3.8 epochs against ~2.5. Transcript quality, supervision volume and
training budget all move together, so the observed drop (QWK 0.345 vs 0.368 on matched
rows) supports only the weaker claim that gold transcripts do not help enough to overcome a
smaller training set. The clean control is listed in §9.

Unaffected by any of this: the separately-demonstrated disfluency erasure. Whisper
transcribed **0 of 333** gold-annotated filled pauses on the notebook-01 sample, and **0
of 3,058** across all 600 gold-covered responses; the erased signal is predictive of score
(gold disfluency rate, r = −0.25, p = .046). This is the largest ASR-attributable harm in
the system and it appears in no WER figure — the lenient reference strips hesitations by
design, and the strict one charges only +4.1 points for them.

`error_analysis.md` carries the full anatomy: the composition of the ASR errors
(substitutions 36.7%, insertions 33.9%, deletions 29.4%), five case classes with their
counts and the scorer's behaviour on each, and why the class table must be read against
each class's mean proficiency rather than against zero.

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

1. Any future model change must beat the **word-count baseline**, reported alongside —
   *and* report its own untrained or ablated floor. The two are different questions and
   §4 shipped the second one unanswered: the LoRA arm's headline was compared only against
   other pipelines, which cannot say whether the training did anything.
2. Apply and evaluate a **calibration correction** for band compression before any
   stakes-adjacent use.
3. ~~Add **content features**, then re-run the ASR-propagation analysis.~~ **Done (§4, §6).**
   A LoRA-fine-tuned transcript scorer reached QWK 0.446 against the forest's 0.558 —
   below the word-count baseline, though +0.302 above its own untrained floor — and the
   ASR-propagation null replicated under it. The
   follow-ups this raises, in priority order: run the **clean transcript-quality control**
   (Whisper transcripts restricted to the same 589 gold-covered rows, so the arms differ
   only in transcript source, unlike the confounded ablation in §6); keep a
   **best-checkpoint fallback** when fine-tuning, since three of five folds ended on
   weights worse than ones the run passed through; and test whether the negative result
   holds at larger scale before generalising it beyond 1.5B / rank 8.
4. Re-run the audio-quality slice once P1/P5 transcription completes (more Q2/QX coverage).
5. Report per-candidate reliability (per-speaker error distribution), not only aggregate QWK.
