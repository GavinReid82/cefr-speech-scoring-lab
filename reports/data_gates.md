# Data verification gates — Speak & Improve Corpus 2025

Answered 2026-06-04 from the base corpus package (`reference-materials/`), before any
modelling. These three questions gated the headline metrics in the design doc.
All figures here are aggregates; no learner data is reproduced.

## Gate 1 — Score scale: CONFIRMED (finer than assumed)

- Human proficiency scores are on a **6-point scale (0–6)** mapping approximately to
  CEFR: 1=A1, 2=A2, 3=B1, 4=B2, 5=C1 (`sla-marks/README.txt`).
- **Per-part scores** (parts P1, P3, P4, P5) are manual scores at **0.5 intervals**.
- **Overall score** = mean of the four part scores → 0.125 gradation.
- Scoring criteria are modelled on the Linguaskill Speaking Global Assessment Criteria.
- Dev set: **438 speakers**, **5,616 utterance-level audio files** (verified against the
  union of all dev file lists after download). Parts differ in shape: P1 ≈ several short
  question responses per speaker (2,575 files), P3/P4 = one long turn each (438 files
  each), P5 ≈ several responses (2,165 files). Scores exist per part and overall.

**Implication:** quadratic weighted kappa is computed on per-part ordinal scores
(or band-binned scores); RMSE/MAE on the continuous overall score; the per-band
confusion matrix uses integer CEFR bands. The design doc's "A2–C1+ → integers"
assumption holds, with finer resolution available than assumed.

## Gate 2 — Label structure: SINGLE consensus label per response

- One manual score per fileID per part; **no multi-rater labels are shipped**.
- **Implication:** inter-rater reliability is NOT computable from this corpus.
  As pre-agreed in the design doc, the reliability deliverable is therefore
  **model–human agreement** plus **intra-scorer consistency** (QWK between two
  independent LLM scoring passes). Published human–human benchmarks from the
  corpus/challenge papers can be cited for context.

## Gate 3 — Gold transcripts: YES, richer than hoped

- `stms/` ships **manual disfluent transcripts** (for ASR WER scoring), plus
  **fluent** (disfluencies removed) and **GEC-corrected** versions.
- `annotations/*-trans-ref.json` ships **word-level annotations with explicit
  disfluency marks**, question text, and speaking/thinking times
  (dev: 3,249 annotated responses).
- **Implication:** the ASR-error→score-error analysis is fully supported (WER vs
  gold), and fluency features (filled pauses, repetitions, false starts) can be
  validated against gold disfluency marks — no manual spot-check fallback needed.

## Bonus finding — NO L1 metadata: fairness plan revised

- The corpus paper states there was *"insufficient L1 and speaker data provided
  to incorporate this information into the data set selection"*; no per-file L1
  or demographic labels ship with the release.
- However, every utterance carries an **audio quality label** (Q2–Q5, QX) and
  every part a **CEFR grade band label** in the STM category tags.

**Implication:** fairness slices by L1 are not possible. The fairness analysis
pivots to the slices the data supports:
1. **Audio/recording quality** (Q2–QX) — does the scorer penalise poor recordings
   rather than poor speaking? (Was already a planned slice in the original design.)
2. **Proficiency band** — is scoring error uniform across A2–C1, or does the
   model systematically misjudge low/high performers?
3. **Test part / question** — per-part error parity.

This also simplifies subsetting: with no L1 stratification needed, the
**entire dev set (438 speakers) is the working dataset** — squarely inside the
design doc's 300–500 target, with no sampling decisions to defend.
