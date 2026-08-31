# Design: CEFR Spoken Response Scoring Lab

Status: APPROVED 2026-06-04 · gate outcomes resolved same day (see `reports/data_gates.md`)
· amended 2026-08-27 (premise 5) and 2026-08-31 (success criteria) · all roadmap items closed

## Problem statement

Automated speaking assessment research tends to compete on grader accuracy — leaderboard RMSE from end-to-end models. What rarely gets published is the evaluation science around a scorer: reliability, validity, fairness, and error analysis. This project builds a small, laptop-scale evaluation lab over the Speak & Improve Corpus 2025 (Cambridge): score short L2 English spoken responses against real human CEFR-aligned ratings, and produce the rigorous evaluation report that scorers usually lack.

The scorer is a means; the evaluation report is the product.

## Why this corpus

- **Real human ratings** from trained annotators on Linguaskill-derived criteria — weighted kappa, correlation and consistency analyses are genuine measurement, not pseudo-labels from an LLM.
- **Gold transcripts** in three forms (disfluent, fluent, grammar-corrected) plus word-level disfluency marks — ASR errors and their propagation into features can be quantified exactly.
- **Per-utterance audio-quality labels and per-part CEFR bands** — fairness slices the data actually supports.

## Premises

1. **DATA** — Use the corpus dev set, not self-recorded audio. (Original plan to record 30–100 responses was dropped at design review: real labels beat convenient data.)
2. **LICENCE** — The corpus is research-licensed. The public repo contains code, notebooks, reports and aggregate figures only. `data/` is gitignored from commit zero; committed notebook outputs show aggregates only.
3. **HAND-BUILT FIRST** — Feature extraction and models are built by hand in readable `src/` code. The corpus's published baseline systems are a comparison reference to read and cite, not a pipeline to run as a black box.
4. **DELIVERABLE** — The evaluation report: model–human agreement, intra-scorer consistency, fairness slices, ASR-error→score-error analysis. Scorer accuracy is not the goal.
5. **SCALE** — Laptop-scale throughout: Whisper small/medium for ASR, librosa/parselmouth for acoustics, scikit-learn plus a small PyTorch MLP. wav2vec2 embeddings + regression head is the only stretch goal; no fine-tuning of large models. — amended 2026-08-27, see Amendments

## Amendments

### 2026-08-27 — Premise 5: LoRA fine-tuning admitted

**What changed** — a fifth scoring arm is added: a LoRA-fine-tuned transcript scorer, `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, fine-tuned locally with `mlx-lm` on Apple Silicon.

**Why it is still laptop-scale** — LoRA rank 8 over the top 8 layers trains roughly 0.15% of parameters; the base weights are 4-bit. A smoke run on the 0.5B model measured **2.07 GB peak memory**. There is no cloud GPU and no full-parameter training, so the spirit of premise 5 — laptop-scale throughout — is unbroken. What is retired is only the clause "no fine-tuning of large models".

**Why it is necessary** — Recommendation 3 of `reports/evaluation_report.md`. Every existing arm is content-blind (fluency, prosody, word counts), so the report's ASR-propagation null result is *conditional* on that blindness. Testing it needs a scorer that reads what the candidate actually said.

**Why local, not an API** — the corpus licence says "do not share with LLMs with training retention". A local model is therefore the **compliant** architecture, not merely the affordable one. The licence also forbids releasing corpus-derived models without approval, so adapter weights stay local and gitignored.

**Premise 3 is unaffected** — mlx-lm's trainer is used as a tool (as Whisper and scikit-learn already are), while the ordinal band decode, the prompt contract and the evaluation stay hand-built in `src/speechlab/` (`ordinal.py`, `llm_scorer.py`).

**Protocol is unchanged** — the same speaker-grouped `GroupKFold(5)` over the same 876 P3+P4 dev responses, scored by the same `speechlab.evaluation`. Comparability with the published Random Forest baseline is the point.

**Outcome (added when the arm had run)** — the amendment was worth making and the arm lost. QWK 0.446 (argmax) / 0.415 (expected) against the Random Forest's 0.558 on identical rows, below the word-count floor of 0.500; the ASR-propagation null replicated under it, which is what the amendment existed to test. Premise 5 stays retired rather than reinstated — the question it blocked turned out to be answerable at laptop scale, and the answer is a reported negative result. See notebook 05 and `docs/session-2026-08-27.md`.

### 2026-08-31 — Success criteria: consistency measured, feedback examples retired

**What changed** — the three success criteria that were still open are closed, two of them
by delivery and one by retirement.

**1. `error_analysis.md` is delivered, with its "five concrete cases" clause reinterpreted.**
The criterion asked for five concrete ASR-error→score-error *cases*. The licence forbids
committing transcript text, utterance ids or speaker ids, so the conventional form of that
deliverable — five bad transcripts quoted beside their scores — cannot be written. The
report delivers five **case classes** instead: each defined by a measurable property of the
ASR output, cut at the sample's own deciles, and reported with its count, its mean WER, its
mean human score and the scorer's behaviour on it. One class (total ASR failure) turns out
to be empty at n = 876, and is reported as empty rather than dropped.

The criterion was also written before the answer was known, and the answer came back null:
there is no population of responses where ASR error demonstrably caused a scoring error,
because the propagation effect is mediated by proficiency. Five *examples* of a
non-existent effect would have been five coincidences. What the file reports instead is the
anatomy of the failure that does exist, plus one analysis the evaluation report never ran —
propagation by error *type* rather than by aggregate WER — which turns up a small residual
for deletions in the opposite direction to the original alarm.

**2. "Labelled feedback examples" is retired.** The criterion asked for scorer-generated
feedback tagged useful / vague / unfair / overconfident. **No feedback generation was ever
built.** Every arm in this project emits a number; none emits prose, and none was ever
planned to after the design settled on regression and ordinal decoding. This is a genuine
reduction in scope rather than a reinterpretation: feedback quality is a real evaluation
dimension and this project does not cover it. It is recorded here rather than quietly
dropped so that the gap between what was designed and what was built stays visible.

**3. Consistency is measured, but not as specified.** The Evaluation design defines it as
QWK between two independent LLM rubric-scoring passes, exempting deterministic models
because a repeat run cannot disagree with itself. **The LLM arm turned out to be exempt on
the same grounds**, which the design did not anticipate: `LoRABandScorer` takes one forward
pass and reads eight logits at fixed token ids, so a second pass is bit-identical and
pass-to-pass QWK is 1.0 by construction. That number measures determinism, not reliability.

The substitute is agreement across a benign **rewording** of the prompt — the two zero-shot
passes of notebook 05 §4, blank band letters against the same letters anchored to CEFR
names. Pass-to-pass QWK 0.512 (expectation decode) against 0.144 with humans, so the arm
sits a quarter of the way to the ceiling its own instability imposes: a validity problem,
not a reliability one. Reported in §5 of `evaluation_report.md`.

**4. Roadmap item 8 is out of scope.** wav2vec2 embeddings plus a regression head, and SHAP
attribution, were the stated stretch goal. Neither is built. The wav2vec2 arm is the weaker
version of a question notebook 05 already answered at greater cost — whether a learned
representation beats hand-built fluency features — and it came back negative there. SHAP on
a 17-feature Random Forest whose dominant feature is already known to be word count would
confirm what §4 of the evaluation report states from the word-count baseline directly.

**What is unaffected** — every metric, protocol and finding. This amendment changes what the
project claims to have delivered, not any number.

## Approaches considered

**A — Vertical-slice notebook lab.** Dev set only, ~100 responses, one notebook end-to-end. Fast first result, thin artefact.

**B — Full evaluation lab (chosen).** `src/` modules (asr, acoustic_features, text_features, scoring_models, evaluation, fairness), staged notebooks, `reports/` (model card, evaluation report, error analysis). Scorers: dumb baselines → LLM rubric → Ridge → Random Forest → PyTorch MLP (XGBoost optional). Built vertical-slice-first: approach A is literally week one, then refactored into modules.

**C — Scorer-pluggable benchmark harness.** Evaluation engine first, scorers as plugins, auto-regenerated reports. Strongest evaluation-science narrative, but harness plumbing displaces the feature/model work this project exists to practise.

**Decision: B, built in A's order, stealing C's one killer idea — dumb baselines (word count, speech rate, zero-shot LLM) appear in every results table, always.** The honest measure of any pipeline is what it buys over the trivial alternative.

## Verification gates (resolved)

Three assumptions were gated on metadata inspection before any modelling commitment:

1. **Score scale** — confirmed: 0–6 ≈ CEFR (1=A1 … 5=C1), per-part manual scores at 0.5 intervals, overall = mean of four parts.
2. **Label structure** — single consensus label per response; no multi-rater data ships. Inter-rater reliability is therefore not computable: the deliverable is model–human agreement plus intra-scorer consistency (QWK between two independent LLM scoring passes).
3. **Gold transcripts** — available, in richer form than required.

**Revision from gating:** no L1 metadata ships with the corpus, so fairness slices pivot from L1 to audio quality, proficiency band, and test part. With no stratification needed, the full dev set (438 speakers, 5,616 utterances) is the working dataset.

## Evaluation design

- **Agreement:** Pearson/Spearman correlation, MAE/RMSE on the continuous overall score, quadratic weighted kappa on the 0.5-step ordinal grid, confusion matrix by CEFR band.
- **Consistency:** QWK between two independent LLM rubric-scoring passes (deterministic feature models need no repeat). — amended 2026-08-31: the LLM arm is deterministic too, so this is delivered as agreement across a reworded prompt; see Amendments.
- **Fairness:** per-slice score-gap (mean predicted-score difference vs pooled mean) and error-gap (per-group MAE difference vs pooled MAE), every metric reported with its slice *n*, findings framed as indicative at this sample size.
- **Error analysis:** WER against gold transcripts; documented cases where ASR errors propagated to scoring errors; labelled examples of useful / vague / unfair / overconfident feedback.
- **Known hazards, planned for:** Whisper suppresses filled pauses (quantified against gold disfluency marks rather than patched); Praat-based pitch extraction on noisy L2 audio needs voiced-frame filtering and sanity clamps — features get dropped rather than shipped noisy.

## Success criteria

- ✓ `evaluation_report.md` with all agreement, consistency and fairness metrics for **every** scorer, dumb baselines included. — consistency delivered as prompt-variant agreement, see the 2026-08-31 amendment.
- ✓ `error_analysis.md` with at least five concrete ASR-error→score-error cases ~~and labelled feedback examples~~. — five case *classes* rather than quoted cases, for licence reasons; the feedback clause is retired, see the 2026-08-31 amendment.
- ✓ `model_card.md` covering intended use, data provenance and licence, metrics, fairness findings, limitations.
- ✓ Pre-push licence checklist passes: no corpus audio, transcripts or per-speaker rows anywhere in git history; notebook outputs aggregate-only; malformed/silent responses logged and excluded explicitly. — enforced by `scripts/licence_scan.py`, run against the corpus itself rather than by eye. Two historical blobs (a real utterance id in an explanatory comment, committed June 2026, fixed forward 31 August) remain in the object graph; see the 2026-08-31 session report.

## Roadmap

1. ~~Corpus access, repo skeleton, licence rules~~ ✓
2. ~~Metadata gates~~ ✓
3. ~~Vertical slice: 100 P3 responses → Whisper → fluency features → Ridge → first QWK~~ ✓ (see `notebooks/01_vertical_slice.ipynb`)
4. ~~Notebook 02: WER vs gold transcripts; acoustic/prosody features (pitch, intensity) from raw audio~~ ✓
5. ~~Refactor into `src/` modules with pytest fixtures; scale to the full dev set~~ ✓
6. ~~Notebook 03: text features (MTLD, embedding relevance, LLM-derived grammar/coherence with JSON-schema validation), Random Forest, PyTorch MLP~~ ✓
7. ~~Notebook 04 + reports: full evaluation, fairness slices, error analysis, model card~~ ✓
8. ~~Stretch: wav2vec2 embeddings + regression head; SHAP feature attribution~~ — **out of scope**, see the 2026-08-31 amendment
9. ~~Notebook 05: LoRA transcript scorer arm (`mlx-community/Qwen2.5-1.5B-Instruct-4bit` via `mlx-lm`); ASR-propagation re-run~~ ✓
10. ~~`reports/error_analysis.md`; intra-scorer consistency; propagation by ASR error type~~ ✓

All roadmap items are closed. Outstanding *research* follow-ups — the clean
transcript-quality control, a best-checkpoint fallback, decode calibration on an inner
split, and scale above 1.5B — are listed as future work in §9 of
`reports/evaluation_report.md`, not as unfinished deliverables.
