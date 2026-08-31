# Error analysis — where the scorer fails, and whether ASR is to blame

**System:** whisper-small ASR → 17 fluency/prosody features → Random Forest, plus the
LoRA transcript scorer of §4 of `evaluation_report.md` as an independent check
**Data:** Speak & Improve Corpus 2025, scored long-turn dev set (876 responses, 438
speakers); the ASR analyses use the **600** of those with gold transcripts
**Companion:** `evaluation_report.md` §5–§6 (the summary form of everything here),
notebooks 02, 04 and 05 (the working), `scripts/error_analysis.py` (regenerates every
aggregate below)

---

## 1. Why this file exists separately

`evaluation_report.md` answers *how well does the scorer agree with humans*. This one
answers *what does it do when it is wrong, and does the ASR cause it*. The two questions
share a dataset and nothing else: the first is served by aggregate agreement statistics,
and the second is destroyed by them — every finding below is invisible in QWK 0.558.

It also carries one analysis that appears nowhere else. §6 of the evaluation report tests
whether **overall WER** propagates into scoring error and finds that it does not. That is
not the same as testing whether a particular **kind** of ASR error propagates, and the
kinds are not interchangeable: `n_words` is the Random Forest's dominant feature, so a
transcript that loses words loses the feature the scorer leans on hardest, while a
transcript that swaps one word for another leaves it untouched. §4 below runs that test.

**On the licence.** The corpus is research-licensed and this file is committed, so there
are no transcript excerpts here and no utterance or speaker identifiers — the usual way to
write an error analysis, quoting five bad transcripts beside their scores, is exactly what
the licence forbids. Cases are therefore stated as **classes with counts**: a class is
defined by a measurable property of the ASR output, and every figure attached to it is a
mean over at least seven responses. Nothing below identifies a response.

## 2. The scorer's own failure shape

Before ASR is implicated in anything, this is what the errors look like without it.

| Band | n | MAE | Signed error |
|---|---|---|---|
| 2 | 42 | 0.964 | **+0.964** |
| 3 | 290 | 0.455 | +0.361 |
| 4 | 412 | 0.327 | −0.120 |
| 5 | 132 | 0.701 | **−0.701** |

At band 2 the signed error equals the MAE, which means **every band-2 response is
over-scored**. This is regression to the mean on an imbalanced score distribution, and it
is the single largest error term in the system — roughly a full band. Every ASR effect
measured in this file is an order of magnitude smaller. Per speaker (mean of two
responses): median MAE 0.39, 90th percentile 0.83, worst 1.71.

The consequence for everything that follows: **any variable correlated with proficiency
will appear to cause scoring error**, because proficiency causes scoring error. WER is
such a variable (r = −0.28 against score). So is deletion rate. So is nearly every ASR
quality measure. Mediation is not a subtlety here, it is the default state, and an
analysis that skips the control will report a fabricated effect in the right direction
with a convincing p-value.

## 3. What the ASR errors actually are

whisper-small against the gold disfluent transcripts, lenient normalisation, 600
responses. Corpus WER — total errors over total gold words — is **16.6%**; per-response
mean 20.9%, median 12.5%. (Notebook 02 reports 16.8% / 22.2% / 12.1% for the same
measures over the 65-response notebook-01 sample; §6 of the evaluation report quotes those.
The gap between the two is what a 65-response sample is worth on a right-skewed quantity.)

| | mean | median | p90 |
|---|---|---|---|
| WER (lenient) | 0.209 | 0.125 | 0.420 |
| substitution rate | 0.067 | 0.055 | 0.138 |
| deletion rate | 0.049 | 0.038 | 0.108 |
| insertion rate | 0.093 | **0.013** | 0.266 |
| length ratio (hyp/gold) | 1.044 | 0.985 | 1.245 |

All three rates share one denominator — gold words — so they are comparable and sum to the
lenient WER. By share of total errors: substitutions **36.7%**, insertions **33.9%**,
deletions **29.4%**.

The insertion row is the one that does not behave. Its mean is seven times its median, and
the maximum insertion rate is **6.15** — one response for which Whisper emitted roughly
seven times as many words as were spoken. That is not transcription noise, it is the
model's known looping failure on sparse or noisy audio, and it means "insertion rate"
describes two different populations: a large majority near zero and a tail where the
transcript is mostly generated text.

## 4. Does ASR error propagate into scoring error?

The analysis runs in three steps, and the first two are in §6 of the evaluation report.

**Step 1 — the alarm.** WER correlates with *signed* scoring error, r = **+0.151**
(p = .0002). Read naively: badly-transcribed responses get over-scored.

**Step 2 — the mediation.** Controlling for the human score, the partial correlation is
**−0.014** (p = .73). High-WER speakers are low-proficiency speakers, and low-proficiency
speakers are over-scored by band compression. The apparent ASR effect was proficiency
wearing a hat.

**Step 3 — error type, which step 2 did not test.** Repeating step 2 for each error type
separately:

| Variable | Raw r vs signed error | Partial (human score held) |
|---|---|---|
| WER (lenient) | +0.151 (p = .0002) | −0.014 (p = .733) |
| substitution rate | +0.242 (p < .0001) | **−0.101** (p = .013) |
| deletion rate | +0.082 (p = .044) | **−0.119** (p = .004) |
| insertion rate | +0.107 (p = .009) | +0.017 (p = .683) |
| length ratio | +0.094 (p = .022) | +0.032 (p = .431) |

Two things changed. First, **deletions and substitutions do show a residual effect where
overall WER shows none** — the aggregate WER figure was averaging an effect against
nothing and getting nothing. Second, and more usefully, **the surviving effect points the
other way**: the partial correlations are *negative*, meaning that once proficiency is
held constant, a transcript with more words dropped is **under**-scored. The raw alarm in
step 1 warned about over-scoring. The real residual effect is the opposite sign.

The mechanism is not mysterious and it is the reason the test was worth running: deletions
shorten the transcript, `n_words` is the dominant feature, and a shorter response looks
like a weaker one to this scorer. Substitutions plausibly ride along with deletions on the
same badly-recorded responses.

**How much of this to believe.** Not a great deal, and the size is the reason. Both
surviving correlations sit near r = −0.11, about **1% of the variance in signed error**,
against a band-compression term of nearly a full band. Five variables were tested, so at
α = .05 one false positive is expected; deletion rate (p = .004) survives a Bonferroni
correction across the five, substitution rate (p = .013) does not. The partial correlation
also assumes the score–error and score–WER relationships are linear, which is an
approximation at the distribution's edges.

**It replicates on a second scorer, though.** Running the same test against the LoRA
transcript scorer — a model that reads nothing but the words, so a different failure
mechanism entirely — gives deletion rate vs signed error: raw +0.119 (p = .003), partial
**−0.114** (p = .005). Same sign, near-identical magnitude. The rows are the same 600, so
this is not independent data; it is an independent *scorer* over shared data, which rules
out a feature-pipeline artefact but not a property of the responses themselves.

**Where that leaves §6.** Its conclusion stands as an operational statement — there is no
ASR-induced scoring bias here worth correcting for, and certainly none of the size the raw
correlation advertised. What needs amending is the strength of the phrasing: the null
holds for aggregate WER, and *not quite* for deletion, where a small effect in the
opposite direction to the alarm is detectable at n = 600.

## 5. The five case classes

Classes are cut at the sample's own deciles rather than round thresholds, because a fixed
cut would define the classes by whisper-small's absolute error level and every figure in
this file is already specific to that model. Rows overlap; they are not a partition.

| Class | n | mean WER | mean human score | signed error | MAE |
|---|---|---|---|---|---|
| **1. Truncation** — top-decile deletions | 60 | 0.287 | 3.62 | +0.054 | 0.419 |
| **2. Corruption** — top-decile substitutions | 60 | 0.465 | 3.35 | +0.259 | 0.460 |
| **3. Hallucination** — top-decile insertions | 60 | 0.820 | 3.59 | +0.209 | 0.459 |
| **4. Compound** — top-decile on both 1 and 2 | 7 | 0.472 | 3.07 | +0.088 | 0.365 |
| **5. Total failure** — empty transcript | **0** | — | — | — | — |
| *control:* clean — bottom-quartile WER | 150 | 0.049 | 4.33 | −0.245 | 0.463 |
| *all gold-covered* | 600 | 0.209 | 3.98 | −0.028 | 0.455 |

**Read the signed-error column against the human-score column, not against zero.** Every
badly-transcribed class is also a low-proficiency class — 3.07 to 3.62 against the clean
class's 4.33 — and band compression over-scores low-proficiency responses regardless of
their transcript. The table is descriptive; §4's partial correlations are the test. The
clean class being *under*-scored by 0.245 makes the point: its transcripts are nearly
perfect and its errors are larger in the other direction, because it is the strong-speaker
class and strong speakers are compressed downwards.

What each class is, and what it does to a scorer:

1. **Truncation.** Words present in the audio never reach the transcript. This is the
   class that should matter most and, per §4, the only one whose residual effect survives
   correction — it attacks `n_words` directly. MAE is nonetheless the *lowest* of the four
   at 0.419, which is the whole lesson of this file in one number.
2. **Corruption.** Content words replaced by other content words, length roughly intact.
   Invisible to a content-blind feature set by construction — the word count is unchanged
   — and, per §4, its residual effect does not survive multiple-comparison correction.
3. **Hallucination.** Whisper loops or invents on sparse audio; mean WER 0.820 in this
   class, with a maximum insertion rate of 6.15 across the set. This is the most
   *dramatic* failure and the one with **no residual effect at all** (partial r = +0.017,
   p = .68). Inflated transcripts inflate `n_words`, which should over-score — the
   over-scoring in the table is fully accounted for by the class's low proficiency.
4. **Compound.** Both at once, n = 7. Too small for inference and reported only because a
   class defined by co-occurrence is the one a reader will ask about; the honest reading
   of a 7-response cell is that it exists and nothing more.
5. **Total failure.** Whisper returned an empty transcript for **zero** of the 876
   responses. The modelling pipeline carries an `n_words > 0` filter for this case and it
   never fires on this data. Reported because a case class that turns out to be empty is a
   finding — the failure mode most likely to be assumed is absent here.

## 6. The error no WER figure charges for

Across the 600 gold-covered responses the annotations mark **3,058 filled pauses**
(`hesitation` tags) and 2,560 disfluency-marked words. Whisper's output contains **0**
filled pauses. The erasure is total, and it is confirmed against annotation rather than
inferred from the model's own silence.

This does not appear in the lenient WER at all — the lenient reference strips hesitations
precisely so that content accuracy can be measured separately — and it appears in the
strict WER only as a **+4.1 point** average penalty. Neither figure describes the actual
loss, which is that a fluency scorer fed Whisper transcripts cannot see hesitation. The
erased signal is predictive: gold disfluency rate correlates with score at r = −0.25
(p = .046), hesitation rate at r = −0.21.

So the largest ASR-attributable harm in this system is **not** in any of §5's classes. It
is uniform across every response, invisible to WER, and consists of a cue that was never
transcribed rather than one transcribed wrongly. An error analysis organised only around
WER would have missed it entirely, which is why it has its own section here.

## 7. On labelled feedback examples

`docs/design.md` originally paired this file's case analysis with "labelled feedback
examples" — instances of scorer-generated feedback tagged useful / vague / unfair /
overconfident. **No feedback generation was ever built.** Every arm in this project emits
a number, and none emits prose, so there is nothing to label. This is recorded as a
retired criterion in the 2026-08-31 amendment to `docs/design.md` rather than quietly
dropped, and it is a genuine reduction in scope: feedback quality is a real evaluation
dimension and this project does not cover it.

## 8. Limitations

- **One ASR system, one size.** Every figure is whisper-small. Error composition — the
  36.7 / 33.9 / 29.4 split, and the insertion tail in particular — is a property of that
  model, and the class definitions inherit it.
- **600 of 876.** The gold transcripts cover 68% of the modelled responses. Nothing here
  is weighted or reweighted to the full set.
- **Small effects at moderate n.** The §4 residuals explain about 1% of signed-error
  variance. They are reported because their sign is informative, not because they are
  large.
- **Shared rows across scorers.** The LoRA replication uses the same 600 responses, so it
  is an independent scorer, not an independent sample.
- **The dominant error is not an ASR error.** Band compression is roughly ten times the
  size of anything measured here, and no ASR improvement addresses it.
