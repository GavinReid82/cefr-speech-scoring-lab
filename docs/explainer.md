# Scoring the Scorer — the plain-English explainer

A computer listened to people speaking English and gave them a grade. This is the story of
checking how badly it got things wrong — and why that check was the point.

**876 spoken responses · 438 speakers · June–August 2026**

> **The short version.** We built a machine that grades spoken English. It is *okay*, not
> great. Then we spent most of the project proving exactly *how* it is wrong, and writing
> that down honestly.
>
> The grading machine was never the product. The honest report about the grading machine
> was the product.

This file is the non-technical companion to `reports/evaluation_report.md`,
`reports/model_card.md` and `reports/error_analysis.md`. Every figure here comes from those
three; nothing new is claimed. Where they are precise, this one is readable.

---

## The six terms you'll need to say out loud

| Short | Full term | How to say it | What it means |
|---|---|---|---|
| **CEFR** | Common European Framework of Reference for Languages | "SEF-er" | The standard English levels — A1 (beginner) up to C1 (advanced). Here they're numbers 1 to 5, in half-steps. |
| **ASR** | Automatic Speech Recognition | "AY-ESS-AR" | Speech-to-text. The software that turns audio into words on a page. Ours is Whisper, from OpenAI. |
| **QWK** | Quadratic Weighted Kappa | "Q-W-K" or "quadratic kappa" | One score for "how much does the machine agree with the human marker?" 0 = coin flip. 1 = perfect. *Weighted* because being wrong by two grades counts as far worse than being wrong by one. |
| **WER** | Word Error Rate | "W-E-R" or "word error rate" | How much the speech-to-text got wrong. 20% means roughly one word in five is missing, wrong, or invented. |
| **r** | Pearson correlation coefficient | "correlation of…" | Do two things move together? 0 = unrelated. 1 = locked together. Minus means one goes up as the other goes down. |
| **LoRA** | Low-Rank Adaptation | "LOR-ah" | A cheap way to retrain a big language model. You freeze almost all of it and nudge a tiny fraction — about 0.15% here — so it fits on a laptop. |

---

## 01 — The job: grade the speaking, then grade the grader

Cambridge released a research collection of real learners speaking English. Each recording
already has a grade from a **trained human marker**. That's rare and valuable — most
projects have to fake their labels.

So the plan: build a machine that guesses those grades, then measure the machine properly.
Does it agree with humans? Is it wrong in a fair way? Is it wrong because the speech-to-text
messed up?

> **Why we did it that way.** Almost every paper in this field competes on accuracy. Very
> few publish the boring, careful part — reliability, fairness, error analysis. That gap is
> what this project exists to fill. So accuracy was allowed to be mediocre, as long as the
> honesty was not.

---

## 02 — Before building: we checked three things first, and one answer changed the plan

Before writing any model code, we opened the corpus paperwork and asked three questions. It
took a day and saved weeks. (Written up in `reports/data_gates.md`.)

- **What scale are the grades on?** 0–6, in half-steps, roughly CEFR. Finer than we
  assumed. Good.
- **How many humans marked each one?** Only one final grade per response. That killed a
  planned measurement — you can't check "do two humans agree?" if there's only one human.
- **Are there perfect human transcripts to compare against?** Yes, and better than hoped —
  they even mark every "um" and "uh".

A fourth thing turned up by accident: **the corpus records nobody's first language.** That's
the single most important fairness question in language testing — does the machine punish
Spanish speakers more than Japanese speakers? We cannot answer it. Not "we didn't get round
to it". We *cannot*.

> **Why we did it that way.** We swapped the fairness plan to what the data actually
> supports: recording quality, ability level, and which task the person was doing. And we
> wrote the missing first-language slice into the report as a stated hole, rather than
> quietly not mentioning it.

---

## 03 — The machine: what it actually does

Three steps, no magic:

1. **Listen.** Whisper turns ~60 seconds of audio into words, with a timestamp on every word.
2. **Measure.** 17 numbers come out. How fast they talked. How long they paused. How many
   different words they used. Pitch wobble. That sort of thing.
3. **Guess.** A Random Forest — basically a big committee of simple yes/no flowcharts —
   turns those 17 numbers into a grade.

> **Why we did it that way.** Everything runs on a laptop. No cloud GPUs. And the features
> are hand-built in readable code rather than pulled from a black box, because you can't
> write an honest error analysis about a system you can't see inside.

### Two small decisions with receipts

We measure vocabulary richness with **MTLD** (Measure of Textual Lexical Diversity) instead
of the textbook method, because the textbook method scored **backwards**. It said people
with better vocabulary were worse. It was really just measuring length. Swapping it gained
**+0.026** agreement.

We threw away loudness. Loudness depends on the microphone, not the speaker. We kept only
how much the loudness *varied*.

---

## 04 — The first result: it looked good, so we made it look worse

First run on 100 responses: agreement with human markers of **0.647**. From here on,
"agreement" always means QWK. Nice result.

Then we ran it properly on all 876. It fell to **0.558**.

> **0.647 → 0.558.** Same code. Same features. More data. The first number was small-sample
> luck, and the public README now carries the lower one.

Two things caused the drop. More data is just harder. But also — we fixed a cheat.

Each person recorded **two** responses. If one lands in the practice pile and the other in
the test pile, the machine has effectively already met that person. We forced both of a
person's responses into the same pile.

> **Why we did it that way.** We didn't just assume the cheat mattered — we measured it. It
> was worth about **+0.015** on one model and roughly nothing on the Random Forest. That
> near-zero is itself informative: it means these features barely encode *who* the speaker
> is, only how good they are.
>
> The technical name for this is **speaker-grouped cross-validation**. Cross-validation just
> means "split the data into piles, train on most of them, test on the one you held back,
> repeat". Speaker-grouped means a person is never split across piles.

---

## 05 — The main flaw: it squashes everybody toward the middle

This is the finding that matters most, and it has nothing to do with speech-to-text.

The machine is **too kind to weak speakers and too harsh on strong ones.** It rarely says
anyone is very bad or very good. Everyone gets pushed toward average.

| Human grade | How many | Where the scorer put them | Error |
|---|---|---|---|
| 2 | 42 | ≈ 2.96 | **+0.96 too generous** |
| 3 | 290 | ≈ 3.36 | +0.36 |
| 4 | 412 | ≈ 3.88 | −0.12 |
| 5 | 132 | ≈ 4.30 | **−0.70 too harsh** |

Humans use the range 2 to 5. The scorer only ever really uses **3.0 to 4.3**. That narrow
band is the whole problem.

At grade 2, the error and the *average* error are the same size — which is only possible if
**every single grade-2 response was marked too high.** Not most. All 42.

> **81.5%** of guesses land within half a grade of the human. That sounds fine. It hides the
> fact that the misses are all in the same direction, at exactly the point where a pass/fail
> line would sit.

---

## 06 — The scare: "bad transcripts cause bad grades!" — no they don't

Obvious worry: when Whisper mishears someone, does that person get a wrong grade?

The first check said **yes**. Messy transcripts went with over-generous grades — a
correlation (*r*) of **+0.15** — and the statistics looked convincing.

It was fake. Here's the trap:

- Whisper struggles more with weaker speakers. (True.)
- The scorer is too generous to weaker speakers. (True — that's chapter 05.)
- So "messy transcript" and "over-generous grade" both just mean **"weak speaker"**.

Hold ability level constant, and the effect vanishes: **r = −0.01**. The speech-to-text was
innocent. It was ability wearing a disguise.

The name for this is **mediation** — when A looks like it causes B, but really C is quietly
causing both. The fix is a **partial correlation**: measure A against B again while holding
C still.

> **Why we did it that way.** Published as-is, that first number would have been a headline
> about AI unfairness that wasn't real. This is the single best argument for the whole
> project: the careful check is what stopped us publishing a fake alarm.

### Then we pushed on it twice more

**Once by error type.** Word Error Rate lumps all mistakes together. But the machine's
favourite feature is *how many words you said* — so words being *dropped* should hurt, while
words being *swapped* shouldn't. Split apart, dropped words do keep a small effect:
**−0.12**. And the minus sign matters — it means dropped words cause **under**-scoring, the
*opposite* of the original scare.

It's tiny — about 1% of the error, against a squashing problem worth nearly a whole grade.
So it changes the wording, not the conclusion.

**Once with a totally different machine** — see chapter 08. Same result.

---

## 07 — The embarrassing baselines: just counting words gets you 90% of the way

We kept a deliberately stupid model in every results table: **count the words, ignore
everything else.**

> **0.500 vs 0.558.** Word-counting scores 0.500. All seventeen carefully-built features
> score 0.558. Months of feature engineering bought **+0.058**.

That's not a failure, it's the honest measurement. It means the system is largely measuring
*how much someone talks*, which is a known property of fluency features and now a
written-down limitation rather than a secret.

> **Why we did it that way.** Dumb baselines appear in every single results table, on
> purpose. Compare a new model only to "guess the average" and everything looks brilliant.
> The rule in the report is now: beat word-counting, or you haven't done anything.

### And one thing nobody was measuring at all

The human transcripts mark every "um" and "uh". Across 600 responses there are **3,058** of
them.

Whisper transcribed **zero**. Not few — zero. It's built to tidy them away.

Hesitation is a real signal of proficiency (**r = −0.25** in the human transcripts). So this
is probably the biggest speech-to-text harm in the whole system — and **no error-rate score
charges anything for it**, because the standard comparison strips "um" out of both sides
before comparing.

---

## 08 — The obvious next idea: we tried a language model, and it lost

Fair objection to everything above: none of these features *read what the person said*. They
count pauses and words. Maybe all the conclusions are artefacts of that blindness.

So we took a small language model — Qwen2.5, 1.5 billion parameters — and fine-tuned it on
the transcripts using LoRA, on the laptop, so it outputs a grade. Same responses, same
rules, head to head.

| Scorer | Agreement (QWK) |
|---|---|
| **Random Forest — 17 features** | **0.558** |
| Ridge regression — 17 features | 0.548 |
| Ridge regression — word count only | 0.500 |
| Neural net — multi-layer perceptron | 0.475 |
| LoRA fine-tuned language model | 0.446 |
| Same language model, untrained | 0.144 |
| Guess the average every time | 0.000 |

All figures on identical held-out responses under identical rules. "Ridge regression" is a
straight-line fit; "Random Forest" is a committee of flowcharts.

It came **fifth**. Below the word-counting baseline. And it made the squashing problem
*worse* — **+1.03** and **−1.01** at the two ends, against the Random Forest's +0.96 / −0.70.

### But did the training do anything at all?

The first write-up couldn't answer that, because 0.446 had only ever been compared against
*other* systems. So we ran the same model with the training switched off. It scores **0.144**.

> **+0.302.** The training roughly tripled its performance. The model still lost — but it
> lost because of its ceiling, not because the training failed. Two very different stories,
> and only one is true.

> **Why we did it that way.** The model runs entirely on the laptop, never in the cloud. Not
> to save money — the corpus licence forbids sending the data to any AI service that might
> train on it. Local was the *legal* option, not just the cheap one. The trained weights
> stay off the internet for the same reason.

And the point of the whole exercise: this machine reads **nothing but the words**, and the
speech-to-text scare from chapter 06 *still* comes back innocent (+0.16 raw, −0.04 once
ability is controlled). The finding wasn't an artefact. Question closed.

---

## 09 — Two honest gaps

### A reliability score that couldn't exist

The plan said: run the AI grader twice, see if it agrees with itself. Reasonable.

Except our grader is fully deterministic — no randomness anywhere. Run it twice and you get
*bit-identical* answers. The score would have been a flawless **1.0**, it would have
satisfied the plan, and it would have measured **absolutely nothing**.

So we measured something real instead: reword the instructions harmlessly — same task, band
labels renamed — and see if it still agrees with itself. It doesn't, much. Renaming the
labels moves **60%** of its answers.

That's still useful, because agreeing with yourself is a *ceiling* on agreeing with anyone
else. This model's ceiling is **0.512** and it scores **0.144** against humans — so it's
nowhere near its own limit. Its problem is that it's wrong, not that it's wobbly.

### One thing we promised and never built

The original plan asked for examples of the machine's *written feedback*, labelled useful /
vague / unfair.

No feedback generator was ever built. Every part of this project outputs a number; nothing
outputs a sentence. So there was nothing to label.

> **Why we did it that way.** It's written into `docs/design.md` as a dated, signed-off cut
> — not deleted. Feedback quality is a genuine evaluation dimension and this project does
> not cover it. Leaving the promise visible next to the retraction is the whole habit this
> project is trying to practise.

---

## 10 — Where it landed: the final scorecard

| Question | Answer |
|---|---|
| Does it agree with human markers? | Moderately — QWK 0.558 |
| Exactly right? | 34.1% |
| Within half a grade? | 81.5% |
| Within a full grade? | 97.7% |
| Is the error even across ability? | **No — squashed to the middle** |
| Does bad speech-to-text bias grades? | No (checked three ways) |
| Does poor audio quality bias grades? | No sign of it — but thin data |
| Does first language bias grades? | **Unknown — data doesn't exist** |
| Worst-served speaker | **Off by 1.71 grades** |

Written up as three documents: a **model card** (what it is, what it must not be used for),
an **evaluation report** (every number, every caveat), and an **error analysis** (what it
does when it fails, and who to blame).

> **Why we did it that way.** The corpus licence forbids publishing any learner's words. So
> the error analysis can't do the normal thing — quote five bad transcripts next to their
> grades. It reports five *types* of failure with counts and averages instead. One type —
> "speech-to-text produced nothing at all" — turned out to have **zero** cases, and is
> reported as zero rather than dropped. A failure everyone assumes exists, that doesn't, is
> a finding.

---

## The bottom line: the best result here is a "no"

The grader is mediocre and the fancy upgrade lost to word-counting. Both are written down
plainly, with the numbers.

What the project actually produced is a set of results that would have been reported wrongly
by anyone moving faster: a fake fairness alarm that dissolved under one control, a
self-consistency score that would have read as perfect while measuring nothing, a promising
first number that shrank when the data grew, and a "biggest problem" — grade squashing —
that no accuracy headline shows and no speech-to-text improvement would fix.

None of that is a leaderboard win. It is the report that leaderboard wins usually skip.

---

**Data:** Speak & Improve Corpus 2025 (Cambridge University Press & Assessment / ELiT),
research licence. No corpus audio, transcripts or per-speaker data appear in this document
or the public repository.

**System:** whisper-small → 17 fluency & prosody features → Random Forest. All figures
out-of-fold, speaker-grouped 5-fold cross-validation.

**Not for use in any real assessment decision about any real candidate.**
