# Project journal — CEFR Spoken Response Scoring Lab

A Python evaluation lab that scores L2 English spoken responses from the Speak & Improve Corpus 2025 against human ratings, and evaluates the scorers themselves: correlation, weighted kappa, consistency, fairness slices, and ASR-error impact. The evaluation report is the product; scorer accuracy is a means. Built to close the speech-ML evaluation gap in my portfolio for language-assessment AI roles.

## 2026-06-05 — ASR error analysis and prosody (notebook 02)

Measured what notebook 01 could only infer: whisper-small scores **16.8% WER** against gold transcripts (65/100 sample responses are gold-covered), rising as proficiency falls (r = −0.28) — ASR noise concentrates on the weakest speakers, now a core fairness-analysis thread. Gold annotations confirm **100% filled-pause erasure** (333 spoken → 0 transcribed) and show the erased signal is predictive (r = −0.25). Swapping TTR for MTLD lifted QWK to **0.647** (best so far); five Praat prosody features added nothing at n = 100 — a reported null result, with `intensity_sd_db` flagged as a likely recording-gain confound. Also caught a CV reproducibility trap: row order silently changes fold assignment (±0.04 QWK at this n). Full detail: `docs/session-2026-06-05.md`.

## 2026-06-04 — Inception to first results

Designed, scoped and shipped the vertical slice in one session: approved design doc with five premises and three data-verification gates; corpus metadata inspected and all gates answered (0–6 score scale, single consensus labels, gold transcripts available, no L1 metadata → fairness pivoted to audio-quality and band slices); dev audio downloaded and verified (5,616 files, 438 speakers); end-to-end pipeline on 100 Part-3 responses producing **QWK 0.62 / r 0.67** (full Ridge) vs **QWK 0.58** for a word-count-only baseline; repo published at github.com/GavinReid82/cefr-speech-scoring-lab with licence-clean history verified. Full detail: `docs/session-2026-06-04.md`.
