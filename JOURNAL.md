# Project journal — CEFR Spoken Response Scoring Lab

A Python evaluation lab that scores L2 English spoken responses from the Speak & Improve Corpus 2025 against human ratings, and evaluates the scorers themselves: correlation, weighted kappa, consistency, fairness slices, and ASR-error impact. The evaluation report is the product; scorer accuracy is a means. Built to close the speech-ML evaluation gap in my portfolio for language-assessment AI roles.

## 2026-06-04 — Inception to first results

Designed, scoped and shipped the vertical slice in one session: approved design doc with five premises and three data-verification gates; corpus metadata inspected and all gates answered (0–6 score scale, single consensus labels, gold transcripts available, no L1 metadata → fairness pivoted to audio-quality and band slices); dev audio downloaded and verified (5,616 files, 438 speakers); end-to-end pipeline on 100 Part-3 responses producing **QWK 0.62 / r 0.67** (full Ridge) vs **QWK 0.58** for a word-count-only baseline; repo published at github.com/GavinReid82/cefr-speech-scoring-lab with licence-clean history verified. Full detail: `docs/session-2026-06-04.md`.
