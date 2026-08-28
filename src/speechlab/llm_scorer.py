"""LoRA transcript scorer: prompt construction, fold export, and band-logit scoring.

The model-facing half of the ordinal scorer. `speechlab.ordinal` owns the maths and
stays numpy-only; everything here touches a tokenizer or MLX, so it is imported lazily
and kept out of the test path (tests use a stub tokenizer, never a download).

The design in one line: fine-tune a small instruct model to answer with **one letter**,
then read the band distribution off the logits at that single position instead of
generating text. One forward pass per response, no sampling, no parsing, and
`ordinal.expected_score` turns the distribution into a continuous score that drops
straight into `evaluation.evaluate` beside Ridge and Random Forest.

Two constraints are load-bearing and easy to violate silently:

1. **Prompt parity.** `mlx_lm.tuner.datasets.CompletionsDataset` builds exactly
   `[{"role": "user"}, {"role": "assistant"}]` from each JSONL row, and with
   `--mask-prompt` the loss starts at
   `apply_chat_template(messages[:-1], add_generation_prompt=True)`. Inference must
   therefore rebuild *exactly one user message* with `add_generation_prompt=True` and
   **no system message**. A system prompt at inference raises no error — QWK just sags.
   `assert_prompt_parity` checks this against the real tokenizer rather than trusting it.

2. **Single-token labels.** The whole scheme collapses if a label tokenises to more than
   one id, because then the band distribution is spread over positions rather than
   sitting in one logit row. `band_token_ids` asserts it.
"""

import json
from pathlib import Path

import numpy as np

from speechlab.ordinal import BandCodec, argmax_score, band_probabilities, expected_score

# Small enough to fine-tune and run on laptop unified memory, instruct-tuned so the chat
# template exists, 4-bit so the base weights stay ~0.4 GB. The corpus licence forbids
# sharing learner data with LLMs that retain it for training, so a *local* model is the
# compliant architecture here, not merely the affordable one.
DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# Whisper returns nothing for some responses (silence, or audio the model rejects).
# Those rows are still scored by humans, so they must reach the model as a real prompt
# rather than an empty string — an empty transcript is itself evidence about proficiency.
EMPTY_TRANSCRIPT = "(no speech transcribed)"


def transcript_text(words: list[dict]) -> str:
    """Whisper word list -> the surface string the model reads.

    Deliberately *not* `text_features.normalise_tokens`: that strips case and punctuation
    because the count-based features need stable types, whereas a language model can use
    exactly what normalisation throws away. Whisper's own punctuation is a fluency signal.
    """
    return " ".join(w["w"] for w in words).strip()


def build_prompt(transcript: str, part: str, codec: BandCodec) -> str:
    """The single user message. Must be byte-identical between training and inference.

    Letters, not CEFR names: 'B2' carries pretrained baggage that the fine-tune would
    have to fight, whereas A..H arrives blank and only has to learn the ordering. The
    ordering is stated in the prompt so the labels are not arbitrary from the start.
    """
    text = (transcript or "").strip() or EMPTY_TRANSCRIPT
    lo, hi = codec.labels[0], codec.labels[-1]
    return (
        "Rate the spoken English proficiency of this transcribed learner response.\n\n"
        f"Test part: {part}\n"
        f"Transcript: {text}\n\n"
        f"Answer with a single letter from {lo} (lowest) to {hi} (highest)."
    )


def build_record(transcript: str, part: str, score: float, codec: BandCodec) -> dict:
    """One JSONL row in the {"prompt", "completion"} format CompletionsDataset expects."""
    return {
        "prompt": build_prompt(transcript, part, codec),
        "completion": codec.encode([score])[0],
    }


def write_jsonl(path: Path, rows, codec: BandCodec) -> int:
    """Write prompt/completion records for `rows` — (transcript, part, score) triples.

    Returns the number of records written, so callers can log fold sizes without
    re-reading the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for transcript, part, score in rows:
            f.write(json.dumps(build_record(transcript, part, score, codec)) + "\n")
            n += 1
    return n


def write_fold_data(data_dir: Path, train_rows, valid_rows, codec: BandCodec) -> dict:
    """Lay out one fold the way `mlx_lm.lora --data` expects: train.jsonl + valid.jsonl.

    valid.jsonl is optional to mlx-lm (it warns and skips evaluation), but a held-out
    slice of *training* speakers is the only honest way to watch for overfitting: the
    test fold must stay untouched or the out-of-fold predictions stop being out-of-fold.
    """
    data_dir = Path(data_dir)
    counts = {"train": write_jsonl(data_dir / "train.jsonl", train_rows, codec)}
    if valid_rows is not None:
        counts["valid"] = write_jsonl(data_dir / "valid.jsonl", valid_rows, codec)
    return counts


def band_token_ids(tokenizer, codec: BandCodec) -> list[int]:
    """Vocabulary ids of the band labels, asserting each is exactly one token.

    Encoded without special tokens, which is how the label appears in the chat template:
    immediately after the assistant turn header, with no leading space.
    """
    ids = []
    for label in codec.labels:
        enc = tokenizer.encode(label, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(
                f"label {label!r} tokenises to {len(enc)} tokens ({enc}) — the single-position "
                "band distribution assumes one token per label; pick a different alphabet"
            )
        ids.append(int(enc[0]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"band labels collide in the vocabulary: {ids}")
    return ids


def load_prompt_tokenizer(model_path: str = DEFAULT_MODEL):
    """Tokenizer only, no weights — enough to check parity before committing to a run.

    `mlx_lm.load` would pull the model into memory for what is a pure text question, and
    its `TokenizerWrapper` only delegates `encode`/`apply_chat_template` to this object
    anyway. Going through transformers keeps the check cheap and independent of mlx-lm
    internals, which move between releases.
    """
    from transformers import AutoTokenizer  # lazy: heavy, and only needed for a real run

    return AutoTokenizer.from_pretrained(model_path)


def inference_tokens(tokenizer, prompt: str) -> list[int]:
    """Exactly what the model sees at scoring time: one user message, generation prompt on."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=False,
    )


def assert_prompt_parity(tokenizer, codec: BandCodec, prompt: str, label: str) -> None:
    """Fail loudly if inference would diverge from `--mask-prompt` training.

    Reconstructs both sides through the real chat template and checks that the inference
    token sequence is exactly the masked prefix of the training sequence, and that the
    first *unmasked* training token is the label. That is the whole contract: the model
    is trained to put the band in the position inference reads.
    """
    train = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": label}],
        return_dict=False,
    )
    infer = inference_tokens(tokenizer, prompt)
    if train[: len(infer)] != list(infer):
        raise ValueError(
            "inference prompt is not a prefix of the training sequence — the chat template "
            "renders the two paths differently; check for a stray system message"
        )
    expected = band_token_ids(tokenizer, codec)[codec.labels.index(label)]
    if train[len(infer)] != expected:
        raise ValueError(
            f"first unmasked training token is {train[len(infer)]}, expected the label "
            f"token {expected} for {label!r}"
        )


class LoRABandScorer:
    """Scores transcripts by reading the band distribution off one logit row.

    Usage mirrors the sklearn arms only loosely — there is no `fit` here, because
    training runs through the `mlx_lm.lora` CLI in `scripts/finetune_lora.py` (its
    trainer, schedulers and checkpointing are not worth reimplementing). This class is
    the predict half: point it at a base model plus a fold's adapter and it returns
    both decodes for the same forward pass.
    """

    def __init__(self, codec: BandCodec, model_path: str = DEFAULT_MODEL,
                 adapter_path: str | None = None, temperature: float = 1.0,
                 batch_size: int = 8):
        self.codec = codec
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.temperature = temperature
        self.batch_size = batch_size
        self.model_ = None
        self.tokenizer_ = None

    def load(self) -> "LoRABandScorer":
        """Load base weights plus (optionally) one fold's adapter. Idempotent per instance."""
        from mlx_lm import load  # lazy heavy import, per scoring_models.py

        if self.model_ is None:
            self.model_, self.tokenizer_ = load(self.model_path, adapter_path=self.adapter_path)
            self.band_ids_ = band_token_ids(self.tokenizer_, self.codec)
        return self

    def band_logits(self, prompts: list[str]) -> np.ndarray:
        """(n_prompts, n_bands) logits gathered at each prompt's final position.

        Padding is on the right, which is safe precisely because attention is causal:
        position i never sees i+1, so pad tokens appended after a prompt cannot reach the
        position we read. Left-padding would need an attention mask the plain forward
        call does not take.
        """
        import mlx.core as mx

        self.load()
        pad = getattr(self.tokenizer_, "pad_token_id", None)
        if pad is None:
            pad = self.tokenizer_.eos_token_id

        out = np.empty((len(prompts), len(self.codec)), dtype=float)
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            toks = [list(inference_tokens(self.tokenizer_, p)) for p in chunk]
            width = max(len(t) for t in toks)
            padded = mx.array([t + [pad] * (width - len(t)) for t in toks])
            logits = self.model_(padded)
            last = mx.array([len(t) - 1 for t in toks])
            rows = logits[mx.arange(len(toks)), last, :]        # (chunk, vocab)
            out[start:start + len(chunk)] = np.array(rows[:, mx.array(self.band_ids_)])
            mx.clear_cache()                                     # long runs otherwise creep
        return out

    def score_batch(self, prompts: list[str]) -> dict:
        """Both decodes plus the raw distribution, from a single pass over `prompts`.

        Returning both is the point: `expected` is continuous and should win on QWK,
        `argmax` cannot shrink towards the centre, and the gap between them *is* the
        compression trade that `reports/evaluation_report.md` §5 documents. Notebook 05
        measures it rather than assuming which way it goes.
        """
        logits = self.band_logits(prompts)
        probs = band_probabilities(logits, temperature=self.temperature)
        return {
            "expected": expected_score(probs, self.codec.grid),
            "argmax": argmax_score(probs, self.codec.grid),
            "probs": probs,
        }
