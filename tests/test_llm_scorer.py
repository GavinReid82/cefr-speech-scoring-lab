"""Prompt construction, fold export, and tokenizer contracts — stub tokenizer, no corpus.

Nothing here downloads a model. The properties worth testing are the ones that fail
*silently* in a real run: a prompt that differs between training and inference, a label
that is not one token, a completion that disagrees with the QWK gold class.
"""

import json

import pytest

from speechlab.llm_scorer import (
    EMPTY_TRANSCRIPT,
    assert_prompt_parity,
    band_rubric,
    band_token_ids,
    build_prompt,
    build_record,
    inference_tokens,
    transcript_text,
    write_fold_data,
    write_jsonl,
)
from speechlab.ordinal import BandCodec


class StubTokenizer:
    """Character-level stand-in: every character is one token, so single uppercase
    letters are single tokens — the one property the band-label scheme depends on."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=False, return_dict=False):
        s = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        if add_generation_prompt:
            s += "<assistant>"
        return self.encode(s)


class SystemInjectingTokenizer(StubTokenizer):
    """A template that adds a system turn on the generation path only — the exact
    asymmetry that makes inference diverge from `--mask-prompt` training with no error."""

    def apply_chat_template(self, messages, add_generation_prompt=False, return_dict=False):
        if add_generation_prompt:
            messages = [{"role": "system", "content": "be helpful"}, *messages]
        return super().apply_chat_template(messages, add_generation_prompt, return_dict)


class MultiTokenTokenizer(StubTokenizer):
    def encode(self, text, add_special_tokens=False):
        return [1, 2]


@pytest.fixture
def codec():
    return BandCodec(2.0, 5.5)                                # the observed dev range: A..H


def test_transcript_text_keeps_what_the_feature_pipeline_strips(codec):
    # case and punctuation are Whisper's own and carry fluency information; the
    # count-based features normalise them away, the language model should not
    words = [{"w": "Um,", "s": 0.0, "e": 0.3}, {"w": "I", "s": 0.4, "e": 0.5},
             {"w": "don't", "s": 0.5, "e": 0.8}]
    assert transcript_text(words) == "Um, I don't"
    assert transcript_text([]) == ""


def test_prompt_states_the_label_range_and_the_part(codec):
    p = build_prompt("i went to the park", "P3", codec)
    assert "A (lowest)" in p and "H (highest)" in p
    assert "Test part: P3" in p


def test_empty_transcript_becomes_an_explicit_placeholder(codec):
    # a silent response is still human-scored; it must reach the model as a real prompt
    for empty in ["", "   ", None]:
        assert EMPTY_TRANSCRIPT in build_prompt(empty, "P3", codec)


def test_training_and_inference_prompts_are_the_same_string(codec):
    # the JSONL row and the scoring call must not drift apart
    rec = build_record("hello there", "P4", 4.0, codec)
    assert rec["prompt"] == build_prompt("hello there", "P4", codec)


def test_the_rubric_leaves_the_unanchored_prompt_byte_identical(codec):
    # the untrained floor is only a floor for the trained arm if the two were scored on
    # the same string; adding the rubric option must not perturb the default by a byte
    assert build_prompt("hello there", "P4", codec) == (
        "Rate the spoken English proficiency of this transcribed learner response.\n\n"
        "Test part: P4\n"
        "Transcript: hello there\n\n"
        "Answer with a single letter from A (lowest) to H (highest)."
    )


def test_rubric_names_whole_bands_only(codec):
    # the corpus scores at 0.5 intervals; CEFR has no name for the midpoint, and
    # inventing one would put a label in the prompt that no rater ever used
    rubric = band_rubric(codec)
    assert rubric.startswith("A = A2, C = B1, E = B2, G = C1")
    assert not any(f"{half} =" in rubric for half in ("B", "D", "F", "H"))

    p = build_prompt("hello there", "P3", codec, rubric=True)
    assert f"Scale: {rubric}" in p
    assert "Transcript: hello there" in p and "A (lowest)" in p


def test_completion_is_the_grid_snapped_class(codec):
    assert build_record("x", "P3", 4.0, codec)["completion"] == codec.encode([4.0])[0]
    assert build_record("x", "P3", 4.2, codec)["completion"] == codec.encode([4.2])[0]
    assert set(build_record("x", "P3", 9.9, codec)["completion"]) <= set(codec.labels)


def test_jsonl_rows_use_the_keys_CompletionsDataset_reads(tmp_path, codec):
    rows = [("one two", "P3", 3.0), ("three", "P4", 5.0)]
    n = write_jsonl(tmp_path / "train.jsonl", rows, codec)
    lines = (tmp_path / "train.jsonl").read_text().splitlines()
    assert n == len(lines) == 2
    assert all(set(json.loads(ln)) == {"prompt", "completion"} for ln in lines)


def test_fold_data_writes_the_layout_mlx_lm_expects(tmp_path, codec):
    counts = write_fold_data(tmp_path / "f0", [("a", "P3", 3.0)], [("b", "P4", 4.0)], codec)
    assert counts == {"train": 1, "valid": 1}
    assert (tmp_path / "f0" / "train.jsonl").exists()
    assert (tmp_path / "f0" / "valid.jsonl").exists()


def test_fold_data_omits_valid_when_none(tmp_path, codec):
    counts = write_fold_data(tmp_path / "f1", [("a", "P3", 3.0)], None, codec)
    assert counts == {"train": 1}
    assert not (tmp_path / "f1" / "valid.jsonl").exists()


def test_band_token_ids_are_one_per_label_and_distinct(codec):
    ids = band_token_ids(StubTokenizer(), codec)
    assert len(ids) == len(codec) == len(set(ids))
    assert ids == [ord(c) for c in codec.labels]


def test_multi_token_labels_are_rejected_loudly(codec):
    # spreading the distribution over two positions would break the single-row decode
    with pytest.raises(ValueError, match="tokenises to 2 tokens"):
        band_token_ids(MultiTokenTokenizer(), codec)


def test_parity_holds_for_a_symmetric_template(codec):
    tok = StubTokenizer()
    prompt = build_prompt("i think so", "P3", codec)
    assert_prompt_parity(tok, codec, prompt, "E")
    # and the masked prefix really is what inference sends
    assert len(inference_tokens(tok, prompt)) < len(
        tok.apply_chat_template([{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": "E"}])
    )


def test_a_generation_only_system_message_is_caught(codec):
    # nothing errors in a real run — this is the failure that just quietly costs QWK
    prompt = build_prompt("i think so", "P3", codec)
    with pytest.raises(ValueError, match="not a prefix"):
        assert_prompt_parity(SystemInjectingTokenizer(), codec, prompt, "E")
