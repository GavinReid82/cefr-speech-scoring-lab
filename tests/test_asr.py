"""Synthetic STM text and hand-written hypotheses — no corpus data."""

from speechlab.asr import parse_stm, wer_pair

STM = """;; comment line
UTT1 1 SPK1 0.00 5.00 <o,Q1,C,P3> the cat (%hesitation%) sat
UTT1 1 SPK1 5.00 9.00 <o,Q1,C,P3> on the (m-) mat
UTT2 1 SPK2 0.00 4.00 <o,Q1,C,P3> IGNORE_TIME_SEGMENT_IN_SCORING
UTT2 1 SPK2 4.00 8.00 <o,Q1,C,P3> hello world
"""


def test_parse_stm_orders_segments_and_skips_ignores(tmp_path):
    path = tmp_path / "test.stm"
    path.write_text(STM)
    gold = parse_stm(path)
    assert gold["UTT1"] == ["the", "cat", "(%hesitation%)", "sat", "on", "the", "(m-)", "mat"]
    assert gold["UTT2"] == ["hello", "world"]


def test_parse_stm_sorts_out_of_order_segments(tmp_path):
    path = tmp_path / "test.stm"
    path.write_text(
        "U 1 S 5.00 9.00 <o> second part\n"
        "U 1 S 0.00 5.00 <o> first part\n"
    )
    assert parse_stm(path)["U"] == ["first", "part", "second", "part"]


def test_wer_perfect_hypothesis_lenient_zero():
    gold = ["the", "cat", "(%hesitation%)", "sat", "on", "the", "(m-)", "mat"]
    out = wer_pair(gold, "The cat sat on the mat.")
    assert out["wer_lenient"] == 0.0
    assert out["n_gold"] == 6
    # strict keeps the hesitation and partial as gold tokens the hypothesis lacks
    assert out["wer_strict"] > 0.0


def test_wer_normalisation_ignores_case_and_punctuation():
    out = wer_pair(["i'm", "happy"], "I'm happy!")
    assert out["wer_lenient"] == 0.0


def test_wer_counts_substitutions():
    out = wer_pair(["the", "cat"], "the dog")
    assert out["wer_lenient"] == 0.5
    assert out["subs"] == 1
