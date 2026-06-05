"""All fixtures are synthetic — no corpus data may appear in tests (licence rule)."""

from speechlab.text_features import count_filled_pauses, mtld, normalise_tokens, ttr


def test_normalise_tokens_strips_case_punctuation_and_empties():
    assert normalise_tokens(["Well,", "I'm", "...", "HAPPY!"]) == ["well", "i'm", "happy"]


def test_count_filled_pauses_matches_common_forms():
    assert count_filled_pauses(["um", "cat", "uhh", "erm", "hmm", "dog"]) == 4
    assert count_filled_pauses(["umbrella", "error", "summer"]) == 0


def test_ttr_basic():
    assert ttr(["a", "b", "a", "b"]) == 0.5
    assert ttr([]) == 0.0


def test_mtld_short_text_returns_zero():
    assert mtld(["a"] * 9) == 0.0


def test_mtld_repetition_scores_lower_than_diversity():
    repetitive = ["the", "cat", "sat"] * 20
    diverse = [f"word{i}" for i in range(60)]
    assert mtld(repetitive) < mtld(diverse)


def test_mtld_fully_unique_text_caps_at_length_not_zero():
    # never completes a factor -> undefined (infinite); must not collapse to 0
    assert mtld([f"word{i}" for i in range(60)]) == 60.0


def test_mtld_is_more_length_stable_than_ttr():
    # Zipf-ish synthetic speech: doubling the text should crater TTR but barely move MTLD
    import random
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(50)]
    weights = [1 / (i + 1) for i in range(50)]
    base = rng.choices(vocab, weights=weights, k=150)
    doubled = base + rng.choices(vocab, weights=weights, k=150)
    ttr_change = abs(ttr(doubled) - ttr(base)) / ttr(base)
    mtld_change = abs(mtld(doubled) - mtld(base)) / mtld(base)
    assert mtld_change < ttr_change
