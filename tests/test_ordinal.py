"""Ordinal codec and decoding — synthetic arrays only, no tokenizer, no corpus."""

import numpy as np
import pytest

from speechlab.evaluation import to_grid
from speechlab.ordinal import (
    BandCodec,
    argmax_score,
    band_probabilities,
    expected_score,
)


def test_grid_covers_the_half_step_scale_inclusively():
    codec = BandCodec(1.0, 6.0)
    assert len(codec) == 11                                   # 1.0, 1.5, ..., 6.0
    assert codec.grid[0] == 1.0 and codec.grid[-1] == 6.0
    assert "".join(codec.labels) == "ABCDEFGHIJK"


def test_encode_agrees_with_the_evaluation_grid():
    # the whole point of delegating to to_grid: a training label and the QWK gold
    # class can never disagree about which band a score belongs to
    codec = BandCodec(1.0, 6.0)
    scores = np.array([1.0, 2.4, 3.5, 4.9, 6.0, 7.2, -3.0])
    labels = codec.encode(scores)
    from_grid = to_grid(scores, 1.0, 6.0) - 2                 # doubled ints -> grid index
    assert [codec.labels[i] for i in from_grid] == labels


def test_encode_decode_round_trips_on_grid_points():
    codec = BandCodec(1.0, 6.0)
    on_grid = codec.grid
    assert np.array_equal(codec.decode(codec.encode(on_grid)), on_grid)


def test_encode_clips_out_of_range_scores_to_the_end_bands():
    codec = BandCodec(2.0, 5.0)
    assert codec.encode([9.0]) == [codec.labels[-1]]
    assert codec.encode([-1.0]) == [codec.labels[0]]


def test_from_scores_uses_observed_bounds():
    codec = BandCodec.from_scores([2.0, 3.5, 4.0, 5.0])
    assert (codec.lo, codec.hi) == (2.0, 5.0)
    assert len(codec) == 7


def test_rejects_degenerate_and_oversized_ranges():
    with pytest.raises(ValueError):
        BandCodec(4.0, 4.0)
    with pytest.raises(ValueError):
        BandCodec(0.0, 20.0)                                  # 41 bands > 26 single-token labels


def test_probabilities_are_a_distribution_over_bands_only():
    # the restricted softmax must renormalise over bands, discarding the rest of the vocab
    logits = np.array([[10.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    p = band_probabilities(logits)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[1], 1 / 3)                           # flat logits -> uniform


def test_probabilities_are_stable_at_extreme_logits():
    p = band_probabilities(np.array([[1e4, -1e4, 0.0]]))      # naive exp() would overflow
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(), 1.0)


def test_temperature_sharpens_towards_argmax():
    logits = np.array([[2.0, 1.0, 0.0]])
    sharp = band_probabilities(logits, temperature=0.1)
    soft = band_probabilities(logits, temperature=10.0)
    assert sharp.max() > soft.max()
    with pytest.raises(ValueError):
        band_probabilities(logits, temperature=0.0)


def test_expected_score_is_the_probability_weighted_band():
    grid = np.array([1.0, 2.0, 3.0])
    probs = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(expected_score(probs, grid), [1.5, 3.0])


def test_expectation_shrinks_towards_the_centre_and_argmax_does_not():
    # the documented trade: expectation is continuous but pulled inwards by any
    # mass on neighbouring bands — the mechanism behind band compression
    grid = np.array([1.0, 2.0, 3.0])
    probs = np.array([[0.7, 0.2, 0.1]])
    assert expected_score(probs, grid)[0] > argmax_score(probs, grid)[0]
    assert argmax_score(probs, grid)[0] == 1.0


def test_confident_predictions_agree_under_both_decodes():
    grid = np.array([1.0, 2.0, 3.0])
    probs = np.array([[0.001, 0.998, 0.001]])
    assert np.allclose(expected_score(probs, grid), argmax_score(probs, grid), atol=1e-2)
