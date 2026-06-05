"""Synthetic fixtures only: hand-built timestamp lists and a generated sine wave."""

import numpy as np
import pytest
import soundfile as sf

from speechlab.acoustic_features import fluency_features, prosody_features


def w(word, s, e):
    return {"w": word, "s": s, "e": e}


def test_fluency_empty_input_flags_silent_response():
    out = fluency_features([], duration=30.0)
    assert out["n_words"] == 0.0
    assert out["duration_sec"] == 30.0


def test_fluency_counts_pauses_and_runs():
    # two words, 0.5s gap (pause), then adjacent word; one long pause of 1.2s at the end
    words = [w("a", 0.0, 0.2), w("b", 0.7, 0.9), w("c", 0.95, 1.1), w("d", 2.3, 2.5)]
    out = fluency_features(words, duration=3.0)
    assert out["n_words"] == 4
    assert out["n_pauses"] == 2                      # 0.5s and 1.2s gaps
    assert out["n_long_pauses"] == 1                 # only the 1.2s gap
    assert out["total_pause_time"] == pytest.approx(1.7)
    assert out["mean_length_of_run"] == pytest.approx(4 / 3)  # runs: [a], [b, c], [d]


def test_fluency_rates():
    words = [w("one", 0.0, 0.5), w("two", 0.5, 1.0)]   # no gaps, 1s phonation
    out = fluency_features(words, duration=4.0)
    assert out["speech_rate_wpm"] == pytest.approx(30.0)       # 2 words / 4s
    assert out["articulation_rate"] == pytest.approx(120.0)    # 2 words / 1s speaking


def test_prosody_recovers_sine_pitch(tmp_path):
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, tone, sr)

    out = prosody_features(path)
    assert out["f0_median_hz"] == pytest.approx(220.0, abs=5.0)
    assert out["voiced_ratio"] > 0.8
    assert out["f0_iqr_st"] < 1.0                     # constant pitch -> almost no spread


def test_prosody_silence_returns_zero_row(tmp_path):
    sr = 16000
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(sr, dtype=np.float32), sr)
    out = prosody_features(path)
    assert out["f0_median_hz"] == 0.0
    assert out["voiced_ratio"] == 0.0
