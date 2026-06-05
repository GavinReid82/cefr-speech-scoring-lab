"""Features from word timestamps (fluency) and raw audio (prosody)."""

from pathlib import Path

import numpy as np

from speechlab.text_features import count_filled_pauses, normalise_tokens, ttr

PAUSE_GAP = 0.3      # silence >= 0.3s between words counts as a pause
LONG_PAUSE = 1.0     # silence >= 1.0s counts as a long pause

FLUENCY_FEATURES = ["duration_sec", "n_words", "speech_rate_wpm", "articulation_rate",
                    "n_pauses", "total_pause_time", "mean_pause_dur", "n_long_pauses",
                    "mean_length_of_run", "ttr", "filled_pauses"]

PROSODY_FEATURES = ["f0_median_hz", "f0_iqr_st", "f0_range_st", "voiced_ratio", "intensity_sd_db"]


def fluency_features(words: list[dict], duration: float,
                     pause_gap: float = PAUSE_GAP, long_pause: float = LONG_PAUSE) -> dict:
    """Compute fluency features from ASR word timestamps.

    words: [{'w': str, 's': start_sec, 'e': end_sec}, ...] in time order
    duration: audio file length in seconds
    """
    n = len(words)
    if n == 0:                       # silent / failed ASR -> all-zero row, flagged for exclusion
        return dict.fromkeys(FLUENCY_FEATURES, 0.0) | {"duration_sec": duration}

    gaps = [w2["s"] - w1["e"] for w1, w2 in zip(words, words[1:])]
    pauses = [g for g in gaps if g >= pause_gap]
    phonation = sum(w["e"] - w["s"] for w in words)            # time spent actually speaking
    runs = np.split(np.arange(n), [i + 1 for i, g in enumerate(gaps) if g >= pause_gap])
    tokens = normalise_tokens([w["w"] for w in words])

    return {
        "duration_sec": duration,
        "n_words": n,
        "speech_rate_wpm": n / duration * 60,
        "articulation_rate": n / phonation * 60 if phonation > 0 else 0.0,
        "n_pauses": len(pauses),
        "total_pause_time": sum(pauses),
        "mean_pause_dur": float(np.mean(pauses)) if pauses else 0.0,
        "n_long_pauses": sum(g >= long_pause for g in gaps),
        "mean_length_of_run": float(np.mean([len(r) for r in runs])),
        "ttr": ttr(tokens),
        "filled_pauses": count_filled_pauses(tokens),
    }


def prosody_features(path: Path) -> dict:
    """Pitch and intensity statistics straight from the audio — ASR-independent.

    Pitch spread is measured in semitones (log-scale, comparable across voices).
    Intensity *means* are deliberately excluded: recording gain varies by device,
    so only the spread is trustworthy.
    """
    import parselmouth  # heavy native import — keep lazy so the module imports without it

    snd = parselmouth.Sound(str(path))
    f0 = snd.to_pitch(time_step=0.01).selected_array["frequency"]   # 0 where unvoiced
    voiced = f0[f0 > 0]
    db = snd.to_intensity().values[0]
    db = db[np.isfinite(db) & (db > 0)]
    if len(voiced) < 10:                       # silent / no voicing detected
        return dict.fromkeys(PROSODY_FEATURES, 0.0)

    q05, q25, med, q75, q95 = np.percentile(voiced, [5, 25, 50, 75, 95])

    def st(hi: float, lo: float) -> float:     # Hz ratio -> semitones
        return 12 * float(np.log2(hi / lo))

    return {
        "f0_median_hz": float(med),
        "f0_iqr_st": st(q75, q25),
        "f0_range_st": st(q95, q05),
        "voiced_ratio": float(len(voiced) / len(f0)),
        "intensity_sd_db": float(np.std(db)),
    }
