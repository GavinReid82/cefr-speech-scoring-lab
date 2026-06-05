"""Text-side features computed from token sequences (ASR or gold)."""

import re

# Filled-pause surface forms an ASR system might emit (Whisper rarely does — see notebook 02)
FILLED = re.compile(r"^(um+|uh+|er+m?|eh+|h?mm+)$", re.IGNORECASE)

_WORD = re.compile(r"[^a-z']")


def normalise_tokens(words: list[str]) -> list[str]:
    """Lowercase, strip everything but letters/apostrophes, drop empties."""
    toks = (_WORD.sub("", w.lower()) for w in words)
    return [t for t in toks if t]


def count_filled_pauses(tokens: list[str]) -> int:
    return sum(bool(FILLED.match(t)) for t in tokens)


def ttr(tokens: list[str]) -> float:
    """Type-token ratio. Kept for comparison only — length-sensitive (notebook 01)."""
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

    Mean factor length at the given TTR threshold, averaged over a forward and a
    backward pass. Length-insensitive, unlike raw TTR. Returns 0.0 below 10 tokens
    (too short for the statistic to mean anything).
    """
    def one_pass(seq: list[str]) -> float:
        factors, types, count = 0.0, set(), 0
        for t in seq:
            count += 1
            types.add(t)
            if len(types) / count <= threshold:
                factors += 1.0
                types, count = set(), 0
        if count:
            factors += (1 - len(types) / count) / (1 - threshold)
        # text so diverse it never completes a factor: MTLD is undefined (infinite);
        # cap at the text length rather than collapsing to 0 (minimum diversity)
        return len(seq) / factors if factors else float(len(seq))

    if len(tokens) < 10:
        return 0.0
    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2
