"""ASR transcription (faster-whisper), gold STM parsing, and WER against gold."""

import json
import re
from pathlib import Path

HES = "(%hesitation%)"
PARTIAL = re.compile(r"^\(\S+-\)$")            # gold partial-word tokens like (s-)
IGNORE = "IGNORE_TIME_SEGMENT_IN_SCORING"


def parse_stm(path: Path) -> dict[str, list[str]]:
    """STM line: `utt chan speaker start end <tags> words...` -> utt: tokens in time order."""
    segs: dict[str, list] = {}
    for line in open(path):
        if line.startswith(";;"):
            continue
        p = line.split(maxsplit=6)
        if len(p) < 7:
            continue
        utt, start, text = p[0], float(p[3]), p[6].strip()
        if text == IGNORE:
            continue
        segs.setdefault(utt, []).append((start, text.split()))
    return {u: [t for _, seg in sorted(v, key=lambda x: x[0]) for t in seg]
            for u, v in segs.items()}


def parse_stm_tags(path: Path) -> dict[str, dict]:
    """Per-utterance category tags from the first STM segment line.

    `<o,Q4,B2,P3>` -> {"quality": "Q4", "grade": "B2", "part": "P3"} — audio quality
    (Q2–QX) and CEFR grade labels are the corpus's fairness-slice metadata.
    """
    out: dict[str, dict] = {}
    for line in open(path):
        if line.startswith(";;"):
            continue
        p = line.split(maxsplit=6)
        if len(p) < 6 or p[0] in out:
            continue
        m = re.search(r"<([^>]*)>", p[5])
        if not m:
            continue
        fields = m.group(1).split(",")
        if len(fields) >= 4:
            out[p[0]] = {"quality": fields[1], "grade": fields[2], "part": fields[3]}
    return out


def wer_pair(gold_tokens: list[str], hyp_text: str) -> dict:
    """WER of an ASR hypothesis against one gold disfluent-form transcript.

    lenient: hesitations and partial words removed from the gold — content-word accuracy.
    strict:  kept — the gap between the two is the disfluency erasure cost.
    Both sides share one normalisation (case, punctuation), so contractions aren't errors.
    """
    import jiwer  # lazy: keeps this module importable without the dependency

    norm = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    lenient_ref = " ".join(t for t in gold_tokens if t != HES and not PARTIAL.match(t))
    strict_ref = " ".join(gold_tokens)
    o_len = jiwer.process_words(lenient_ref, hyp_text, norm, norm)
    o_str = jiwer.process_words(strict_ref, hyp_text, norm, norm)
    return {
        "wer_lenient": o_len.wer,
        "wer_strict": o_str.wer,
        "n_gold": len(lenient_ref.split()),
        "subs": o_len.substitutions,
        "dels": o_len.deletions,
        "ins": o_len.insertions,
    }


def transcribe(path: Path, model) -> list[dict]:
    """Transcribe one file with a faster-whisper model -> word-timestamp list."""
    segments, _ = model.transcribe(str(path), word_timestamps=True, language="en")
    return [{"w": w.word.strip(), "s": w.start, "e": w.end}
            for seg in segments for w in seg.words]


def load_or_transcribe(cache_path: Path, items: dict[str, Path],
                       model_name: str = "small", save_every: int = 25) -> dict:
    """Resumable transcription: utt -> word list, cached as JSON.

    items: utt -> audio path. Already-cached utts are skipped, so the function can
    be re-run after an interruption and only does the remaining work.
    """
    cache: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = {u: p for u, p in items.items() if u not in cache}
    if not todo:
        return cache

    from faster_whisper import WhisperModel  # lazy heavy import
    model = WhisperModel(model_name, compute_type="int8")

    for i, (utt, path) in enumerate(todo.items(), 1):
        cache[utt] = transcribe(path, model)
        if i % save_every == 0 or i == len(todo):
            cache_path.write_text(json.dumps(cache))
            print(f"{i}/{len(todo)} transcribed ({len(cache)} total cached)", flush=True)
    return cache
