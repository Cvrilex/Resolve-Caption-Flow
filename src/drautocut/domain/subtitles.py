from __future__ import annotations

import re
from typing import Any

from .srt import Cue, ms_to_timestamp, split_timing, timestamp_to_ms


DEFAULT_REMOVE_PUNCTUATION = "，,"


def visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def clean_punctuation(text: str, punctuation: str = DEFAULT_REMOVE_PUNCTUATION) -> str:
    if not punctuation:
        return text
    table = str.maketrans("", "", punctuation)
    cleaned = text.translate(table)
    cleaned = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n\s+", "\n", cleaned)
    return cleaned.strip()


def split_preserving_english_spaces(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if visible_len(text) <= max_chars:
        return [text] if text else []

    segments: list[str] = []
    current = ""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+./-]*|[\u4e00-\u9fff]|[^\s]", text)
    for token in tokens:
        joiner = (
            " "
            if current and re.match(r"^[A-Za-z0-9]", token) and re.search(r"[A-Za-z0-9+./-]$", current)
            else ""
        )
        candidate = f"{current}{joiner}{token}" if current else token
        if current and visible_len(candidate) > max_chars:
            segments.append(current)
            current = token
        else:
            current = candidate
    if current:
        segments.append(current)
    return segments


def enforce_max_chars(segments: list[str], max_chars: int) -> list[str]:
    enforced: list[str] = []
    for segment in segments:
        if visible_len(segment) <= max_chars:
            enforced.append(segment)
        else:
            enforced.extend(split_preserving_english_spaces(segment, max_chars))
    return [segment for segment in enforced if segment]


def interpolate_timestamps(start_ts: str, end_ts: str, segments: list[str]) -> list[tuple[str, str]]:
    start_ms = timestamp_to_ms(start_ts)
    end_ms = timestamp_to_ms(end_ts)
    total_ms = end_ms - start_ms
    if total_ms <= 0 or not segments:
        return [(start_ts, end_ts)]

    lengths = [max(1, visible_len(segment)) for segment in segments]
    total_len = sum(lengths)
    if total_len == 0:
        return [(start_ts, end_ts)]

    timings: list[tuple[str, str]] = []
    cursor = start_ms
    min_ms = 500 if total_ms >= len(segments) * 500 else max(1, total_ms // len(segments))
    for index, _segment in enumerate(segments):
        fraction = lengths[index] / total_len
        segment_duration = max(min_ms, int(round(fraction * total_ms)))
        if index == len(segments) - 1:
            segment_end = end_ms
        else:
            latest_end = end_ms - (len(segments) - index - 1) * min_ms
            segment_end = min(cursor + segment_duration, latest_end)
            segment_end = max(cursor + 1, segment_end)
        timings.append((ms_to_timestamp(cursor), ms_to_timestamp(segment_end)))
        cursor = segment_end
    return timings


def split_cue(cue: Cue, segments: list[str]) -> list[Cue]:
    start, end = split_timing(cue.timing)
    timings = interpolate_timestamps(start, end, segments)
    return [
        Cue(index="", timing=f"{segment_start} --> {segment_end}", lines=[segment])
        for segment, (segment_start, segment_end) in zip(segments, timings)
    ]


def clean_and_split_cues(
    cues: list[Cue],
    max_chars: int,
    punctuation: str = DEFAULT_REMOVE_PUNCTUATION,
) -> tuple[list[Cue], dict[str, Any]]:
    cleaned_cues: list[Cue] = []
    punctuation_changes: list[dict[str, Any]] = []
    for cue in cues:
        before = cue.single_line_text
        after = clean_punctuation(before, punctuation)
        cleaned_cues.append(Cue(index=cue.index, timing=cue.timing, lines=[after] if after else [""]))
        if before != after:
            punctuation_changes.append({"cue": cue.index, "timing": cue.timing, "before": before, "after": after})

    expanded: list[Cue] = []
    overlong_changes: list[dict[str, Any]] = []
    for cue in cleaned_cues:
        text = cue.single_line_text
        if visible_len(text) <= max_chars:
            expanded.append(Cue(index="", timing=cue.timing, lines=[text]))
            continue

        segments = split_preserving_english_spaces(text, max_chars)
        expanded.extend(split_cue(cue, segments))
        overlong_changes.append(
            {
                "cue": cue.index,
                "timing": cue.timing,
                "before": text,
                "after": segments,
                "split_count": len(segments),
                "segment_lengths": [visible_len(segment) for segment in segments],
            }
        )

    for index, cue in enumerate(expanded, start=1):
        cue.index = str(index)

    report = {
        "max_chars": max_chars,
        "removed_punctuation": punctuation,
        "cue_count_before": len(cues),
        "cue_count_after": len(expanded),
        "punctuation_changed_cue_count": len(punctuation_changes),
        "overlong_detected_count": len(overlong_changes),
        "overlong_changed_cue_count": len(overlong_changes),
        "punctuation_changes": punctuation_changes,
        "overlong_changes": overlong_changes,
    }
    return expanded, report

