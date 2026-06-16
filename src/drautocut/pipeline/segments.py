from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from drautocut.domain.srt import Cue, ms_to_timestamp, split_timing, timestamp_to_ms


DEFAULT_TARGET_SEGMENT_MS = 10 * 60 * 1000
DEFAULT_MAX_SEGMENT_MS = 12 * 60 * 1000
DEFAULT_MIN_SEGMENT_MS = 90 * 1000
DEFAULT_SILENCE_WINDOW_MS = 45 * 1000


@dataclass(frozen=True)
class SilenceRange:
    start_ms: int
    end_ms: int

    @property
    def midpoint_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2


@dataclass(frozen=True)
class MediaSegment:
    index: int
    start_ms: int
    end_ms: int

    @property
    def offset_ms(self) -> int:
        return self.start_ms

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "offset_ms": self.offset_ms,
            "duration_ms": self.duration_ms,
            "start": ms_to_timestamp(self.start_ms),
            "end": ms_to_timestamp(self.end_ms),
        }


class SegmentPlanError(ValueError):
    pass


def normalize_silence_ranges(ranges: Iterable[SilenceRange | tuple[int, int]]) -> list[SilenceRange]:
    normalized: list[SilenceRange] = []
    for item in ranges:
        silence = item if isinstance(item, SilenceRange) else SilenceRange(start_ms=item[0], end_ms=item[1])
        if silence.end_ms <= silence.start_ms:
            continue
        normalized.append(silence)
    return sorted(normalized, key=lambda item: item.start_ms)


def plan_media_segments(
    duration_ms: int,
    *,
    target_segment_ms: int = DEFAULT_TARGET_SEGMENT_MS,
    max_segment_ms: int = DEFAULT_MAX_SEGMENT_MS,
    min_segment_ms: int = DEFAULT_MIN_SEGMENT_MS,
    silence_window_ms: int = DEFAULT_SILENCE_WINDOW_MS,
    silence_ranges: Iterable[SilenceRange | tuple[int, int]] = (),
) -> list[MediaSegment]:
    if duration_ms <= 0:
        raise SegmentPlanError("duration_ms must be greater than zero")
    if min_segment_ms <= 0 or target_segment_ms <= 0 or max_segment_ms <= 0:
        raise SegmentPlanError("segment durations must be greater than zero")
    if min_segment_ms > target_segment_ms:
        raise SegmentPlanError("min_segment_ms cannot exceed target_segment_ms")
    if target_segment_ms > max_segment_ms:
        raise SegmentPlanError("target_segment_ms cannot exceed max_segment_ms")

    silences = normalize_silence_ranges(silence_ranges)
    segments: list[MediaSegment] = []
    cursor = 0

    while cursor < duration_ms:
        remaining = duration_ms - cursor
        if remaining <= max_segment_ms:
            segments.append(MediaSegment(index=len(segments) + 1, start_ms=cursor, end_ms=duration_ms))
            break

        desired = cursor + target_segment_ms
        search_start = max(cursor + min_segment_ms, desired - silence_window_ms)
        search_end = min(cursor + max_segment_ms, desired + silence_window_ms, duration_ms)
        cut_ms = _best_silence_cut(silences, desired, search_start, search_end)
        if cut_ms is None:
            cut_ms = min(cursor + target_segment_ms, duration_ms)

        tail_ms = duration_ms - cut_ms
        if 0 < tail_ms < min_segment_ms:
            cut_ms = duration_ms

        segments.append(MediaSegment(index=len(segments) + 1, start_ms=cursor, end_ms=cut_ms))
        cursor = cut_ms

    return segments


def _best_silence_cut(
    silences: Sequence[SilenceRange],
    desired_ms: int,
    search_start_ms: int,
    search_end_ms: int,
) -> int | None:
    candidates = [
        silence.midpoint_ms
        for silence in silences
        if search_start_ms <= silence.midpoint_ms <= search_end_ms
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda cut_ms: abs(cut_ms - desired_ms))


def offset_timing(timing: str, offset_ms: int) -> str:
    start, end = split_timing(timing)
    return f"{ms_to_timestamp(timestamp_to_ms(start) + offset_ms)} --> {ms_to_timestamp(timestamp_to_ms(end) + offset_ms)}"


def offset_cues(cues: Sequence[Cue], offset_ms: int) -> list[Cue]:
    return [
        Cue(index=cue.index, timing=offset_timing(cue.timing, offset_ms), lines=list(cue.lines))
        for cue in cues
    ]


def merge_segment_cues(segment_cues: Sequence[tuple[MediaSegment, Sequence[Cue]]]) -> list[Cue]:
    merged: list[Cue] = []
    for segment, cues in sorted(segment_cues, key=lambda item: item[0].index):
        merged.extend(offset_cues(cues, segment.offset_ms))
    for index, cue in enumerate(merged, start=1):
        cue.index = str(index)
    return merged
