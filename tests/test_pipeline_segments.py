import pytest

from caption_core.domain.srt import Cue
from caption_core.pipeline.segments import (
    SegmentPlanError,
    merge_segment_cues,
    offset_timing,
    plan_media_segments,
)


def test_plan_media_segments_uses_target_duration_for_long_video() -> None:
    segments = plan_media_segments(58 * 60 * 1000, target_segment_ms=10 * 60 * 1000)

    assert len(segments) == 6
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 10 * 60 * 1000
    assert segments[-1].duration_ms == 8 * 60 * 1000


def test_plan_media_segments_prefers_silence_near_target() -> None:
    segments = plan_media_segments(
        25 * 60 * 1000,
        target_segment_ms=10 * 60 * 1000,
        max_segment_ms=12 * 60 * 1000,
        silence_window_ms=60 * 1000,
        silence_ranges=[(9 * 60 * 1000 + 55 * 1000, 9 * 60 * 1000 + 57 * 1000)],
    )

    assert segments[0].end_ms == 9 * 60 * 1000 + 56 * 1000
    assert segments[1].start_ms == segments[0].end_ms


def test_plan_media_segments_rejects_invalid_settings() -> None:
    with pytest.raises(SegmentPlanError):
        plan_media_segments(0)


def test_offset_timing_adds_segment_offset() -> None:
    assert (
        offset_timing("00:00:01,250 --> 00:00:02,500", 10 * 60 * 1000)
        == "00:10:01,250 --> 00:10:02,500"
    )


def test_merge_segment_cues_offsets_and_renumbers() -> None:
    segments = plan_media_segments(20 * 60 * 1000, target_segment_ms=10 * 60 * 1000)
    merged = merge_segment_cues(
        [
            (
                segments[1],
                [Cue(index="1", timing="00:00:00,000 --> 00:00:01,000", lines=["第二段开头"])],
            ),
            (
                segments[0],
                [Cue(index="7", timing="00:00:02,000 --> 00:00:03,000", lines=["第一段"])],
            ),
        ]
    )

    assert [cue.index for cue in merged] == ["1", "2"]
    assert merged[0].timing == "00:00:02,000 --> 00:00:03,000"
    assert merged[1].timing == "00:10:00,000 --> 00:10:01,000"
