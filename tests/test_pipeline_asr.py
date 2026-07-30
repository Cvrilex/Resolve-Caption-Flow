from pathlib import Path

import pytest

from caption_core.domain.srt import Cue
from caption_core.pipeline.asr import (
    FunctionTranscriber,
    SegmentedAsrError,
    build_segment_tasks,
    run_segmented_asr,
)
from caption_core.pipeline.segments import plan_media_segments


def test_run_segmented_asr_merges_offsets_and_emits_progress(tmp_path: Path) -> None:
    segments = plan_media_segments(20 * 60 * 1000, target_segment_ms=10 * 60 * 1000)
    audio_paths = [tmp_path / "part1.m4a", tmp_path / "part2.m4a"]
    events = []

    def transcribe(audio_path: Path, progress):
        if progress:
            progress(50, "halfway")
            progress(100, "done")
        return [Cue(index="1", timing="00:00:01,000 --> 00:00:02,000", lines=[audio_path.stem])]

    merged, results = run_segmented_asr(
        build_segment_tasks(segments, audio_paths),
        FunctionTranscriber("fake", transcribe),
        progress=events.append,
    )

    assert len(results) == 2
    assert [cue.index for cue in merged] == ["1", "2"]
    assert merged[0].timing == "00:00:01,000 --> 00:00:02,000"
    assert merged[1].timing == "00:10:01,000 --> 00:10:02,000"
    assert events[0]["status"] == "running"
    assert any(event["status"] == "segment_progress" and event["segment_percent"] == 50 for event in events)
    assert events[-1]["status"] == "done"
    assert events[-1]["percent"] == 100


def test_run_segmented_asr_reports_failed_segment(tmp_path: Path) -> None:
    segments = plan_media_segments(20 * 60 * 1000, target_segment_ms=10 * 60 * 1000)
    audio_paths = [tmp_path / "part1.m4a", tmp_path / "part2.m4a"]
    events = []

    def transcribe(audio_path: Path, progress):
        if audio_path.name == "part2.m4a":
            raise RuntimeError("remote quota exceeded")
        return [Cue(index="1", timing="00:00:00,000 --> 00:00:01,000", lines=["ok"])]

    with pytest.raises(SegmentedAsrError) as exc_info:
        run_segmented_asr(
            build_segment_tasks(segments, audio_paths),
            FunctionTranscriber("fake", transcribe),
            progress=events.append,
        )

    assert exc_info.value.failures[0]["segment"] == 2
    assert "remote quota exceeded" in exc_info.value.failures[0]["error"]
    assert events[-1]["status"] == "failed"
    assert events[-1]["failed_segments"] == 1


def test_build_segment_tasks_requires_matching_lengths(tmp_path: Path) -> None:
    segments = plan_media_segments(10 * 60 * 1000)

    with pytest.raises(SegmentedAsrError):
        build_segment_tasks(segments, [tmp_path / "only-one.m4a", tmp_path / "extra.m4a"])
