from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from drautocut.domain.srt import Cue, write_srt
from drautocut.integrations.ffmpeg import detect_silences, extract_audio_segments, probe_media
from drautocut.integrations.online_asr import OnlineAsrTranscriber
from drautocut.pipeline.asr import build_segment_tasks, run_segmented_asr
from drautocut.pipeline.segments import (
    DEFAULT_MAX_SEGMENT_MS,
    DEFAULT_MIN_SEGMENT_MS,
    DEFAULT_SILENCE_WINDOW_MS,
    DEFAULT_TARGET_SEGMENT_MS,
    MediaSegment,
    SilenceRange,
    plan_media_segments,
)


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class LongAsrResult:
    srt_path: Path
    cues: list[Cue]
    segments: list[MediaSegment]
    audio_paths: list[Path]
    silences: list[SilenceRange]


def run_online_long_video_asr(
    *,
    video_path: Path,
    output_srt: Path,
    work_dir: Path,
    tool_dir: Path,
    engine: str,
    max_workers: int = 1,
    target_segment_ms: int = DEFAULT_TARGET_SEGMENT_MS,
    max_segment_ms: int = DEFAULT_MAX_SEGMENT_MS,
    min_segment_ms: int = DEFAULT_MIN_SEGMENT_MS,
    silence_window_ms: int = DEFAULT_SILENCE_WINDOW_MS,
    progress: ProgressCallback | None = None,
) -> LongAsrResult:
    def emit(status: str, message: str, **extra: Any) -> None:
        if progress is None:
            return
        event = {"step": "asr_prepare", "status": status, "message": message}
        event.update(extra)
        progress(event)

    emit("running", "probing media duration", video=str(video_path))
    media = probe_media(video_path)
    emit("done", "media duration read", duration_ms=media.duration_ms)

    emit("running", "detecting silence cut points")
    silences = detect_silences(video_path)
    emit("done", "silence detection complete", silence_count=len(silences))

    segments = plan_media_segments(
        media.duration_ms,
        target_segment_ms=target_segment_ms,
        max_segment_ms=max_segment_ms,
        min_segment_ms=min_segment_ms,
        silence_window_ms=silence_window_ms,
        silence_ranges=silences,
    )
    emit(
        "done",
        "ASR segment plan created",
        segment_count=len(segments),
        segments=[segment.to_dict() for segment in segments],
    )

    audio_dir = work_dir / "asr_segments" / video_path.stem
    emit("running", "extracting audio segments", audio_dir=str(audio_dir))
    audio_paths = extract_audio_segments(video_path, audio_dir, segments)
    emit("done", "audio segments extracted", audio_count=len(audio_paths))

    transcriber = OnlineAsrTranscriber(engine, tool_dir=tool_dir)
    cues, _segment_results = run_segmented_asr(
        build_segment_tasks(segments, audio_paths),
        transcriber,
        max_workers=max_workers,
        progress=progress,
    )

    write_srt(cues, output_srt)
    emit("done", "merged SRT written", srt=str(output_srt), cue_count=len(cues))
    return LongAsrResult(
        srt_path=output_srt,
        cues=cues,
        segments=segments,
        audio_paths=audio_paths,
        silences=silences,
    )
