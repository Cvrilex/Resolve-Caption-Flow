from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from caption_core.domain.srt import Cue
from caption_core.pipeline.segments import MediaSegment, merge_segment_cues


ProgressCallback = Callable[[dict[str, Any]], None]


class SegmentTranscriber(Protocol):
    name: str

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress: Callable[[int, str], None] | None = None,
    ) -> list[Cue]:
        ...


@dataclass(frozen=True)
class SegmentAsrTask:
    segment: MediaSegment
    audio_path: Path


@dataclass(frozen=True)
class SegmentAsrResult:
    segment: MediaSegment
    audio_path: Path
    cues: list[Cue]


class SegmentedAsrError(RuntimeError):
    def __init__(self, message: str, *, failures: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.failures = failures or []


class FunctionTranscriber:
    def __init__(self, name: str, fn: Callable[[Path, Callable[[int, str], None] | None], list[Cue]]):
        self.name = name
        self._fn = fn

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress: Callable[[int, str], None] | None = None,
    ) -> list[Cue]:
        return self._fn(audio_path, progress)


def run_segmented_asr(
    tasks: Sequence[SegmentAsrTask],
    transcriber: SegmentTranscriber,
    *,
    max_workers: int = 1,
    fail_fast: bool = False,
    progress: ProgressCallback | None = None,
) -> tuple[list[Cue], list[SegmentAsrResult]]:
    if not tasks:
        raise SegmentedAsrError("No ASR segment tasks provided")
    if max_workers <= 0:
        raise SegmentedAsrError("max_workers must be greater than zero")

    total = len(tasks)
    segment_progress: dict[int, int] = {task.segment.index: 0 for task in tasks}
    results: list[SegmentAsrResult] = []
    failures: list[dict[str, Any]] = []

    def emit(status: str, message: str, task: SegmentAsrTask | None = None, **extra: Any) -> None:
        if progress is None:
            return
        event: dict[str, Any] = {
            "step": "asr",
            "status": status,
            "message": message,
            "engine": transcriber.name,
            "total_segments": total,
            "completed_segments": len(results),
            "failed_segments": len(failures),
            "percent": _aggregate_percent(segment_progress),
        }
        if task is not None:
            event.update(
                {
                    "segment": task.segment.index,
                    "segment_start": task.segment.to_dict()["start"],
                    "segment_end": task.segment.to_dict()["end"],
                    "audio": str(task.audio_path),
                }
            )
        event.update(extra)
        progress(event)

    def run_one(task: SegmentAsrTask) -> SegmentAsrResult:
        emit("segment_running", f"ASR segment {task.segment.index}/{total} started", task)

        def on_segment_progress(percent: int, message: str) -> None:
            bounded = max(0, min(100, int(percent)))
            segment_progress[task.segment.index] = bounded
            emit(
                "segment_progress",
                message,
                task,
                segment_percent=bounded,
            )

        cues = transcriber.transcribe(task.audio_path, progress=on_segment_progress)
        if not cues:
            raise SegmentedAsrError(f"ASR segment {task.segment.index} returned no cues")
        segment_progress[task.segment.index] = 100
        return SegmentAsrResult(segment=task.segment, audio_path=task.audio_path, cues=cues)

    emit("running", "segmented ASR started", max_workers=max_workers)
    if max_workers == 1:
        for task in tasks:
            try:
                result = run_one(task)
                results.append(result)
                emit(
                    "segment_done",
                    f"ASR segment {task.segment.index}/{total} completed",
                    task,
                    cue_count=len(result.cues),
                )
            except Exception as exc:
                failure = _failure_for(task, exc)
                failures.append(failure)
                segment_progress[task.segment.index] = 0
                emit("segment_failed", failure["error"], task, error=failure["error"])
                if fail_fast:
                    break
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task: dict[Future[SegmentAsrResult], SegmentAsrTask] = {
                executor.submit(run_one, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    emit(
                        "segment_done",
                        f"ASR segment {task.segment.index}/{total} completed",
                        task,
                        cue_count=len(result.cues),
                    )
                except Exception as exc:
                    failure = _failure_for(task, exc)
                    failures.append(failure)
                    segment_progress[task.segment.index] = 0
                    emit("segment_failed", failure["error"], task, error=failure["error"])
                    if fail_fast:
                        for pending in future_to_task:
                            pending.cancel()
                        break

    if failures:
        emit("failed", f"segmented ASR failed: {len(failures)} segment(s) failed", failures=failures)
        raise SegmentedAsrError("Segmented ASR failed", failures=failures)

    merged = merge_segment_cues([(result.segment, result.cues) for result in results])
    emit("done", "segmented ASR completed", cue_count=len(merged), percent=100)
    return merged, sorted(results, key=lambda item: item.segment.index)


def build_segment_tasks(segments: Sequence[MediaSegment], audio_paths: Sequence[Path]) -> list[SegmentAsrTask]:
    if len(segments) != len(audio_paths):
        raise SegmentedAsrError("segments and audio_paths must have the same length")
    return [
        SegmentAsrTask(segment=segment, audio_path=audio_path)
        for segment, audio_path in zip(segments, audio_paths)
    ]


def _aggregate_percent(segment_progress: dict[int, int]) -> int:
    if not segment_progress:
        return 0
    return int(round(sum(segment_progress.values()) / len(segment_progress)))


def _failure_for(task: SegmentAsrTask, exc: Exception) -> dict[str, Any]:
    return {
        "segment": task.segment.index,
        "audio": str(task.audio_path),
        "start": task.segment.to_dict()["start"],
        "end": task.segment.to_dict()["end"],
        "error": str(exc),
    }
