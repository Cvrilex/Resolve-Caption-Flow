from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from caption_core.pipeline.segments import MediaSegment, SilenceRange


@dataclass(frozen=True)
class MediaInfo:
    duration_ms: int
    width: int | None = None
    height: int | None = None
    fps: float | None = None


class FfmpegError(RuntimeError):
    pass


def require_command(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise FfmpegError(f"Missing required command: {name}")
    return found


def run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip()
        raise FfmpegError(stderr[-1200:] or f"Command failed: {' '.join(args)}")
    return proc


def probe_media(path: Path) -> MediaInfo:
    ffprobe = require_command("ffprobe")
    proc = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(proc.stdout)
    duration = float(data.get("format", {}).get("duration") or 0)
    video_stream = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), {})
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    return MediaInfo(
        duration_ms=int(round(duration * 1000)),
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        fps=fps,
    )


def detect_silences(
    path: Path,
    *,
    noise_db: int = -35,
    min_duration_seconds: float = 0.4,
) -> list[SilenceRange]:
    ffmpeg = require_command("ffmpeg")
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_duration_seconds}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip()
        raise FfmpegError(stderr[-1200:] or "Silence detection failed")
    return parse_silencedetect_output(proc.stderr)


def extract_audio_segment(
    source: Path,
    output: Path,
    segment: MediaSegment,
    *,
    sample_rate: int = 16000,
    bitrate: str = "64k",
) -> Path:
    ffmpeg = require_command("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
            "-y",
            "-ss",
            _seconds_text(segment.start_ms),
            "-t",
            _seconds_text(segment.duration_ms),
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-b:a",
            bitrate,
            str(output),
        ]
    )
    return output


def extract_audio_segments(
    source: Path,
    output_dir: Path,
    segments: Sequence[MediaSegment],
    *,
    stem: str | None = None,
) -> list[Path]:
    name = stem or source.stem
    return [
        extract_audio_segment(source, output_dir / f"{name}.part{segment.index:03}.m4a", segment)
        for segment in segments
    ]


def parse_silencedetect_output(text: str) -> list[SilenceRange]:
    silences: list[SilenceRange] = []
    current_start_ms: int | None = None
    for line in text.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            current_start_ms = _seconds_to_ms(start_match.group(1))
            continue

        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and current_start_ms is not None:
            end_ms = _seconds_to_ms(end_match.group(1))
            if end_ms > current_start_ms:
                silences.append(SilenceRange(start_ms=current_start_ms, end_ms=end_ms))
            current_start_ms = None
    return silences


def _parse_fps(text: str | None) -> float | None:
    if not text:
        return None
    if "/" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    numerator, denominator = text.split("/", 1)
    try:
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    except ValueError:
        return None


def _seconds_to_ms(text: str) -> int:
    return int(round(float(text) * 1000))


def _seconds_text(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"
