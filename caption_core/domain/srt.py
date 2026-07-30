from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cue:
    index: str
    timing: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def single_line_text(self) -> str:
        return " ".join(self.lines)


class SrtError(RuntimeError):
    pass


def parse_srt_text(text: str) -> list[Cue]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in normalized.split("\n\n"):
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        cues.append(Cue(index=lines[0].strip(), timing=lines[1].strip(), lines=lines[2:]))
    if not cues:
        raise SrtError("No SRT cues found")
    return cues


def parse_srt(path: Path) -> list[Cue]:
    try:
        return parse_srt_text(path.read_text(encoding="utf-8-sig"))
    except SrtError as exc:
        raise SrtError(f"No SRT cues found in {path}") from exc


def render_srt(cues: list[Cue]) -> str:
    blocks = []
    for cue_number, cue in enumerate(cues, start=1):
        blocks.append("\n".join([str(cue_number), cue.timing, *cue.lines]))
    return "\n\n".join(blocks) + "\n"


def write_srt(cues: list[Cue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_srt(cues), encoding="utf-8")


def timestamp_to_ms(timestamp: str) -> int:
    hours, minutes, rest = timestamp.split(":")
    seconds, milliseconds = rest.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(milliseconds)
    )


def ms_to_timestamp(milliseconds: int) -> str:
    hours = milliseconds // 3_600_000
    minutes = (milliseconds % 3_600_000) // 60_000
    seconds = (milliseconds % 60_000) // 1_000
    ms_remainder = milliseconds % 1_000
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms_remainder:03}"


def split_timing(timing: str) -> tuple[str, str]:
    start, end = timing.split(" --> ", 1)
    return start, end

