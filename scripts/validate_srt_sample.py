#!/usr/bin/env python3
"""Validate subtitle sample metrics for repeatable manual/CI checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def timestamp_to_ms(timestamp: str) -> int:
    hours, minutes, rest = timestamp.split(":")
    seconds, milliseconds = rest.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(milliseconds)
    )


def parse_srt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        start, end = [part.strip() for part in lines[1].split(" --> ", 1)]
        cue_text = "".join(lines[2:])
        cues.append(
            {
                "index": lines[0].strip(),
                "timing": lines[1].strip(),
                "start_ms": timestamp_to_ms(start),
                "end_ms": timestamp_to_ms(end),
                "text": cue_text,
                "length": visible_len(cue_text),
            }
        )
    return cues


def summarize_srt(path: Path, min_chars: int, max_chars: int) -> dict[str, Any]:
    cues = parse_srt(path)
    bad_timing: list[dict[str, Any]] = []
    previous_end = -1
    for cue in cues:
        if cue["start_ms"] < previous_end or cue["end_ms"] < cue["start_ms"]:
            bad_timing.append({"index": cue["index"], "timing": cue["timing"]})
        previous_end = cue["end_ms"]

    text = path.read_text(encoding="utf-8-sig")
    lengths = [cue["length"] for cue in cues]
    return {
        "srt": str(path),
        "cue_count": len(cues),
        "under_min_count": sum(1 for length in lengths if 0 < length < min_chars),
        "over_max_count": sum(1 for length in lengths if length > max_chars),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "bad_timing_count": len(bad_timing),
        "bad_timing": bad_timing[:20],
        "last_end_ms": cues[-1]["end_ms"] if cues else 0,
        "contains_mmhg": text.count("mmHg"),
        "contains_chinese_mmhg": text.count("毫米汞柱"),
        "contains_spaced_mm_hg": len(re.findall(r"mm\s+Hg", text, flags=re.IGNORECASE)),
    }


def load_report_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        "cue_count_before",
        "cue_count_after",
        "short_detected_count",
        "short_window_count",
        "short_changed_window_count",
        "overlong_detected_count",
        "overlong_changed_cue_count",
        "llm_fallback_error_count",
        "punctuation_changed_cue_count",
    ]
    return {key: data.get(key) for key in keys if key in data}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SRT sample metrics.")
    parser.add_argument("--srt", required=True, help="SRT file to inspect.")
    parser.add_argument("--report", help="Optional subtitle optimization report JSON.")
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=20)
    parser.add_argument("--max-under-min", type=int)
    parser.add_argument("--max-over-max", type=int)
    parser.add_argument("--max-bad-timing", type=int, default=0)
    parser.add_argument("--max-chinese-mmhg", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    srt_path = Path(args.srt).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if args.report else None
    summary = summarize_srt(srt_path, args.min_chars, args.max_chars)
    report_summary = load_report_summary(report_path)
    if report_summary is not None:
        summary["report"] = report_summary

    failures: list[str] = []
    if args.max_under_min is not None and summary["under_min_count"] > args.max_under_min:
        failures.append(f"under_min_count {summary['under_min_count']} > {args.max_under_min}")
    if args.max_over_max is not None and summary["over_max_count"] > args.max_over_max:
        failures.append(f"over_max_count {summary['over_max_count']} > {args.max_over_max}")
    if summary["bad_timing_count"] > args.max_bad_timing:
        failures.append(f"bad_timing_count {summary['bad_timing_count']} > {args.max_bad_timing}")
    if args.max_chinese_mmhg is not None and summary["contains_chinese_mmhg"] > args.max_chinese_mmhg:
        failures.append(f"contains_chinese_mmhg {summary['contains_chinese_mmhg']} > {args.max_chinese_mmhg}")

    summary["passed"] = not failures
    summary["failures"] = failures
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
