#!/usr/bin/env python3
"""Normalize course metadata codes in SRT files using PDF filename rules."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # Supports both package imports and direct script execution.
    from .term_corrector import Cue, parse_srt, write_srt
except ImportError:  # pragma: no cover - direct script execution path
    from term_corrector import Cue, parse_srt, write_srt


CHINESE_DIGITS = "零一二三四五六七八九"


@dataclass(frozen=True)
class CourseCodeRule:
    standard: str
    year: str
    month: int
    day: int
    serial: int
    serial_width: int
    level: str = ""
    source: str = ""


def extract_course_code_from_filename(path: Path | str) -> CourseCodeRule | None:
    """Extract a course code such as 2026-03-08-018（国） from a filename."""
    source = Path(path)
    stem = source.stem
    pattern = re.compile(
        r"(?P<year>20\d{2})"
        r"[-_./年\s]*"
        r"(?P<month>\d{1,2})"
        r"[-_./月\s]*"
        r"(?P<day>\d{1,2})"
        r"[-_./日\s]*"
        r"(?P<serial>\d{1,4})"
        r"\s*(?:[（(]\s*(?P<level>[\u4e00-\u9fffA-Za-z]{1,8})\s*[）)]|(?P<bare_level>国))?"
    )
    for match in pattern.finditer(stem):
        month = int(match.group("month"))
        day = int(match.group("day"))
        serial = int(match.group("serial"))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        level = match.group("level") or match.group("bare_level") or ""
        serial_text = match.group("serial")
        serial_width = max(3, len(serial_text))
        standard = f"{match.group('year')}-{month:02d}-{day:02d}-{serial:0{serial_width}d}"
        if level:
            standard += f"（{level}）"
        return CourseCodeRule(
            standard=standard,
            year=match.group("year"),
            month=month,
            day=day,
            serial=serial,
            serial_width=serial_width,
            level=level,
            source=str(source),
        )
    return None


def _digit_by_digit(number_text: str) -> str:
    return "".join(CHINESE_DIGITS[int(ch)] for ch in number_text)


def _digit_by_digit_pattern(number_text: str) -> str:
    parts: list[str] = []
    for char in number_text:
        if char == "0":
            parts.append("[零〇]")
        else:
            parts.append(CHINESE_DIGITS[int(char)])
    return "".join(parts)


def _cn_number_under_100(number: int) -> str:
    if number < 10:
        return CHINESE_DIGITS[number]
    if number < 20:
        return "十" + (CHINESE_DIGITS[number % 10] if number % 10 else "")
    tens, ones = divmod(number, 10)
    return CHINESE_DIGITS[tens] + "十" + (CHINESE_DIGITS[ones] if ones else "")


def _optional_zero_cn(number: int) -> str:
    return f"(?:{_cn_number_under_100(number)}|[零〇]{CHINESE_DIGITS[number]})" if number < 10 else _cn_number_under_100(number)


def _serial_cn_alternatives(serial: int, width: int) -> list[str]:
    padded = f"{serial:0{width}d}"
    variants = {_digit_by_digit_pattern(padded), _cn_number_under_100(serial)}
    if serial < 100 and padded.startswith("0"):
        variants.add("[零〇]" + _cn_number_under_100(serial))
    return sorted(variants, key=len, reverse=True)


def _course_code_pattern(rule: CourseCodeRule) -> re.Pattern[str]:
    year = re.escape(rule.year)
    compact_digits = f"{rule.year}{rule.month:02d}{rule.day:02d}{rule.serial:0{rule.serial_width}d}"
    sep = r"[\s\-_/／.．]*"
    date_word_sep = r"[\s\-_/／.．]*(?:年|月|日)?[\s\-_/／.．]*"
    if rule.level:
        level_tokens = [re.escape(rule.level)]
        if rule.level == "国":
            level_tokens.extend(["国家级", "国字号"])
        level = rf"(?:[（(]?\s*(?:{'|'.join(level_tokens)})\s*[）)]?)?"
    else:
        level = ""

    month_ar = f"0?{rule.month}"
    day_ar = f"0?{rule.day}"
    serial_ar = f"0*{rule.serial}"
    arabic = rf"(?<!\d){year}(?:{sep}|年){month_ar}(?:{sep}|月){day_ar}(?:{sep}|日){serial_ar}{level}(?!\d)"
    compact_arabic = rf"(?<!\d){re.escape(compact_digits)}{level}(?!\d)"

    year_cn = _digit_by_digit_pattern(rule.year)
    month_cn = _optional_zero_cn(rule.month)
    day_cn = _optional_zero_cn(rule.day)
    serial_cn = "|".join(_serial_cn_alternatives(rule.serial, rule.serial_width))
    chinese = rf"{year_cn}{date_word_sep}{month_cn}{date_word_sep}{day_cn}{date_word_sep}(?:{serial_cn}){level}"

    return re.compile(rf"(?:{compact_arabic}|{arabic}|{chinese})")


def normalize_text_with_course_code(text: str, rule: CourseCodeRule) -> tuple[str, int]:
    pattern = _course_code_pattern(rule)
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return rule.standard

    normalized = pattern.sub(repl, text)
    if normalized == text:
        count = 0
    return normalized, count


def normalize_srt_course_code(
    srt: Path,
    output: Path,
    report_path: Path,
    *,
    context: Path | None = None,
    rule: CourseCodeRule | None = None,
) -> dict[str, Any]:
    active_rule = rule or (extract_course_code_from_filename(context) if context else None)
    cues = parse_srt(srt)
    changed: list[dict[str, Any]] = []
    replacement_count = 0

    if active_rule:
        for cue in cues:
            before = "\n".join(cue.lines)
            after, count = normalize_text_with_course_code(before, active_rule)
            if count and after != before:
                cue.lines = after.split("\n")
                replacement_count += count
                changed.append({"cue": cue.index, "before": before, "after": after, "replacement_count": count})

    write_srt(cues, output)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_srt": str(srt),
        "output_srt": str(output),
        "source_context": str(context) if context else None,
        "rule": active_rule.__dict__ if active_rule else None,
        "changed_cue_count": len(changed),
        "replacement_count": replacement_count,
        "changed_cues": changed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize course code variants in an SRT file.")
    parser.add_argument("--srt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--context", help="PDF/course filename used to extract the standard course code.")
    args = parser.parse_args()
    result = normalize_srt_course_code(
        Path(args.srt),
        Path(args.output),
        Path(args.report),
        context=Path(args.context) if args.context else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
