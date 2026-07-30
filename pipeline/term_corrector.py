#!/usr/bin/env python3
"""Apply an auditable medical terminology replacement map to an SRT file."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
DEFAULT_TERMS = REPO_ROOT / "resources" / "terms.sample.json"
DEFAULT_WORK_DIR = REPO_ROOT / "data" / "work"


@dataclass(frozen=True)
class Replacement:
    wrong: str
    correct: str
    note: str = ""
    aliases: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    confidence: str = ""
    evidence: str = ""


@dataclass
class Cue:
    index: str
    timing: str
    lines: list[str]


class TermCorrectionError(RuntimeError):
    pass


def parse_srt(path: Path) -> list[Cue]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in text.split("\n\n"):
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        cues.append(Cue(index=lines[0].strip(), timing=lines[1].strip(), lines=lines[2:]))
    if not cues:
        raise TermCorrectionError(f"No SRT cues found in {path}")
    return cues


def write_srt(cues: list[Cue], path: Path) -> None:
    blocks = []
    for cue_number, cue in enumerate(cues, start=1):
        blocks.append("\n".join([str(cue_number), cue.timing, *cue.lines]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def load_replacements(path: Path) -> list[Replacement]:
    data = json.loads(path.read_text(encoding="utf-8"))
    replacements: list[Replacement] = []

    if isinstance(data, dict) and isinstance(data.get("replacements"), list):
        for item in data["replacements"]:
            aliases = tuple(str(v).strip() for v in item.get("aliases", []) or [] if str(v).strip())
            patterns = tuple(str(v).strip() for v in item.get("patterns", []) or [] if str(v).strip())
            replacements.append(
                Replacement(
                    wrong=str(item.get("wrong", "")),
                    correct=str(item.get("correct", "")),
                    note=str(item.get("note", "")),
                    aliases=aliases,
                    patterns=patterns,
                    confidence=str(item.get("confidence", "")),
                    evidence=str(item.get("evidence", "")),
                )
            )
    elif isinstance(data, dict) and isinstance(data.get("terms"), list):
        for item in data["terms"]:
            correct = str(item.get("term", "") or item.get("correct", ""))
            for alias in item.get("aliases", []) or []:
                replacements.append(Replacement(wrong=str(alias), correct=correct, note=str(item.get("note", ""))))
    elif isinstance(data, dict):
        for wrong, correct in data.items():
            replacements.append(Replacement(wrong=str(wrong), correct=str(correct)))
    else:
        raise TermCorrectionError("Terms file must be a JSON object")

    cleaned = [r for r in replacements if r.wrong and r.correct and r.wrong != r.correct]
    return sorted(cleaned, key=lambda r: len(r.wrong), reverse=True)


def _chinese_under_1000(number: int) -> str:
    units = ["", "十", "百"]
    digits = "零一二三四五六七八九"
    if number <= 0 or number >= 1000:
        return ""
    parts: list[str] = []
    zero_pending = False
    value = number
    for place in (100, 10, 1):
        digit = value // place
        value %= place
        if digit:
            if zero_pending:
                parts.append("零")
                zero_pending = False
            if not (place == 10 and digit == 1 and not parts):
                parts.append(digits[digit])
            parts.append(units[len(str(place)) - 1])
        elif parts and value:
            zero_pending = True
    return "".join(parts)


_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _normalize_number_token(token: str) -> str | None:
    value = token.strip().translate(str.maketrans("０１２３４５６７８９．", "0123456789."))
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return value
    parsed = _chinese_to_int_under_1000(value)
    return str(parsed) if parsed is not None else None


def _chinese_to_int_under_1000(text: str) -> int | None:
    value = text.strip().replace(" ", "")
    if not value or not re.fullmatch(r"[零〇一二两三四五六七八九十百]+", value):
        return None

    if "十" not in value and "百" not in value:
        digits: list[str] = []
        for char in value:
            digit = _CHINESE_DIGITS.get(char)
            if digit is None:
                return None
            digits.append(str(digit))
        return int("".join(digits))

    total = 0
    rest = value
    had_hundred = False
    if "百" in rest:
        left, rest = rest.split("百", 1)
        hundred = 1 if left == "" else _CHINESE_DIGITS.get(left)
        if hundred is None or hundred <= 0:
            return None
        total += hundred * 100
        had_hundred = True

    rest = rest.lstrip("零〇")
    if not rest:
        return total

    if "十" in rest:
        left, right = rest.split("十", 1)
        ten = 1 if left == "" else _CHINESE_DIGITS.get(left)
        if ten is None or ten <= 0:
            return None
        total += ten * 10
        if right:
            one = _CHINESE_DIGITS.get(right)
            if one is None:
                return None
            total += one
        return total

    one = _CHINESE_DIGITS.get(rest)
    if one is None:
        return None
    total += one * 10 if had_hundred else one
    return total


def _split_joined_chinese_bp_pair(token: str) -> tuple[int, int] | None:
    value = token.strip().replace(" ", "")
    if value.count("百") < 2:
        return None
    for split_at in range(2, len(value) - 1):
        systolic = _chinese_to_int_under_1000(value[:split_at])
        diastolic = _chinese_to_int_under_1000(value[split_at:])
        if systolic is None or diastolic is None:
            continue
        if 50 <= systolic <= 300 and 30 <= diastolic <= 200 and systolic > diastolic:
            return systolic, diastolic
    return None


_BP_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百]+)"
_BP_GAP = r"[ \t]*"
_BP_PAIR_CONNECTOR = r"(?:或|/|／|over|欧文|欧尔)"
_BP_RANGE_CONNECTOR = r"(?:~|～|-|到|至)"
_BP_UNIT = rf"(?:个?毫米汞柱|mm{_BP_GAP}(?:汞柱|hg))"
_BUILTIN_MEDICAL_TERMS: tuple[tuple[str, str, str], ...] = (
    ("消普纳", "硝普钠", "药物名音近误识别归一"),
    ("硝普纳", "硝普钠", "药物名音近误识别归一"),
    ("阿贝维尔", "拉贝洛尔", "药物名音近误识别归一"),
    ("阿贝尔", "拉贝洛尔", "药物名音近误识别归一"),
    ("拉贝落尔", "拉贝洛尔", "药物名音近误识别归一"),
    ("艾斯维尔", "艾司洛尔", "药物名音近误识别归一"),
    ("艾斯洛尔", "艾司洛尔", "药物名音近误识别归一"),
    ("无法比尔", "乌拉地尔", "药物名音近误识别归一"),
    ("乌拉比尔", "乌拉地尔", "药物名音近误识别归一"),
    ("乌拉迪尔", "乌拉地尔", "药物名音近误识别归一"),
    ("不花地尔", "乌拉地尔", "药物名音近误识别归一"),
    ("尼卡的平", "尼卡地平", "药物名音近误识别归一"),
    ("尼卡地坪", "尼卡地平", "药物名音近误识别归一"),
    ("可乐啶", "可乐定", "药物名音近误识别归一"),
    ("可乐丁", "可乐定", "药物名音近误识别归一"),
    ("夫塞米", "呋塞米", "药物名音近误识别归一"),
    ("福塞米", "呋塞米", "药物名音近误识别归一"),
    ("敷塞米", "呋塞米", "药物名音近误识别归一"),
    ("儿查酚胺", "儿茶酚胺", "医学名词音近误识别归一"),
    ("蛛网膜下枪出血", "蛛网膜下腔出血", "医学名词音近误识别归一"),
    ("嗜铬细包瘤", "嗜铬细胞瘤", "医学名词音近误识别归一"),
    ("先兆子闲", "先兆子痫", "医学名词音近误识别归一"),
    ("尿漪留", "尿潴留", "医学名词音近误识别归一"),
    ("尿储留", "尿潴留", "医学名词音近误识别归一"),
    ("非笛体", "非甾体", "医学名词音近误识别归一"),
    ("A C E I", "ACEI", "医学缩写空格归一"),
    ("B N P", "BNP", "医学缩写空格归一"),
)


def _mmhg_patterns(correct: str) -> list[str]:
    single = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*mm\s*hg\s*", correct, flags=re.IGNORECASE)
    pair = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*mm\s*hg\s*",
        correct,
        flags=re.IGNORECASE,
    )
    if not single and not pair:
        return []
    if pair:
        systolic, diastolic = pair.group(1), pair.group(2)
        patterns = [
            rf"(?<![A-Za-z0-9.]){re.escape(systolic)}{_BP_GAP}{_BP_PAIR_CONNECTOR}{_BP_GAP}{re.escape(diastolic)}{_BP_GAP}{_BP_UNIT}(?![A-Za-z0-9])",
        ]
        if systolic.isdigit() and diastolic.isdigit():
            sys_chinese = _chinese_under_1000(int(systolic))
            dia_chinese = _chinese_under_1000(int(diastolic))
            if sys_chinese and dia_chinese:
                patterns.append(
                    rf"{re.escape(sys_chinese)}{_BP_GAP}{_BP_PAIR_CONNECTOR}{_BP_GAP}{re.escape(dia_chinese)}{_BP_GAP}{_BP_UNIT}"
                )
        return patterns

    number = single.group(1)
    escaped = re.escape(number)
    patterns = [
        rf"(?<![A-Za-z0-9.]){escaped}{_BP_GAP}mm{_BP_GAP}hg(?![A-Za-z0-9])",
        rf"(?<![A-Za-z0-9.]){escaped}{_BP_GAP}mm{_BP_GAP}汞柱",
        rf"(?<![A-Za-z0-9.]){escaped}{_BP_GAP}个?毫米汞柱",
    ]
    if number.isdigit():
        chinese = _chinese_under_1000(int(number))
        if chinese:
            patterns.append(rf"{re.escape(chinese)}{_BP_GAP}个?毫米汞柱")
    return patterns


def _literal_candidates(replacement: Replacement) -> list[str]:
    candidates = [replacement.wrong, *replacement.aliases]
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = str(candidate).strip()
        if not value or value == replacement.correct or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _pattern_candidates(replacement: Replacement) -> list[str]:
    patterns = [*replacement.patterns, *_mmhg_patterns(replacement.correct)]
    deduped: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        value = str(pattern).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def builtin_unit_normalizations(text: str) -> tuple[str, list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []

    def replace_range_bp(match: re.Match[str]) -> str:
        original = match.group(0)
        low = _normalize_number_token(match.group("low"))
        high = _normalize_number_token(match.group("high"))
        if low is None or high is None:
            return original
        connector = match.group("connector")
        replacement = f"{low}{connector}{high}mmHg"
        if original != replacement:
            changes.append(
                {
                    "wrong": original,
                    "correct": replacement,
                    "count": 1,
                    "matched_by": ["builtin:mmHg_range"],
                    "note": "血压单位范围格式标准化",
                }
            )
        return replacement

    def replace_quoted_bp(match: re.Match[str]) -> str:
        original = match.group(0)
        systolic = _normalize_number_token(match.group("sys"))
        diastolic = _normalize_number_token(match.group("dia"))
        if systolic is None or diastolic is None:
            return original
        replacement = f"{systolic}/{diastolic}mmHg"
        if original != replacement:
            changes.append(
                {
                    "wrong": original,
                    "correct": replacement,
                    "count": 1,
                    "matched_by": ["builtin:mmHg_pair"],
                    "note": "血压单位格式标准化",
                }
            )
        return replacement

    def replace_joined_chinese_bp(match: re.Match[str]) -> str:
        original = match.group(0)
        pair = _split_joined_chinese_bp_pair(match.group("pair"))
        if pair is None:
            return original
        systolic, diastolic = pair
        replacement = f"{systolic}/{diastolic}mmHg"
        if original != replacement:
            changes.append(
                {
                    "wrong": original,
                    "correct": replacement,
                    "count": 1,
                    "matched_by": ["builtin:mmHg_joined_pair"],
                    "note": "血压单位格式标准化",
                }
            )
        return replacement

    text = re.sub(
        rf"(?<![A-Za-z0-9.])(?P<pair>[零〇一二两三四五六七八九十百]{{4,}}){_BP_GAP}毫米汞柱(?![A-Za-z0-9])",
        replace_joined_chinese_bp,
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        rf"(?<![A-Za-z0-9.])(?P<low>{_BP_NUMBER}){_BP_GAP}(?P<connector>{_BP_RANGE_CONNECTOR}){_BP_GAP}(?P<high>{_BP_NUMBER}){_BP_GAP}的?{_BP_GAP}{_BP_UNIT}(?![A-Za-z0-9])",
        replace_range_bp,
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        rf"(?<![A-Za-z0-9.])(?P<sys>{_BP_NUMBER}){_BP_GAP}{_BP_PAIR_CONNECTOR}{_BP_GAP}(?P<dia>{_BP_NUMBER}){_BP_GAP}{_BP_UNIT}(?![A-Za-z0-9])",
        replace_quoted_bp,
        text,
        flags=re.IGNORECASE,
    )

    def replace_single_mmhg(match: re.Match[str]) -> str:
        original = match.group(0)
        number = _normalize_number_token(match.group("num"))
        if number is None:
            return original
        replacement = f"{number}mmHg"
        if original != replacement:
            changes.append(
                {
                    "wrong": original,
                    "correct": replacement,
                    "count": 1,
                    "matched_by": ["builtin:mmHg_single"],
                    "note": "血压单位格式标准化",
                }
            )
        return replacement

    text = re.sub(
        rf"(?<![A-Za-z0-9.])(?P<num>{_BP_NUMBER}){_BP_GAP}{_BP_UNIT}(?![A-Za-z0-9])",
        replace_single_mmhg,
        text,
        flags=re.IGNORECASE,
    )
    return text, changes


def builtin_medical_term_normalizations(text: str) -> tuple[str, list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    for wrong, correct, note in sorted(_BUILTIN_MEDICAL_TERMS, key=lambda item: len(item[0]), reverse=True):
        count = text.count(wrong)
        if not count:
            continue
        text = text.replace(wrong, correct)
        changes.append(
            {
                "wrong": wrong,
                "correct": correct,
                "count": count,
                "matched_by": ["builtin:medical_term"],
                "note": note,
            }
        )
    return text, changes


def apply_replacements(cues: list[Cue], replacements: list[Replacement]) -> tuple[list[Cue], list[dict[str, Any]], list[dict[str, Any]]]:
    report: list[dict[str, Any]] = []
    corrected: list[Cue] = []
    replacement_stats: dict[tuple[str, str], int] = {(r.wrong, r.correct): 0 for r in replacements}
    for cue in cues:
        before_text = "\n".join(cue.lines)
        after_text = before_text
        cue_changes: list[dict[str, Any]] = []
        for replacement in replacements:
            replacement_count = 0
            matched_by: list[str] = []
            for literal in _literal_candidates(replacement):
                count = after_text.count(literal)
                if not count:
                    continue
                after_text = after_text.replace(literal, replacement.correct)
                replacement_count += count
                matched_by.append(literal)
            for pattern in _pattern_candidates(replacement):
                try:
                    updated, count = re.subn(pattern, replacement.correct, after_text, flags=re.IGNORECASE)
                except re.error as exc:
                    cue_changes.append(
                        {
                            "wrong": replacement.wrong,
                            "correct": replacement.correct,
                            "count": 0,
                            "note": replacement.note,
                            "error": f"invalid pattern {pattern!r}: {exc}",
                        }
                    )
                    continue
                if count and updated != after_text:
                    after_text = updated
                    replacement_count += count
                    matched_by.append(f"pattern:{pattern}")
            if replacement_count:
                replacement_stats[(replacement.wrong, replacement.correct)] += replacement_count
                cue_changes.append(
                    {
                        "wrong": replacement.wrong,
                        "correct": replacement.correct,
                        "count": replacement_count,
                        "matched_by": matched_by,
                        "note": replacement.note,
                    }
                )
        after_text, builtin_term_changes = builtin_medical_term_normalizations(after_text)
        cue_changes.extend(builtin_term_changes)
        after_text, builtin_unit_changes = builtin_unit_normalizations(after_text)
        cue_changes.extend(builtin_unit_changes)
        corrected.append(Cue(index=cue.index, timing=cue.timing, lines=after_text.split("\n")))
        if cue_changes:
            report.append(
                {
                    "cue": cue.index,
                    "timing": cue.timing,
                    "before": before_text,
                    "after": after_text,
                    "changes": cue_changes,
                }
            )
    unmatched = [
        {
            "wrong": replacement.wrong,
            "correct": replacement.correct,
            "aliases": list(replacement.aliases),
            "patterns": list(replacement.patterns),
            "auto_patterns": _mmhg_patterns(replacement.correct),
            "note": replacement.note,
            "confidence": replacement.confidence,
            "evidence": replacement.evidence,
        }
        for replacement in replacements
        if replacement_stats[(replacement.wrong, replacement.correct)] == 0
    ]
    return corrected, report, unmatched


def correct_srt(srt: Path, terms: Path, output: Path | None, report_path: Path | None) -> dict[str, Any]:
    cues = parse_srt(srt)
    replacements = load_replacements(terms)
    corrected, report, unmatched = apply_replacements(cues, replacements)

    if output is None:
        output = srt.with_name(f"{srt.stem}.corrected.srt")
    if report_path is None:
        report_path = DEFAULT_WORK_DIR / f"{srt.stem}.correction-report.json"

    write_srt(corrected, output)
    report_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_srt": str(srt),
        "terms": str(terms),
        "output_srt": str(output),
        "cue_count": len(cues),
        "changed_cue_count": len(report),
        "replacement_count": sum(change["count"] for item in report for change in item["changes"]),
        "unmatched_replacement_count": len(unmatched),
        "unmatched_replacements": unmatched,
        "changes": report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply medical terminology replacements to an SRT file.")
    parser.add_argument("--srt", required=True, help="Input SRT path.")
    parser.add_argument("--terms", default=str(DEFAULT_TERMS), help="Replacement map JSON path.")
    parser.add_argument("--output", help="Corrected SRT path. Defaults beside the input SRT.")
    parser.add_argument("--report", help="JSON correction report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = correct_srt(
        Path(args.srt).expanduser().resolve(),
        Path(args.terms).expanduser().resolve(),
        Path(args.output).expanduser().resolve() if args.output else None,
        Path(args.report).expanduser().resolve() if args.report else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
