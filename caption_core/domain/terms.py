from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .srt import Cue


@dataclass(frozen=True)
class Replacement:
    wrong: str
    correct: str
    note: str = ""
    confidence: str = ""
    evidence: str = ""


class TermError(RuntimeError):
    pass


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

_BP_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百]+)"
_BP_GAP = r"[ \t]*"
_BP_UNIT = rf"(?:毫米汞柱|mm{_BP_GAP}(?:汞柱|hg))"


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


def builtin_unit_normalizations(text: str) -> tuple[str, list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []

    def replace_bp_pair(match: re.Match[str]) -> str:
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
        rf"(?<![A-Za-z0-9.])(?P<sys>{_BP_NUMBER}){_BP_GAP}(?:或|/|／){_BP_GAP}(?P<dia>{_BP_NUMBER}){_BP_GAP}{_BP_UNIT}(?![A-Za-z0-9])",
        replace_bp_pair,
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


def replacements_from_payload(data: Any) -> list[Replacement]:
    replacements: list[Replacement] = []

    if isinstance(data, dict) and isinstance(data.get("replacements"), list):
        for item in data["replacements"]:
            if not isinstance(item, dict):
                continue
            replacements.append(
                Replacement(
                    wrong=str(item.get("wrong", "")),
                    correct=str(item.get("correct", "")),
                    note=str(item.get("note", "")),
                    confidence=str(item.get("confidence", "")),
                    evidence=str(item.get("evidence", "")),
                )
            )
    elif isinstance(data, dict) and isinstance(data.get("terms"), list):
        for item in data["terms"]:
            if not isinstance(item, dict):
                continue
            correct = str(item.get("term", "") or item.get("correct", ""))
            for alias in item.get("aliases", []) or []:
                replacements.append(
                    Replacement(wrong=str(alias), correct=correct, note=str(item.get("note", "")))
                )
    elif isinstance(data, dict):
        for wrong, correct in data.items():
            replacements.append(Replacement(wrong=str(wrong), correct=str(correct)))
    else:
        raise TermError("Terms payload must be a JSON object")

    cleaned = [item for item in replacements if item.wrong and item.correct and item.wrong != item.correct]
    if not cleaned:
        raise TermError("No valid replacements found")
    return sorted(cleaned, key=lambda item: len(item.wrong), reverse=True)


def load_replacements(path: Path) -> list[Replacement]:
    try:
        return replacements_from_payload(json.loads(path.read_text(encoding="utf-8")))
    except TermError as exc:
        raise TermError(f"{exc}: {path}") from exc


def apply_replacements(cues: list[Cue], replacements: list[Replacement]) -> tuple[list[Cue], list[dict[str, Any]]]:
    report: list[dict[str, Any]] = []
    corrected: list[Cue] = []
    for cue in cues:
        before_text = cue.text
        after_text = before_text
        cue_changes: list[dict[str, Any]] = []
        for replacement in replacements:
            count = after_text.count(replacement.wrong)
            if not count:
                continue
            after_text = after_text.replace(replacement.wrong, replacement.correct)
            cue_changes.append(
                {
                    "wrong": replacement.wrong,
                    "correct": replacement.correct,
                    "count": count,
                    "note": replacement.note,
                }
            )
        after_text, builtin_changes = builtin_unit_normalizations(after_text)
        cue_changes.extend(builtin_changes)
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
    return corrected, report


def preview_replacements(cues: list[Cue], replacements: list[Replacement]) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for replacement in replacements:
        affected: list[dict[str, Any]] = []
        for cue in cues:
            before = cue.text
            count = before.count(replacement.wrong)
            if not count:
                continue
            affected.append(
                {
                    "cue": cue.index,
                    "timing": cue.timing,
                    "count": count,
                    "before": before,
                    "after": before.replace(replacement.wrong, replacement.correct),
                }
            )
        previews.append(
            {
                "wrong": replacement.wrong,
                "correct": replacement.correct,
                "note": replacement.note,
                "confidence": replacement.confidence,
                "evidence": replacement.evidence,
                "affected_cue_count": len(affected),
                "replacement_count": sum(item["count"] for item in affected),
                "affected": affected,
            }
        )
    return previews
