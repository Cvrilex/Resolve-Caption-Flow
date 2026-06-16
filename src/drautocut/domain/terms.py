from __future__ import annotations

import json
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

