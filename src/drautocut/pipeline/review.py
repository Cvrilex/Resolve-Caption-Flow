from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drautocut.domain import Cue, Replacement, parse_srt, preview_replacements, replacements_from_payload


@dataclass(frozen=True)
class TermReviewRow:
    id: int
    enabled: bool
    wrong: str
    correct: str
    confidence: str
    evidence: str
    note: str
    affected_cue_count: int
    replacement_count: int
    affected: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "enabled": self.enabled,
            "wrong": self.wrong,
            "correct": self.correct,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "note": self.note,
            "affected_cue_count": self.affected_cue_count,
            "replacement_count": self.replacement_count,
            "affected": self.affected,
        }


def build_term_review_rows(
    cues: list[Cue],
    replacements: list[Replacement],
    *,
    include_unmatched: bool = True,
) -> list[TermReviewRow]:
    rows: list[TermReviewRow] = []
    for index, item in enumerate(preview_replacements(cues, replacements)):
        affected_cue_count = int(item["affected_cue_count"])
        if affected_cue_count == 0 and not include_unmatched:
            continue
        rows.append(
            TermReviewRow(
                id=index,
                enabled=affected_cue_count > 0,
                wrong=str(item["wrong"]),
                correct=str(item["correct"]),
                confidence=str(item["confidence"]),
                evidence=str(item["evidence"]),
                note=str(item["note"]),
                affected_cue_count=affected_cue_count,
                replacement_count=int(item["replacement_count"]),
                affected=list(item["affected"]),
            )
        )
    return rows


def build_term_review_payload(
    *,
    run_id: str,
    terms_path: Path,
    srt_path: Path,
    include_unmatched: bool = True,
) -> dict[str, Any]:
    terms_payload = json.loads(terms_path.read_text(encoding="utf-8"))
    replacements = replacements_from_payload(terms_payload)
    cues = parse_srt(srt_path)
    rows = build_term_review_rows(cues, replacements, include_unmatched=include_unmatched)
    return {
        "run_id": run_id,
        "terms_path": str(terms_path),
        "srt_path": str(srt_path),
        "replacement_count": len(rows),
        "affected_replacement_count": sum(1 for row in rows if row.affected_cue_count > 0),
        "affected_cue_count": len({hit["cue"] for row in rows for hit in row.affected}),
        "replacements": [row.to_dict() for row in rows],
    }


def approved_replacements_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    replacements: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not row.get("enabled", True):
            continue
        wrong = str(row.get("wrong", "")).strip()
        correct = str(row.get("correct", "")).strip()
        if not wrong or not correct or wrong == correct:
            continue
        key = (wrong, correct)
        if key in seen:
            continue
        seen.add(key)
        replacements.append(
            {
                "wrong": wrong,
                "correct": correct,
                "confidence": str(row.get("confidence", "")),
                "evidence": str(row.get("evidence", "")),
                "note": str(row.get("note", "")),
            }
        )
    return replacements
