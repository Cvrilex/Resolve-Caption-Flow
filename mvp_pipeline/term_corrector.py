#!/usr/bin/env python3
"""Apply an auditable medical terminology replacement map to an SRT file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_TERMS = ROOT / "terms.sample.json"
DEFAULT_WORK_DIR = ROOT / "work"


@dataclass(frozen=True)
class Replacement:
    wrong: str
    correct: str
    note: str = ""


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
            replacements.append(
                Replacement(
                    wrong=str(item.get("wrong", "")),
                    correct=str(item.get("correct", "")),
                    note=str(item.get("note", "")),
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
    if not cleaned:
        raise TermCorrectionError(f"No valid replacements found in {path}")
    return sorted(cleaned, key=lambda r: len(r.wrong), reverse=True)


def apply_replacements(cues: list[Cue], replacements: list[Replacement]) -> tuple[list[Cue], list[dict[str, Any]]]:
    report: list[dict[str, Any]] = []
    corrected: list[Cue] = []
    for cue in cues:
        before_text = "\n".join(cue.lines)
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


def correct_srt(srt: Path, terms: Path, output: Path | None, report_path: Path | None) -> dict[str, Any]:
    cues = parse_srt(srt)
    replacements = load_replacements(terms)
    corrected, report = apply_replacements(cues, replacements)

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
