import json

from caption_core.domain.srt import Cue
from caption_core.domain.terms import Replacement
from caption_core.pipeline.review import (
    approved_replacements_from_rows,
    build_term_review_payload,
    build_term_review_rows,
)


def test_build_term_review_rows_adds_impact_preview_and_disables_unmatched() -> None:
    cues = [
        Cue(index="1", timing="00:00:00,000 --> 00:00:01,000", lines=["T E D 患者"]),
        Cue(index="2", timing="00:00:01,000 --> 00:00:02,000", lines=["无需修改"]),
    ]

    rows = build_term_review_rows(cues, [Replacement("T E D", "TED"), Replacement("不存在", "命中不了")])

    assert rows[0].enabled is True
    assert rows[0].affected_cue_count == 1
    assert rows[0].replacement_count == 1
    assert rows[0].affected[0]["after"] == "TED 患者"
    assert rows[1].enabled is False
    assert rows[1].affected == []


def test_build_term_review_payload_loads_terms_and_srt(tmp_path) -> None:
    terms_path = tmp_path / "terms.json"
    srt_path = tmp_path / "caption.srt"
    terms_path.write_text(
        json.dumps({"replacements": [{"wrong": "朱成芳", "correct": "朱晨芳"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n朱成芳主任\n\n",
        encoding="utf-8",
    )

    payload = build_term_review_payload(run_id="r1", terms_path=terms_path, srt_path=srt_path)

    assert payload["run_id"] == "r1"
    assert payload["replacement_count"] == 1
    assert payload["affected_replacement_count"] == 1
    assert payload["affected_cue_count"] == 1
    assert payload["replacements"][0]["affected"][0]["before"] == "朱成芳主任"


def test_approved_replacements_from_rows_filters_disabled_invalid_and_duplicates() -> None:
    approved = approved_replacements_from_rows(
        [
            {"enabled": True, "wrong": "A", "correct": "B", "note": "ok"},
            {"enabled": True, "wrong": "A", "correct": "B", "note": "duplicate"},
            {"enabled": False, "wrong": "C", "correct": "D"},
            {"enabled": True, "wrong": "E", "correct": "E"},
        ]
    )

    assert approved == [{"wrong": "A", "correct": "B", "confidence": "", "evidence": "", "note": "ok"}]
