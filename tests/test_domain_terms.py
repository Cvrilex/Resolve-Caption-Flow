from drautocut.domain.srt import Cue
from drautocut.domain.terms import Replacement, apply_replacements, preview_replacements, replacements_from_payload


def test_replacements_from_payload_sorts_longer_wrong_terms_first() -> None:
    replacements = replacements_from_payload(
        {
            "replacements": [
                {"wrong": "TED", "correct": "甲状腺相关眼病"},
                {"wrong": "T E D", "correct": "TED"},
                {"wrong": "", "correct": "ignored"},
            ]
        }
    )

    assert [item.wrong for item in replacements] == ["T E D", "TED"]


def test_apply_replacements_returns_corrected_cues_and_audit_report() -> None:
    cues = [
        Cue(index="1", timing="00:00:00,000 --> 00:00:02,000", lines=["T E D 患者"]),
        Cue(index="2", timing="00:00:02,000 --> 00:00:03,000", lines=["无需修改"]),
    ]

    corrected, report = apply_replacements(cues, [Replacement("T E D", "TED", "标准缩写")])

    assert corrected[0].lines == ["TED 患者"]
    assert corrected[1].lines == ["无需修改"]
    assert report == [
        {
            "cue": "1",
            "timing": "00:00:00,000 --> 00:00:02,000",
            "before": "T E D 患者",
            "after": "TED 患者",
            "changes": [{"wrong": "T E D", "correct": "TED", "count": 1, "note": "标准缩写"}],
        }
    ]


def test_preview_replacements_lists_affected_cues_without_mutating() -> None:
    cues = [
        Cue(index="1", timing="00:00:00,000 --> 00:00:01,000", lines=["朱成芳主任"]),
        Cue(index="2", timing="00:00:01,000 --> 00:00:02,000", lines=["朱成芳医生"]),
    ]

    preview = preview_replacements(cues, [Replacement("朱成芳", "朱晨芳")])

    assert preview[0]["affected_cue_count"] == 2
    assert preview[0]["replacement_count"] == 2
    assert preview[0]["affected"][0]["after"] == "朱晨芳主任"
    assert cues[0].lines == ["朱成芳主任"]

