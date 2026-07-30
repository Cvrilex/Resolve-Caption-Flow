from caption_core.domain.srt import Cue
from caption_core.domain.terms import (
    Replacement,
    apply_replacements,
    builtin_unit_normalizations,
    preview_replacements,
    replacements_from_payload,
)


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


def test_builtin_unit_normalizations_handles_blood_pressure_variants() -> None:
    text = "血压达到180毫米汞柱，或者一百八十或一百二十毫米汞柱，也可能是140 mm Hg，还有二百一百一十毫米汞柱。"

    normalized, changes = builtin_unit_normalizations(text)

    assert normalized == "血压达到180mmHg，或者180/120mmHg，也可能是140mmHg，还有200/110mmHg。"
    assert [change["correct"] for change in changes] == ["200/110mmHg", "180/120mmHg", "180mmHg", "140mmHg"]


def test_apply_replacements_runs_builtin_unit_normalization_without_terms() -> None:
    cues = [
        Cue(index="1", timing="00:00:00,000 --> 00:00:02,000", lines=["超过一百八十毫米汞柱"]),
    ]

    corrected, report = apply_replacements(cues, [])

    assert corrected[0].lines == ["超过180mmHg"]
    assert report[0]["changes"][0]["matched_by"] == ["builtin:mmHg_single"]


def test_builtin_unit_normalizations_does_not_join_across_newlines() -> None:
    text = "编号882\n或一百毫米汞柱需要重新评估"

    normalized, changes = builtin_unit_normalizations(text)

    assert normalized == "编号882\n或100mmHg需要重新评估"
    assert [change["correct"] for change in changes] == ["100mmHg"]
