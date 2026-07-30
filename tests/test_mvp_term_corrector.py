import json
from pathlib import Path

from pipeline.term_corrector import builtin_medical_term_normalizations, builtin_unit_normalizations, correct_srt


def test_mvp_correct_srt_normalizes_mmhg_and_reports_unmatched(tmp_path: Path) -> None:
    srt = tmp_path / "input.srt"
    srt.write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:02,000",
                "收缩压达到一百八十毫米汞柱",
                "",
                "2",
                "00:00:02,000 --> 00:00:04,000",
                "诊断标准是180或120毫米汞柱",
                "",
            ]
        ),
        encoding="utf-8",
    )
    terms = tmp_path / "terms.json"
    terms.write_text(
        json.dumps(
            {
                "replacements": [
                    {"wrong": "瑞金", "correct": "瑞金医院", "note": "故意未命中"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output.srt"
    report = tmp_path / "report.json"

    result = correct_srt(srt, terms, output, report)

    assert "收缩压达到180mmHg" in output.read_text(encoding="utf-8")
    assert "诊断标准是180/120mmHg" in output.read_text(encoding="utf-8")
    assert result["changed_cue_count"] == 2
    assert result["unmatched_replacement_count"] == 1
    assert result["unmatched_replacements"][0]["wrong"] == "瑞金"
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert report_payload["changes"][0]["changes"][0]["matched_by"] == ["builtin:mmHg_single"]


def test_mvp_builtin_unit_normalization_does_not_join_across_newlines() -> None:
    normalized, changes = builtin_unit_normalizations("编号882\n或一百毫米汞柱需要重新评估")

    assert normalized == "编号882\n或100mmHg需要重新评估"
    assert [change["correct"] for change in changes] == ["100mmHg"]


def test_mvp_builtin_unit_normalization_handles_over_and_ge_mmhg_variants() -> None:
    normalized, changes = builtin_unit_normalizations(
        "A是小于140 over 90个毫米汞柱 B是小于130/80毫米汞柱 C是小于120/70 mmHg"
    )

    assert normalized == "A是小于140/90mmHg B是小于130/80mmHg C是小于120/70mmHg"
    assert [change["correct"] for change in changes] == ["140/90mmHg", "130/80mmHg", "120/70mmHg"]


def test_mvp_builtin_unit_normalization_handles_bp_range_unit() -> None:
    normalized, changes = builtin_unit_normalizations("舒张压85~89的毫米汞柱")

    assert normalized == "舒张压85~89mmHg"
    assert changes[0]["matched_by"] == ["builtin:mmHg_range"]


def test_mvp_builtin_medical_term_normalization_handles_drug_name_variants() -> None:
    normalized, changes = builtin_medical_term_normalizations(
        "首选的药物是阿贝维尔消普纳 也可以选艾斯维尔尼卡的平或无法比尔"
    )

    assert normalized == "首选的药物是拉贝洛尔硝普钠 也可以选艾司洛尔尼卡地平或乌拉地尔"
    assert sorted(change["correct"] for change in changes) == sorted([
        "尼卡地平",
        "拉贝洛尔",
        "艾司洛尔",
        "乌拉地尔",
        "硝普钠",
    ])


def test_mvp_correct_srt_runs_builtin_medical_term_normalization_without_terms(tmp_path: Path) -> None:
    srt = tmp_path / "input.srt"
    srt.write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:02,000",
                "也可以选艾斯维尔尼卡地平或无法比尔",
                "",
            ]
        ),
        encoding="utf-8",
    )
    terms = tmp_path / "terms.json"
    terms.write_text(json.dumps({"replacements": []}), encoding="utf-8")
    output = tmp_path / "output.srt"
    report = tmp_path / "report.json"

    result = correct_srt(srt, terms, output, report)

    assert "也可以选艾司洛尔尼卡地平或乌拉地尔" in output.read_text(encoding="utf-8")
    assert result["changed_cue_count"] == 1
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert report_payload["changes"][0]["changes"][0]["matched_by"] == ["builtin:medical_term"]
