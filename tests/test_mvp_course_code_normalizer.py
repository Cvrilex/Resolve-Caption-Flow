import json
from pathlib import Path

from pipeline.course_code_normalizer import (
    extract_course_code_from_filename,
    normalize_srt_course_code,
    normalize_text_with_course_code,
)


def test_extract_course_code_from_pdf_filename_keeps_level_suffix() -> None:
    pdf = Path("耐药菌重症感染诊治与抗菌药物合理应用。2026-03-08-018（国）李颍.pdf")

    rule = extract_course_code_from_filename(pdf)

    assert rule is not None
    assert rule.standard == "2026-03-08-018（国）"
    assert rule.month == 3
    assert rule.day == 8
    assert rule.serial == 18


def test_extract_course_code_keeps_custom_parenthetical_suffix() -> None:
    pdf = Path("消化疾病规范化诊疗与新进展。2026-03-12-001(沪远)GERD的优化管理_张伟.pdf")

    rule = extract_course_code_from_filename(pdf)

    assert rule is not None
    assert rule.standard == "2026-03-12-001（沪远）"
    assert rule.level == "沪远"


def test_normalize_text_matches_arabic_chinese_and_missing_guo() -> None:
    rule = extract_course_code_from_filename(Path("课程2026-03-08-018（国）.pdf"))
    assert rule is not None
    samples = [
        "本项目编号是2026 03 08 018",
        "项目编号20260308018已经开始",
        "课程编号2026年3月8日018",
        "课程编号2026-3-8-18国",
        "项目编号二零二六零三零八零一八",
        "项目编号二〇二六年三月八日十八",
    ]

    for sample in samples:
        normalized, count = normalize_text_with_course_code(sample, rule)
        assert count == 1
        assert "2026-03-08-018（国）" in normalized


def test_normalize_text_matches_custom_suffix_when_not_spoken() -> None:
    rule = extract_course_code_from_filename(Path("课程2026-03-12-001(沪远).pdf"))
    assert rule is not None

    normalized, count = normalize_text_with_course_code("课程编号2026年3月12日001", rule)

    assert count == 1
    assert "2026-03-12-001（沪远）" in normalized


def test_normalize_srt_course_code_writes_report(tmp_path: Path) -> None:
    pdf = tmp_path / "耐药菌重症感染诊治与抗菌药物合理应用。2026-03-08-018（国）李颍.pdf"
    pdf.write_text("fake", encoding="utf-8")
    srt = tmp_path / "input.srt"
    srt.write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:02,000",
                "今天学习2026年3月8日018项目",
                "",
                "2",
                "00:00:02,000 --> 00:00:04,000",
                "普通日期2026年3月8日不要替换",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output.srt"
    report = tmp_path / "report.json"

    result = normalize_srt_course_code(srt, output, report, context=pdf)

    text = output.read_text(encoding="utf-8")
    assert "2026-03-08-018（国）" in text
    assert "普通日期2026年3月8日不要替换" in text
    assert result["changed_cue_count"] == 1
    assert result["replacement_count"] == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["rule"]["standard"] == "2026-03-08-018（国）"
