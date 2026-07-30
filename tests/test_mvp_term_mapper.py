import json
import sys
from pathlib import Path

from pipeline import term_mapper
from pipeline.term_mapper import filter_context_only_replacements


def test_filter_context_only_replacements_keeps_pdf_backed_name_alias() -> None:
    payload = {
        "replacements": [
            {"wrong": "陈欣", "correct": "陈歆"},
            {"wrong": "先兆子痫", "correct": "先兆子鞍"},
            {"wrong": "高血压危象", "correct": "高血压急症"},
        ]
    }
    context = "上海交通大学医学院附属瑞金医院 陈歆 高血压急症 高血压危象 先兆子痫"

    filtered = filter_context_only_replacements(payload, context)

    assert filtered["replacements"] == [
        {"wrong": "陈欣", "correct": "陈歆"},
        {
            "wrong": "高血压危象",
            "correct": "高血压急症",
            "review_warning": "wrong_also_appears_in_pdf_context",
        },
    ]
    assert [item["drop_reason"] for item in filtered["dropped_replacements"]] == [
        "correct_not_in_pdf_context",
    ]


def test_read_context_includes_source_filename_for_course_names(tmp_path: Path) -> None:
    context = tmp_path / "黄绮芳-高血压课程.txt"
    context.write_text("上海市高血压研究所", encoding="utf-8")

    text = term_mapper.read_context(context)
    filtered = filter_context_only_replacements(
        {"replacements": [{"wrong": "黄启芳", "correct": "黄绮芳"}]},
        text,
    )

    assert "资料文件名：黄绮芳-高血压课程.txt" in text
    assert filtered["replacements"] == [{"wrong": "黄启芳", "correct": "黄绮芳"}]


def test_extract_pdf_text_falls_back_to_bundled_python(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(term_mapper, "BUNDLED_PYTHON", Path("/bin/sh"))

    class FakeCompleted:
        stdout = json.dumps({"text": "## PDF Page 1\n高血压诊断标准"}, ensure_ascii=False)

    def fake_run(command, **kwargs):
        assert command[:2] == ["/bin/sh", "-c"]
        assert command[-1] == str(pdf)
        assert kwargs["check"] is True
        return FakeCompleted()

    monkeypatch.setattr(term_mapper.subprocess, "run", fake_run)

    text = term_mapper.extract_pdf_text_with_python(pdf, ModuleNotFoundError("pypdf"))

    assert "高血压诊断标准" in text


def test_extract_pdf_text_uses_pymupdf_when_pypdf_fails(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    def fake_reader(_path):
        raise RuntimeError("pypdf cannot decode this PDF")

    monkeypatch.setattr(term_mapper, "PdfReader", fake_reader, raising=False)
    monkeypatch.setitem(sys.modules, "pypdf", type("FakePypdf", (), {"PdfReader": fake_reader}))
    monkeypatch.setattr(
        term_mapper,
        "extract_pdf_text_with_pymupdf",
        lambda _path: "## PDF Page 1\n耐药革兰阴性菌 抗菌药物合理应用 " * 4,
    )

    text = term_mapper.extract_pdf_text(pdf)

    assert "耐药革兰阴性菌" in text
    assert "抽取方式：pymupdf" in text


def test_extract_pdf_text_reports_image_only_pdf(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    def fake_reader(_path):
        raise RuntimeError("empty text")

    monkeypatch.setitem(sys.modules, "pypdf", type("FakePypdf", (), {"PdfReader": fake_reader}))
    monkeypatch.setattr(term_mapper, "extract_pdf_text_with_pymupdf", lambda _path: "")

    try:
        term_mapper.extract_pdf_text(pdf)
    except term_mapper.TermMapperError as exc:
        assert "扫描图片版" in str(exc)
    else:
        raise AssertionError("expected TermMapperError")


def test_diagnose_pdf_reports_best_extractor(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    def fake_reader(_path):
        raise RuntimeError("pypdf cannot decode this PDF")

    monkeypatch.setitem(sys.modules, "pypdf", type("FakePypdf", (), {"PdfReader": fake_reader}))
    monkeypatch.setattr(
        term_mapper,
        "extract_pdf_text_with_pymupdf",
        lambda _path: "## PDF Page 1\n耐药革兰阴性菌 抗菌药物合理应用 " * 20,
    )
    monkeypatch.setitem(
        sys.modules,
        "fitz",
        type(
            "FakeFitz",
            (),
            {
                "open": staticmethod(
                    lambda _path: type(
                        "FakeDoc",
                        (),
                        {
                            "__enter__": lambda self: self,
                            "__exit__": lambda self, *_args: None,
                            "__len__": lambda self: 1,
                        },
                    )()
                )
            },
        ),
    )

    report = term_mapper.diagnose_pdf(pdf)

    assert report["status"] == "ok"
    assert report["best_method"] == "pymupdf"
    assert report["best_char_count"] > 80
    assert "耐药革兰阴性菌" in report["preview"]


def test_abbreviation_translation_keeps_english_abbreviation() -> None:
    payload = term_mapper.normalize_terms_payload(
        json.dumps(
            {
                "replacements": [
                    {
                        "wrong": "ACEI",
                        "correct": "血管紧张素转换酶抑制剂",
                        "confidence": "high",
                        "evidence": "PDF",
                        "note": "英文缩写",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )

    assert payload["replacements"][0]["correct"] == "ACEI（血管紧张素转换酶抑制剂）"
    assert "英文缩写保留" in payload["replacements"][0]["note"]


def test_context_filter_normalizes_abbreviation_after_membership_check() -> None:
    payload = {
        "replacements": [
            {
                "wrong": "A C E I",
                "correct": "血管紧张素转换酶抑制剂",
                "confidence": "high",
                "evidence": "PDF",
            }
        ]
    }
    context = "本课程介绍血管紧张素转换酶抑制剂的临床应用"

    filtered = filter_context_only_replacements(payload, context)

    assert filtered["replacements"][0]["correct"] == "ACEI（血管紧张素转换酶抑制剂）"


def test_context_filter_accepts_abbreviation_with_parenthetical_translation() -> None:
    payload = {
        "replacements": [
            {
                "wrong": "gird",
                "correct": "GERD（胃食管反流病）",
                "confidence": "high",
                "evidence": "PDF",
            }
        ]
    }
    context = "GERD的优化管理。胃食管反流病诊疗指南。"

    filtered = filter_context_only_replacements(payload, context)

    assert filtered["replacements"][0]["correct"] == "GERD（胃食管反流病）"
    assert filtered["dropped_replacements"] == []


def test_normalize_terms_payload_drops_cjk_layout_space_only_change() -> None:
    payload = term_mapper.normalize_terms_payload(
        json.dumps(
            {
                "replacements": [
                    {"wrong": "郭燕", "correct": "郭 燕"},
                    {"wrong": "郭艳", "correct": "郭 燕"},
                ]
            },
            ensure_ascii=False,
        )
    )

    assert payload["replacements"] == [
        {
            "wrong": "郭艳",
            "correct": "郭燕",
            "aliases": [],
            "patterns": [],
            "confidence": "medium",
            "evidence": "",
            "note": "",
        }
    ]


def test_remote_context_chunk_default_matches_production_granularity() -> None:
    chunk_chars, max_chunks = term_mapper.context_chunk_limit("https://api.deepseek.com")

    assert chunk_chars == 1800
    assert max_chunks == 16


def test_context_filter_can_keep_unverified_terms_for_manual_review() -> None:
    payload = {
        "replacements": [
            {
                "wrong": "郭艳",
                "correct": "郭燕",
                "confidence": "high",
                "evidence": "PDF title",
                "note": "人名",
            }
        ]
    }
    context = "PDF 抽取文本乱码导致姓名没有逐字出现"

    filtered = filter_context_only_replacements(payload, context, keep_unverified=True)

    assert filtered["replacements"][0]["wrong"] == "郭艳"
    assert filtered["replacements"][0]["correct"] == "郭燕"
    assert filtered["replacements"][0]["confidence"] == "low"
    assert filtered["replacements"][0]["review_warning"] == "correct_not_verified_in_pdf_text"
    assert filtered["dropped_replacements"][0]["kept_for_manual_review"] is True


def test_context_only_generation_chunks_pdf_context_and_merges_terms(tmp_path: Path, monkeypatch) -> None:
    context = tmp_path / "消化课程_张伟.txt"
    context.write_text(
        "\n\n".join(
            [
                "## PDF Page 1\nGERD 胃食管反流病 张伟",
                "## PDF Page 2\nPPI 质子泵抑制剂",
                "## PDF Page 3\nP-CAB 钾离子竞争性酸阻滞剂",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "terms.json"
    calls: list[str] = []
    events: list[dict[str, object]] = []

    def fake_chat(messages, **_kwargs):
        prompt = messages[-1]["content"]
        calls.append(prompt)
        terms = [
            ("gird", "GERD（胃食管反流病）"),
            ("p p i", "PPI（质子泵抑制剂）"),
            ("P cab", "P-CAB（钾离子竞争性酸阻滞剂）"),
        ]
        wrong, correct = terms[(len(calls) - 1) % len(terms)]
        return json.dumps(
            {
                "replacements": [
                    {
                        "wrong": wrong,
                        "correct": correct,
                        "confidence": "high",
                        "evidence": "PDF",
                    }
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(term_mapper, "chat_completion", fake_chat)
    monkeypatch.setenv("TERM_MAPPER_CONTEXT_CHUNK_CHARS", "80")
    monkeypatch.setenv("TERM_MAPPER_CONTEXT_MAX_CHUNKS", "10")

    payload = term_mapper.generate_terms_from_context(
        context=context,
        output=output,
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key="test",
        progress_callback=lambda event: events.append(event),
        keep_unverified=False,
    )

    assert len(calls) >= 2
    assert payload["chunk_count"] == len(calls)
    assert {row["correct"] for row in payload["replacements"]} >= {
        "GERD（胃食管反流病）",
        "PPI（质子泵抑制剂）",
    }
    planned = next(event for event in events if event["status"] == "planned")
    assert planned["chunk_count"] == payload["chunk_count"]
    progress_totals = {event["total"] for event in events if event["status"] == "progress"}
    assert progress_totals == {payload["chunk_count"]}


def test_deepseek_v4_requests_disable_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "{\"replacements\": []}"}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(term_mapper.urllib.request, "urlopen", fake_urlopen)

    result = term_mapper.chat_completion(
        [{"role": "user", "content": "输出 JSON"}],
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_key="test",
    )

    assert result == "{\"replacements\": []}"
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["max_tokens"] >= 4096
