import json
from pathlib import Path

from pipeline import llm_term_reviewer


def _write_srt(path: Path, texts: list[str]) -> None:
    blocks = []
    for idx, text in enumerate(texts, start=1):
        blocks.append(
            "\n".join(
                [
                    str(idx),
                    f"00:00:{idx - 1:02},000 --> 00:00:{idx:02},000",
                    text,
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def test_review_srt_terms_applies_exact_high_confidence_patch(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["患者是一百八十毫米汞柱", "陈欣医生讲课"])

    def fake_chat_completion(*_args, **_kwargs):
        return json.dumps(
            {
                "patches": [
                    {"cue": 1, "old": "一百八十毫米汞柱", "new": "180mmHg", "confidence": "high", "reason": "血压单位"},
                    {"cue": 2, "old": "陈欣", "new": "陈歆", "confidence": "high", "reason": "人名"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        batch_size=100,
        overlap=1,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "180mmHg" in rendered
    assert "陈歆医生讲课" in rendered
    assert result["applied_patch_count"] == 2
    assert result["rejected_patch_count"] == 0


def test_review_srt_terms_rejects_low_confidence_and_missing_old(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["患者血压升高"])

    def fake_chat_completion(*_args, **_kwargs):
        return json.dumps(
            {
                "patches": [
                    {"cue": 1, "old": "血压", "new": "动脉血压", "confidence": "low", "reason": "不确定"},
                    {"cue": 1, "old": "不存在", "new": "存在", "confidence": "high", "reason": "测试"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
    )

    assert "患者血压升高" in output.read_text(encoding="utf-8")
    assert result["applied_patch_count"] == 0
    assert [item["reason_rejected"] for item in result["rejected_patches"]] == [
        "low_confidence",
        "old_text_not_found",
    ]


def test_review_srt_terms_applies_semantic_reflow(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["这是", "一个", "完整句子"])

    def fake_chat_completion(*_args, **_kwargs):
        return json.dumps(
            {
                "patches": [],
                "reflows": [
                    {
                        "cue_ids": [1, 2, 3],
                        "segments": ["这是一个完整句子"],
                        "confidence": "high",
                        "reason": "短字幕语义破碎",
                    }
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        batch_size=100,
        overlap=1,
        min_chars=5,
        max_chars=20,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "这是一个完整句子" in rendered
    assert result["applied_reflow_count"] == 1
    assert result["rejected_reflow_count"] == 0
    assert result["residual_short_count"] == 0


def test_review_srt_terms_splits_residual_overlong_after_patches(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["这是一个需要在最后兜底切分的超长字幕"])

    calls = 0

    def fake_chat_completion(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return json.dumps({"patches": [], "reflows": []}, ensure_ascii=False)
        return json.dumps({"segments": ["这是一个需要", "在最后兜底切分", "的超长字幕"]}, ensure_ascii=False)

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        max_chars=10,
    )

    rendered_blocks = [
        block.splitlines()[2]
        for block in output.read_text(encoding="utf-8").strip().split("\n\n")
    ]
    assert len(rendered_blocks) > 1
    assert all(llm_term_reviewer.visible_len(text) <= 10 for text in rendered_blocks)
    assert result["residual_overlong_count"] == 0
    assert len(result["residual_overlong_splits"]) == 1
    assert result["residual_overlong_splits"][0]["method"] == "llm"


def test_review_srt_terms_runs_final_cleanup_after_llm_review(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["另外，诊室血压。", "嗯我们继续看"])

    def fake_chat_completion(*_args, **_kwargs):
        return json.dumps({"patches": [], "reflows": []}, ensure_ascii=False)

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        punctuation="。",
        fillers=("嗯",),
        comma_as_space=True,
    )

    rendered = output.read_text(encoding="utf-8")
    subtitle_text = "\n".join(
        line
        for line in rendered.splitlines()
        if line and "-->" not in line and not line.isdigit()
    )
    assert "，" not in subtitle_text
    assert "," not in subtitle_text
    assert "。" not in subtitle_text
    assert "另外 诊室血压" in subtitle_text
    assert "我们继续看" in subtitle_text
    assert result["final_cleanup_changed_cue_count"] == 2


def test_review_srt_terms_runs_numeric_unit_review_after_batch_review(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["耐药率都非常高", "在20以内的只有三种抗菌药物", "也就是三剑客"])
    prompts: list[str] = []

    def fake_chat_completion(messages, *_args, **_kwargs):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            return json.dumps({"patches": [], "reflows": []}, ensure_ascii=False)
        return json.dumps(
            {
                "patches": [
                    {
                        "cue": 2,
                        "old": "20以内",
                        "new": "20%以内",
                        "confidence": "high",
                        "reason": "上下文讨论耐药率，应为百分比",
                    }
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        min_chars=5,
        max_chars=20,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "在20%以内的只有三种抗菌药物" in rendered
    assert len(prompts) == 2
    assert "数值单位" in prompts[1] or "漏写数值单位" in prompts[1]
    assert result["numeric_unit_patch_count"] == 1
    assert result["numeric_unit_rejected_count"] == 0


def test_review_srt_terms_merges_residual_short_cues_after_review(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["答案是", "D", "很高危"])

    def fake_chat_completion(*_args, **_kwargs):
        return json.dumps({"patches": [], "reflows": []}, ensure_ascii=False)

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        min_chars=5,
        max_chars=20,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "答案是D很高危" in rendered
    assert result["residual_short_merge_count"] == 2
    assert result["residual_short_count"] == 0


def test_review_srt_terms_repairs_symbol_ending_with_five_cue_window(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["这是前二句", "这是前一句", "收缩压130~", "139mmHg的情况", "这是后一句"])
    calls: list[str] = []

    def fake_chat_completion(messages, *_args, **_kwargs):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:
            return json.dumps({"patches": [], "reflows": []}, ensure_ascii=False)
        return json.dumps(
            {"segments": ["这是前二句", "这是前一句", "收缩压130~139mmHg的情况", "这是后一句"]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        min_chars=5,
        max_chars=20,
    )

    rendered = output.read_text(encoding="utf-8")
    assert len(calls) == 2
    assert "这是前二句" in calls[1]
    assert "这是后一句" in calls[1]
    assert "收缩压130~139mmHg的情况" in rendered
    assert "收缩压130~\n" not in rendered
    assert result["symbol_ending_repair_count"] == 1


def test_review_srt_terms_residual_overlong_uses_neighbor_context(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["前一句", "这是一个需要按照语义重新切分的超长字幕", "后一句"])
    prompts: list[str] = []

    def fake_chat_completion(messages, *_args, **_kwargs):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            return json.dumps({"patches": [], "reflows": []}, ensure_ascii=False)
        return json.dumps({"segments": ["这是一个需要", "按照语义重新切分", "的超长字幕"]}, ensure_ascii=False)

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        max_chars=10,
    )

    assert len(prompts) == 2
    assert "前一句" in prompts[1]
    assert "后一句" in prompts[1]
    assert result["residual_overlong_splits"][0]["method"] == "llm"
    assert result["residual_overlong_count"] == 0


def test_review_srt_terms_rejects_reflow_content_changes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["这是", "一个", "完整句子"])

    def fake_chat_completion(*_args, **_kwargs):
        return json.dumps(
            {
                "patches": [],
                "reflows": [
                    {
                        "cue_ids": [1, 2, 3],
                        "segments": ["这是一个被改写的完整句子"],
                        "confidence": "high",
                        "reason": "测试",
                    }
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm_term_reviewer, "chat_completion", fake_chat_completion)

    result = llm_term_reviewer.review_srt_terms(
        srt=source,
        output=output,
        report_path=report,
        terms=None,
        model="test",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
    )

    rendered = output.read_text(encoding="utf-8")
    assert "被改写" not in rendered
    assert result["applied_reflow_count"] == 0
    assert result["rejected_reflow_count"] == 1
    assert result["rejected_reflows"][0]["reason_rejected"] == "content_not_preserved"


def test_deepseek_v4_term_review_requests_disable_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"patches\": []}"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(llm_term_reviewer.urllib.request, "urlopen", fake_urlopen)

    result = llm_term_reviewer.chat_completion(
        [{"role": "user", "content": "输出 JSON"}],
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_key="test",
    )

    assert result == "{\"patches\": []}"
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
