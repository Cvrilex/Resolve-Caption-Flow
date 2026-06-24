import json
import sys
from pathlib import Path


MVP_DIR = Path(__file__).resolve().parents[1] / "mvp_pipeline"
_INSERTED_MVP_DIR = str(MVP_DIR) not in sys.path
if _INSERTED_MVP_DIR:
    sys.path.insert(0, str(MVP_DIR))

import llm_term_reviewer  # noqa: E402

if _INSERTED_MVP_DIR:
    sys.path.remove(str(MVP_DIR))


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
