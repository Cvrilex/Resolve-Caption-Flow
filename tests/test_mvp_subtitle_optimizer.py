import json
import sys
from pathlib import Path


MVP_DIR = Path(__file__).resolve().parents[1] / "mvp_pipeline"
_INSERTED_MVP_DIR = str(MVP_DIR) not in sys.path
if _INSERTED_MVP_DIR:
    sys.path.insert(0, str(MVP_DIR))

import subtitle_optimizer  # noqa: E402

if _INSERTED_MVP_DIR:
    sys.path.remove(str(MVP_DIR))


def _write_srt(path: Path, texts: list[str]) -> None:
    blocks = []
    for idx, text in enumerate(texts, start=1):
        start = (idx - 1) * 1000
        end = idx * 1000
        blocks.append(
            "\n".join(
                [
                    str(idx),
                    f"00:00:{start // 1000:02},000 --> 00:00:{end // 1000:02},000",
                    text,
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def test_parse_filler_words_accepts_webui_formats() -> None:
    assert subtitle_optimizer.parse_filler_words("嗯,呃，啊、那个 这个\n然后") == (
        "嗯",
        "呃",
        "啊",
        "那个",
        "这个",
        "然后",
    )
    assert subtitle_optimizer.parse_filler_words("") == ()


def test_optimizer_removes_filler_words_during_text_cleanup(tmp_path: Path) -> None:
    source = tmp_path / "fillers.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["嗯因为硝普钠是首选药物", "呃我们要做一个小结", "啊这里需要注意"])

    subtitle_optimizer.optimize_srt(
        srt=source,
        output=output,
        report_path=report,
        max_chars=20,
        min_chars=5,
        punctuation="，,",
        model="test-model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        use_llm=False,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "嗯" not in rendered
    assert "呃" not in rendered
    assert "啊" not in rendered
    assert "因为硝普钠是首选药物" in rendered
    assert "我们要做一个小结" in rendered
    assert "这里需要注意" in rendered


def test_optimizer_uses_custom_filler_words(tmp_path: Path) -> None:
    source = tmp_path / "custom_fillers.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["这个我们来看一下", "啊这里保留"])

    result = subtitle_optimizer.optimize_srt(
        srt=source,
        output=output,
        report_path=report,
        max_chars=20,
        min_chars=5,
        punctuation="，,",
        fillers=subtitle_optimizer.parse_filler_words("这个"),
        model="test-model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        use_llm=False,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "这个" not in rendered
    assert "我们来看一下" in rendered
    assert "啊这里保留" in rendered
    assert result["removed_fillers"] == ["这个"]


def test_optimizer_normalizes_units_after_llm_review(tmp_path: Path) -> None:
    source = tmp_path / "units.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["A呢是小于140 over 90个毫米汞柱", "B呢是小于130/80毫米汞柱", "小于120/70 mmHg"])

    subtitle_optimizer.optimize_srt(
        srt=source,
        output=output,
        report_path=report,
        max_chars=30,
        min_chars=5,
        punctuation="，,",
        model="test-model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        use_llm=False,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "140/90mmHg" in rendered
    assert "130/80mmHg" in rendered
    assert "120/70mmHg" in rendered
    assert "毫米汞柱" not in rendered
    assert " mmHg" not in rendered


def test_short_window_llm_reflow_preserves_text_and_retimes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "short.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["这是", "一个", "完整的句子"])

    def fake_chat_completion(*args, **kwargs):
        return json.dumps({"segments": ["这是一个完整的句子"]}, ensure_ascii=False)

    monkeypatch.setattr(subtitle_optimizer, "chat_completion", fake_chat_completion)

    result = subtitle_optimizer.optimize_srt(
        srt=source,
        output=output,
        report_path=report,
        max_chars=20,
        min_chars=5,
        punctuation="，,",
        model="test-model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        use_llm=True,
    )

    assert result["short_detected_count"] == 2
    assert result["short_changed_window_count"] == 1
    assert "这是一个完整的句子" in output.read_text(encoding="utf-8")


def test_optimizer_can_defer_short_windows_to_full_review(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "short_deferred.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["这是", "一个", "完整的句子"])

    def fake_chat_completion(*args, **kwargs):
        raise AssertionError("short subtitle LLM should be deferred")

    monkeypatch.setattr(subtitle_optimizer, "chat_completion", fake_chat_completion)

    result = subtitle_optimizer.optimize_srt(
        srt=source,
        output=output,
        report_path=report,
        max_chars=20,
        min_chars=5,
        punctuation="，,",
        model="test-model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        use_llm=True,
        optimize_short=False,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "这是" in rendered
    assert "一个" in rendered
    assert result["short_detected_count"] == 2
    assert result["short_window_count"] == 0
    assert result["short_changed_window_count"] == 0
    assert result["optimize_short"] is False


def test_short_window_llm_content_mismatch_falls_back(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "short.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["这是", "一个", "完整的句子"])

    def fake_chat_completion(*args, **kwargs):
        return json.dumps({"segments": ["这是一个被改写的句子"]}, ensure_ascii=False)

    monkeypatch.setattr(subtitle_optimizer, "chat_completion", fake_chat_completion)

    result = subtitle_optimizer.optimize_srt(
        srt=source,
        output=output,
        report_path=report,
        max_chars=20,
        min_chars=5,
        punctuation="，,",
        model="test-model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        use_llm=True,
    )

    rendered = output.read_text(encoding="utf-8")
    assert result["short_changed_window_count"] == 1
    assert result["llm_fallback_error_count"] == 1
    assert "这是一个被改写的句子" not in rendered
    assert "这是一个完整的句子" in rendered


def test_merge_short_segments_reduces_residual_fragments() -> None:
    merged = subtitle_optimizer.merge_short_segments(
        ["那硝苯地平呢", "口服呢", "它吸收是不稳定的"],
        min_chars=5,
        max_chars=20,
    )

    assert merged == ["那硝苯地平呢口服呢", "它吸收是不稳定的"]


def test_repair_protected_term_boundaries_keeps_medical_terms_whole() -> None:
    repaired = subtitle_optimizer.repair_protected_term_boundaries(
        ["嗯因为硝普钠是高血压急", "症首选的一个静脉降压药"]
    )

    assert repaired == ["嗯因为硝普钠是高血压急症", "首选的一个静脉降压药"]


def test_build_issue_windows_does_not_merge_across_large_time_gaps() -> None:
    cues = [
        subtitle_optimizer.Cue(index="1", timing="00:00:00,000 --> 00:00:01,000", lines=["前文"]),
        subtitle_optimizer.Cue(index="2", timing="00:00:01,000 --> 00:00:02,000", lines=["短"]),
        subtitle_optimizer.Cue(index="3", timing="00:00:02,000 --> 00:00:03,000", lines=["后文"]),
        subtitle_optimizer.Cue(index="4", timing="00:10:00,000 --> 00:10:01,000", lines=["前文"]),
        subtitle_optimizer.Cue(index="5", timing="00:10:01,000 --> 00:10:02,000", lines=["短"]),
        subtitle_optimizer.Cue(index="6", timing="00:10:02,000 --> 00:10:03,000", lines=["后文"]),
    ]

    windows = subtitle_optimizer.build_issue_windows([1, 4], len(cues), radius=1, cues=cues)

    assert windows == [(0, 2), (3, 5)]


def test_build_issue_windows_stops_at_meaningful_pause() -> None:
    cues = [
        subtitle_optimizer.Cue(index="1076", timing="00:52:34,807 --> 00:52:38,867", lines=["第四道题是高血压急症快速降压"]),
        subtitle_optimizer.Cue(index="1077", timing="00:52:38,867 --> 00:52:41,487", lines=["首选的静脉药物是"]),
        subtitle_optimizer.Cue(index="1078", timing="00:52:42,047 --> 00:52:43,847", lines=["答案是 B"]),
        subtitle_optimizer.Cue(index="1079", timing="00:52:44,787 --> 00:52:48,487", lines=["嗯因为硝普钠是高血压急症首选的一个静脉降压药"]),
        subtitle_optimizer.Cue(index="1080", timing="00:52:48,487 --> 00:52:49,707", lines=["它起效比较快"]),
    ]

    windows = subtitle_optimizer.build_issue_windows([2], len(cues), radius=2, cues=cues)

    assert windows == [(0, 2)]


def test_build_asr_boundary_windows_selects_three_cues_around_cut() -> None:
    cues = [
        subtitle_optimizer.Cue(index="1", timing="00:09:58,000 --> 00:09:59,000", lines=["上一句"]),
        subtitle_optimizer.Cue(index="2", timing="00:09:59,000 --> 00:10:00,000", lines=["切点前半句"]),
        subtitle_optimizer.Cue(index="3", timing="00:10:00,000 --> 00:10:01,000", lines=["切点后半句"]),
        subtitle_optimizer.Cue(index="4", timing="00:10:01,000 --> 00:10:02,000", lines=["再下一句"]),
    ]

    windows = subtitle_optimizer.build_asr_boundary_windows(cues, [10 * 60 * 1000])

    assert windows == [(0, 2, 10 * 60 * 1000)]


def test_asr_boundary_llm_reflow_preserves_window_text(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "boundary.srt"
    output = tmp_path / "out.srt"
    report = tmp_path / "report.json"
    _write_srt(source, ["我们来看", "高血压急症的", "处理流程"])

    def fake_chat_completion(*args, **kwargs):
        return json.dumps({"segments": ["我们来看高血压急症的处理流程"]}, ensure_ascii=False)

    monkeypatch.setattr(subtitle_optimizer, "chat_completion", fake_chat_completion)

    result = subtitle_optimizer.repair_asr_boundaries(
        srt=source,
        output=output,
        report_path=report,
        boundary_ms=[1000],
        max_chars=20,
        min_chars=5,
        punctuation="，,",
        model="test-model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        use_llm=True,
    )

    assert result["boundary_window_count"] == 1
    assert result["boundary_changed_window_count"] == 1
    assert "我们来看高血压急症的处理流程" in output.read_text(encoding="utf-8")


def test_deepseek_v4_subtitle_requests_disable_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "{\"segments\": [\"一句话\"]}"}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(subtitle_optimizer.urllib.request, "urlopen", fake_urlopen)

    result = subtitle_optimizer.chat_completion(
        [{"role": "user", "content": "输出 JSON"}],
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_key="test",
    )

    assert result == "{\"segments\": [\"一句话\"]}"
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["max_tokens"] >= 2048
