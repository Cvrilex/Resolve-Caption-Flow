from pathlib import Path

import pytest

from drautocut.integrations.online_asr import OnlineAsrError, OnlineAsrTranscriber


def test_online_asr_transcriber_loads_external_tool_and_returns_cues(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    (tool_dir / "online_asr.py").write_text(
        """
class Result:
    def has_data(self):
        return True
    def to_srt(self):
        return "1\\n00:00:00,000 --> 00:00:01,000\\n测试\\n\\n"

class BcutASR:
    def __init__(self, audio):
        self.audio = audio
    def run(self, callback=None):
        if callback:
            callback(100, "completed")
        return Result()

class JianYingASR(BcutASR):
    pass
""",
        encoding="utf-8",
    )
    events = []

    cues = OnlineAsrTranscriber("bcut", tool_dir=tool_dir).transcribe(
        tmp_path / "audio.m4a",
        progress=lambda percent, message: events.append((percent, message)),
    )

    assert cues[0].text == "测试"
    assert events == [(100, "completed")]


def test_jianying_transcriber_applies_custom_sign_service_url(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    (tool_dir / "online_asr.py").write_text(
        """
class Result:
    def has_data(self):
        return True
    def to_srt(self):
        return "1\\n00:00:00,000 --> 00:00:01,000\\n服务已替换\\n\\n"

class JianYingASR:
    SIGN_SERVICE_URL = "default"
    def __init__(self, audio):
        self.audio = audio
    def run(self, callback=None):
        assert self.SIGN_SERVICE_URL == "https://example.test/sign"
        return Result()
""",
        encoding="utf-8",
    )

    cues = OnlineAsrTranscriber(
        "jianying",
        tool_dir=tool_dir,
        jianying_sign_service_url="https://example.test/sign",
    ).transcribe(tmp_path / "audio.m4a")

    assert cues[0].text == "服务已替换"


def test_jianying_preflight_reports_unavailable_sign_service(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    (tool_dir / "online_asr.py").write_text(
        """
class JianYingASR:
    SIGN_SERVICE_URL = "default"
    def __init__(self, audio):
        self.audio = audio
    def _get_sign(self, url):
        raise RuntimeError("HTTP 500")
""",
        encoding="utf-8",
    )

    with pytest.raises(OnlineAsrError, match="Jianying sign service unavailable"):
        OnlineAsrTranscriber("jianying", tool_dir=tool_dir).preflight()
