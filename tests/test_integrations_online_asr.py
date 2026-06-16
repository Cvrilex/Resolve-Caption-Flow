from pathlib import Path

from drautocut.integrations.online_asr import OnlineAsrTranscriber


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
