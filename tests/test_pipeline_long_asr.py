from pathlib import Path

from drautocut.domain.srt import Cue
from drautocut.integrations.ffmpeg import MediaInfo
from drautocut.pipeline.long_asr import run_online_long_video_asr
from drautocut.pipeline.segments import SilenceRange


def test_run_online_long_video_asr_orchestrates_prepare_segment_asr_and_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events = []
    video = tmp_path / "video.mp4"
    output = tmp_path / "out.srt"
    video.write_text("fake", encoding="utf-8")

    monkeypatch.setattr("drautocut.pipeline.long_asr.probe_media", lambda path: MediaInfo(duration_ms=20 * 60 * 1000))
    monkeypatch.setattr(
        "drautocut.pipeline.long_asr.detect_silences",
        lambda path: [SilenceRange(599_000, 601_000)],
    )
    monkeypatch.setattr(
        "drautocut.pipeline.long_asr.extract_audio_segments",
        lambda source, output_dir, segments: [output_dir / f"part{segment.index}.m4a" for segment in segments],
    )

    class FakeTranscriber:
        name = "fake"

        def __init__(self, engine, *, tool_dir):
            self.engine = engine
            self.tool_dir = tool_dir

        def transcribe(self, audio_path, *, progress=None):
            if progress:
                progress(100, "done")
            return [Cue(index="1", timing="00:00:00,000 --> 00:00:01,000", lines=[audio_path.stem])]

    monkeypatch.setattr("drautocut.pipeline.long_asr.OnlineAsrTranscriber", FakeTranscriber)

    result = run_online_long_video_asr(
        video_path=video,
        output_srt=output,
        work_dir=tmp_path / "work",
        tool_dir=tmp_path / "tool",
        engine="bcut",
        progress=events.append,
    )

    assert result.srt_path == output
    assert output.exists()
    assert "00:00:00,000 --> 00:00:01,000" in output.read_text(encoding="utf-8")
    assert "00:10:00,000 --> 00:10:01,000" in output.read_text(encoding="utf-8")
    assert any(event["status"] == "segment_progress" for event in events)
    assert events[-1]["message"] == "merged SRT written"
