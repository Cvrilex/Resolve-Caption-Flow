from caption_core.domain.srt import Cue, parse_srt_text, render_srt


def test_parse_and_render_srt_text_normalizes_indices() -> None:
    text = "7\r\n00:00:00,000 --> 00:00:01,000\r\n第一行\r\n第二行\r\n\r\n"

    cues = parse_srt_text(text)

    assert cues == [Cue(index="7", timing="00:00:00,000 --> 00:00:01,000", lines=["第一行", "第二行"])]
    assert render_srt(cues) == "1\n00:00:00,000 --> 00:00:01,000\n第一行\n第二行\n"

