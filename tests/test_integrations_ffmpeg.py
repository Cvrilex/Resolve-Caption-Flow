from drautocut.integrations.ffmpeg import parse_silencedetect_output


def test_parse_silencedetect_output_returns_ranges_in_ms() -> None:
    output = """
    [silencedetect @ 0x123] silence_start: 599.721
    [silencedetect @ 0x123] silence_end: 600.842 | silence_duration: 1.121
    [silencedetect @ 0x123] silence_start: 1203.5
    [silencedetect @ 0x123] silence_end: 1204.1 | silence_duration: 0.6
    """

    silences = parse_silencedetect_output(output)

    assert [(item.start_ms, item.end_ms) for item in silences] == [(599721, 600842), (1203500, 1204100)]
