from caption_core.domain.srt import Cue
from caption_core.domain.subtitles import (
    clean_and_split_cues,
    clean_punctuation,
    interpolate_timestamps,
    split_preserving_english_spaces,
    visible_len,
)


def test_clean_punctuation_removes_configured_commas_and_compacts_chinese_spaces() -> None:
    assert clean_punctuation("那么， 甲状腺 眼病, TED", "，,") == "那么甲状腺眼病 TED"


def test_split_preserving_english_spaces_keeps_english_words_readable() -> None:
    segments = split_preserving_english_spaces("Georgetown university Cancer Center", max_chars=20)

    assert segments == ["Georgetown university", "Cancer Center"]
    assert all(visible_len(segment) <= 20 for segment in segments)


def test_interpolate_timestamps_distributes_original_range() -> None:
    timings = interpolate_timestamps("00:00:00,000", "00:00:03,000", ["短", "比较长"])

    assert timings[0][0] == "00:00:00,000"
    assert timings[-1][1] == "00:00:03,000"
    assert timings[0][1] == timings[1][0]


def test_clean_and_split_cues_creates_multiple_independent_cues() -> None:
    cues = [
        Cue(
            index="1",
            timing="00:00:00,000 --> 00:00:04,000",
            lines=["上海交通大学甲状腺疾病诊治中心九院分中心组长"],
        )
    ]

    expanded, report = clean_and_split_cues(cues, max_chars=10, punctuation="，,")

    assert len(expanded) > 1
    assert all("\n" not in cue.text for cue in expanded)
    assert expanded[0].index == "1"
    assert expanded[-1].timing.endswith("00:00:04,000")
    assert report["cue_count_before"] == 1
    assert report["cue_count_after"] == len(expanded)
    assert report["overlong_changed_cue_count"] == 1

