from pathlib import Path

from pipeline.caption_pipeline import resolve_output_name_for, resolve_visible_name_for


def test_resolve_names_follow_uploaded_video_stem() -> None:
    video = Path("/tmp/01-张三.mp4")

    assert resolve_visible_name_for(video) == "01-张三"
    assert resolve_output_name_for(video) == "01-张三字幕版"


def test_resolve_names_preserve_spaces_and_clean_illegal_chars() -> None:
    video = Path('/tmp/01 张三:课程?.mov')

    assert resolve_visible_name_for(video) == "01 张三 课程"
    assert resolve_output_name_for(video) == "01 张三 课程字幕版"
