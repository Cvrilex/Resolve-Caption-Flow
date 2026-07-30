import asyncio
import json
import os
import sys
import zipfile
from pathlib import Path

from app import web_server


def test_confirm_terms_preserves_segment_settings(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "course.mp4"
    video.write_bytes(b"fake")
    pdf = tmp_path / "课程2026-03-08-018（国）.pdf"
    pdf.write_text("fake", encoding="utf-8")
    terms = tmp_path / "terms.preflight.json"
    terms.write_text(
        json.dumps(
            {
                "replacements": [
                    {"wrong": "错词", "correct": "正词"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "course-20260101-000000.jsonl"

    captured: dict[str, object] = {}

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(web_server.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        web_server,
        "_current_job",
        {
            "run_id": "run-1",
            "status": "awaiting_terms",
            "terms_path": str(terms),
            "srt_path": None,
            "log_path": str(log_path),
            "pending": {
                "video_path": str(video),
                "pdf_path": str(pdf),
                "engine": "local",
                "segmented_asr": True,
                "asr_max_workers": 2,
                "asr_segment_minutes": 8.0,
                "asr_max_segment_minutes": 15.5,
                "srt_only": True,
                "render_preset": "my-resolve-preset",
                "remove_fillers": "嗯\n呃\n啊\n呢",
                "remove_punctuation": "，,",
                "comma_as_space": True,
                "subtitle_min_chars": 5,
                "subtitle_max_chars": 20,
            },
        },
    )

    result = asyncio.run(
        web_server.confirm_terms(
            "run-1",
            {
                "replacements": [
                    {"enabled": True, "wrong": "错词", "correct": "正词"},
                ]
            },
        )
    )

    args = captured["args"]
    assert result["status"] == "running"
    assert captured["started"] is True
    assert args[8] == pdf
    assert args[13] is True
    assert args[14] == 2
    assert args[15] == 8.0
    assert args[16] == 15.5
    assert args[17] is True
    assert args[18] == "my-resolve-preset"
    assert args[19] == "嗯\n呃\n啊\n呢"
    assert args[20] == "，,"
    assert args[21] == 5
    assert args[22] == 20
    assert args[23] is True
    assert Path(result["approved_terms_path"]).exists()


def test_terms_for_review_returns_dropped_count(tmp_path: Path, monkeypatch) -> None:
    terms = tmp_path / "terms.preflight.json"
    terms.write_text(
        json.dumps(
            {
                "replacements": [
                    {"wrong": "陈欣", "correct": "陈歆"},
                ],
                "dropped_replacements": [
                    {"wrong": "高血压危象", "correct": "高血压急症", "drop_reason": "wrong_already_appears_in_pdf_context"},
                    {"wrong": "先兆子痫", "correct": "先兆子鞍", "drop_reason": "correct_not_in_pdf_context"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        web_server,
        "_current_job",
        {
            "run_id": "run-terms",
            "status": "awaiting_terms",
            "terms_path": str(terms),
            "srt_path": None,
        },
    )

    result = asyncio.run(web_server.terms_for_review("run-terms"))

    assert result["dropped_replacement_count"] == 2
    assert result["replacements"][0]["wrong"] == "陈欣"


def test_course_code_candidate_is_first_in_terms_review(tmp_path: Path) -> None:
    pdf = tmp_path / "耐药菌重症感染诊治与抗菌药物合理应用。2026-03-08-018（国）李颍.pdf"
    pdf.write_text("fake", encoding="utf-8")
    terms = tmp_path / "terms.preflight.json"
    terms.write_text(
        json.dumps(
            {
                "replacements": [
                    {"wrong": "李影", "correct": "李颍", "note": "讲者姓名"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    candidate = web_server._inject_course_code_candidate(terms, pdf)
    rows = web_server._load_terms_for_review(terms)

    assert candidate is not None
    assert rows[0]["kind"] == "course_code"
    assert rows[0]["correct"] == "2026-03-08-018（国）"
    assert "20260308018" in rows[0]["aliases"]
    assert rows[1]["wrong"] == "李影"


def test_terms_review_exposes_unverified_warning(tmp_path: Path) -> None:
    terms = tmp_path / "terms.json"
    terms.write_text(
        json.dumps(
            {
                "replacements": [
                    {
                        "wrong": "郭艳",
                        "correct": "郭燕",
                        "review_warning": "correct_not_verified_in_pdf_text",
                        "note": "PDF 抽取文本未能逐字验证该标准写法，请人工确认",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = web_server._load_terms_for_review(terms)

    assert rows[0]["review_warning"] == "correct_not_verified_in_pdf_text"
    assert "人工确认" in rows[0]["note"]


def test_web_config_preserves_render_preset_and_local_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(web_server, "CONFIG_PATH", tmp_path / "web_config.json")

    saved = web_server._write_web_config(
        {
            "model": "deepseek-v4-pro",
            "base_url": "https://api.deepseek.com",
            "render_preset": "用户输出预设A",
            "api_key": "user-key",
        }
    )
    loaded = web_server._read_web_config(include_secrets=True)
    public = web_server._public_web_config(loaded)

    assert saved["render_preset"] == "用户输出预设A"
    assert loaded["render_preset"] == "用户输出预设A"
    assert loaded["api_key"] == "user-key"
    assert public["has_api_key"] is True
    assert "api_key" not in public


def test_diagnostics_bundle_is_sanitized_and_includes_recent_reports(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    knowledge_dir = work_dir / "knowledge_base"
    for directory in [input_dir, output_dir, work_dir, log_dir, knowledge_dir]:
        directory.mkdir(parents=True)
    config_path = work_dir / "web_config.json"
    log_path = log_dir / "course-20260101-000000.jsonl"
    report_path = work_dir / "pdf_diagnostics" / "20260101-000000" / "course.diagnostic.json"
    terms_path = work_dir / "course-20260101-000000.terms.preflight.json"
    srt_path = work_dir / "course.optimized.srt"
    report_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "model": "qwen",
                "base_url": "https://api.example.com",
                "api_key": "sk-secret-user-key",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log_path.write_text('{"step":"pdf","status":"failed","detail":"sk-secret-user-key"}\n', encoding="utf-8")
    report_path.write_text(json.dumps({"status": "weak"}, ensure_ascii=False), encoding="utf-8")
    terms_path.write_text(json.dumps({"replacements": [{"wrong": "gird", "correct": "GERD（胃食管反流病）"}]}, ensure_ascii=False), encoding="utf-8")
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\n字幕\n", encoding="utf-8")

    monkeypatch.setattr(web_server, "INPUT_DIR", input_dir)
    monkeypatch.setattr(web_server, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(web_server, "WORK_DIR", work_dir)
    monkeypatch.setattr(web_server, "LOG_DIR", log_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_AUTO_FILE", knowledge_dir / "medical_terms.json")
    monkeypatch.setattr(web_server, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        web_server,
        "_current_job",
        {
            "status": "failed",
            "error": "PDF failed",
            "pending": {"api_key": "sk-secret-user-key"},
            "terms_path": str(terms_path),
        },
    )

    bundle = web_server._build_diagnostics_bundle()

    assert bundle.exists()
    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "web_config.sanitized.json" in names
        assert any(name.endswith("course-20260101-000000.jsonl") for name in names)
        assert any(name.endswith("course.diagnostic.json") for name in names)
        assert any(name.endswith("course-20260101-000000.terms.preflight.json") for name in names)
        assert all(not name.endswith(".srt") for name in names)
        raw = b"\n".join(zf.read(name) for name in names)
    assert b"sk-secret-user-key" not in raw
    assert "sk-s...-key".encode() in raw


def test_effective_llm_config_prefers_payload_then_saved_then_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(web_server, "CONFIG_PATH", tmp_path / "web_config.json")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    web_server._write_web_config(
        {
            "model": "saved-model",
            "base_url": "https://saved.example.com",
            "api_key": "saved-key",
        }
    )

    from_saved = web_server._effective_llm_config()
    from_payload = web_server._effective_llm_config(
        model="payload-model",
        base_url="https://payload.example.com",
        api_key="payload-key",
    )

    assert from_saved["model"] == "saved-model"
    assert from_saved["base_url"] == "https://saved.example.com"
    assert from_saved["api_key"] == "saved-key"
    assert from_payload["model"] == "payload-model"
    assert from_payload["base_url"] == "https://payload.example.com"
    assert from_payload["api_key"] == "payload-key"


def test_final_srt_candidates_include_source_video(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    video = tmp_path / "input" / "course.mp4"
    srt = work_dir / "course-20260101-010203.optimized.srt"
    video.parent.mkdir(parents=True)
    work_dir.mkdir()
    log_dir.mkdir()
    video.write_bytes(b"fake video")
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n字幕\n", encoding="utf-8")
    (log_dir / "course-20260101-010203.jsonl").write_text(
        json.dumps(
            {
                "step": "pipeline",
                "status": "complete",
                "message": "done",
                "data": {"video": str(video), "srt": str(srt)},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "WORK_DIR", work_dir)
    monkeypatch.setattr(web_server, "LOG_DIR", log_dir)

    candidates = web_server._list_final_srt_candidates()

    assert candidates[0]["srt"] == str(srt)
    assert candidates[0]["video"] == str(video)
    assert candidates[0]["video_name"] == "course.mp4"


def test_final_srt_candidates_prefer_llm_reviewed_over_optimized(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    video = tmp_path / "input" / "course.mp4"
    optimized = work_dir / "course-20260101-010203.optimized.srt"
    reviewed = work_dir / "course-20260101-010203.llm-reviewed.srt"
    video.parent.mkdir(parents=True)
    work_dir.mkdir()
    log_dir.mkdir()
    video.write_bytes(b"fake video")
    optimized.write_text("1\n00:00:00,000 --> 00:00:01,000\n旧最终字幕\n", encoding="utf-8")
    reviewed.write_text("1\n00:00:00,000 --> 00:00:01,000\nLLM最终字幕\n", encoding="utf-8")
    os.utime(optimized, (3000, 3000))
    os.utime(reviewed, (2000, 2000))
    (log_dir / "course-20260101-010203.jsonl").write_text(
        json.dumps(
            {
                "step": "pipeline",
                "status": "complete",
                "message": "done",
                "data": {"video": str(video), "srt": str(reviewed)},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "WORK_DIR", work_dir)
    monkeypatch.setattr(web_server, "LOG_DIR", log_dir)

    candidates = web_server._list_final_srt_candidates()

    assert candidates[0]["srt"] == str(reviewed)
    assert candidates[0]["stage"] == "llm-reviewed"
    assert all(item["srt"] != str(optimized) for item in candidates)


def test_initial_asr_candidates_only_include_raw_asr_srt(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    video = tmp_path / "input" / "course.mp4"
    raw_srt = work_dir / "course-20260101-010203.bcut.srt"
    optimized_srt = work_dir / "course-20260101-010203.optimized.srt"
    reviewed_srt = work_dir / "course-20260101-010203.llm-reviewed.srt"
    terms = work_dir / "course-20260101-010203.terms.preflight.approved-20260101-010203.json"
    work_dir.mkdir()
    log_dir.mkdir()
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video")
    raw_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n初版字幕\n", encoding="utf-8")
    optimized_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n最终字幕\n", encoding="utf-8")
    reviewed_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nLLM最终字幕\n", encoding="utf-8")
    terms.write_text(json.dumps({"replacements": []}, ensure_ascii=False), encoding="utf-8")
    os.utime(raw_srt, (2000, 2000))
    os.utime(optimized_srt, (3000, 3000))
    os.utime(reviewed_srt, (4000, 4000))
    (log_dir / "course-20260101-010203.jsonl").write_text(
        json.dumps(
            {
                "step": "pipeline",
                "status": "running",
                "message": "running",
                "data": {"video": str(video), "srt": str(raw_srt)},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "WORK_DIR", work_dir)
    monkeypatch.setattr(web_server, "LOG_DIR", log_dir)

    candidates = web_server._list_initial_asr_candidates()

    assert len(candidates) == 1
    assert candidates[0]["srt"] == str(raw_srt)
    assert candidates[0]["video"] == str(video)
    assert candidates[0]["terms"] == str(terms)
    assert candidates[0]["cue_count"] == 1


def test_initial_asr_candidates_fuzzy_match_current_video_name(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    work_dir.mkdir()
    log_dir.mkdir()
    matched = work_dir / "02-黄绮芳-高血压诊断标准_20260624-213510.bcut.srt"
    other = work_dir / "完全不同课程_20260624-213510.bcut.srt"
    matched.write_text("1\n00:00:00,000 --> 00:00:01,000\n初版字幕\n", encoding="utf-8")
    other.write_text("1\n00:00:00,000 --> 00:00:01,000\n其他字幕\n", encoding="utf-8")
    monkeypatch.setattr(web_server, "WORK_DIR", work_dir)
    monkeypatch.setattr(web_server, "LOG_DIR", log_dir)

    candidates = web_server._list_matching_initial_asr_candidates("02-黄绮芳 高血压诊断标准.mp4")

    assert candidates
    assert candidates[0]["srt"] == str(matched)
    assert candidates[0]["match_score"] > 0.7


def test_initial_asr_candidates_match_user_supplied_plain_srt(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    work_dir.mkdir()
    log_dir.mkdir()
    matched = work_dir / "02 黄绮芳 高血压诊断标准 初版转写.srt"
    optimized = work_dir / "02 黄绮芳 高血压诊断标准.optimized.srt"
    other = work_dir / "完全不同课程 初版转写.srt"
    matched.write_text("1\n00:00:00,000 --> 00:00:01,000\n初版字幕\n", encoding="utf-8")
    optimized.write_text("1\n00:00:00,000 --> 00:00:01,000\n最终字幕\n", encoding="utf-8")
    other.write_text("1\n00:00:00,000 --> 00:00:01,000\n其他字幕\n", encoding="utf-8")
    os.utime(optimized, (4000, 4000))
    os.utime(other, (5000, 5000))
    monkeypatch.setattr(web_server, "WORK_DIR", work_dir)
    monkeypatch.setattr(web_server, "LOG_DIR", log_dir)

    candidates = web_server._list_matching_initial_asr_candidates("02-黄绮芳 高血压诊断标准.mp4")

    assert candidates
    assert candidates[0]["srt"] == str(matched)
    assert all(".optimized." not in item["srt"] for item in candidates)


def test_upload_uses_video_path_without_copying_video(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    log_dir = tmp_path / "logs"
    video = tmp_path / "source" / "course.mp4"
    input_dir.mkdir()
    log_dir.mkdir()
    video.parent.mkdir()
    video.write_bytes(b"fake video")
    captured: dict[str, object] = {}

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(web_server, "INPUT_DIR", input_dir)
    monkeypatch.setattr(web_server, "LOG_DIR", log_dir)
    monkeypatch.setattr(web_server.threading, "Thread", FakeThread)
    monkeypatch.setattr(web_server, "_validate_llm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(web_server, "_current_job", {"status": "idle"})

    result = asyncio.run(
        web_server.upload(
            video=None,
            pdf=None,
            video_path=str(video),
            engine="bcut",
            api_key="",
            model="qwen",
            base_url="http://127.0.0.1:1234",
            segmented_asr="",
            asr_max_workers=1,
            asr_segment_minutes=10.0,
            srt_only="1",
            render_preset="用户预设",
            remove_fillers="嗯\n呃\n啊\n呢",
            remove_punctuation="。",
            comma_as_space="1",
            subtitle_min_chars=5,
            subtitle_max_chars=20,
        )
    )

    args = captured["args"]
    assert result["status"] == "running"
    assert captured["started"] is True
    assert args[0] == video
    assert not list(input_dir.rglob("*.mp4"))
    assert web_server._current_job["video"] == str(video)
    assert web_server._current_job["video_source"] == "path"


def test_resume_local_initial_asr_starts_pipeline_without_asr(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "course.mp4"
    srt = tmp_path / "course-20260101-010203.bcut.srt"
    terms = tmp_path / "course-20260101-010203.terms.preflight.approved-20260101-010203.json"
    video.write_bytes(b"fake video")
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n初版字幕\n", encoding="utf-8")
    terms.write_text(json.dumps({"replacements": []}, ensure_ascii=False), encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(web_server.threading, "Thread", FakeThread)
    monkeypatch.setattr(web_server, "_validate_llm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(web_server, "LOG_DIR", tmp_path)
    monkeypatch.setattr(web_server, "_current_job", {"status": "idle"})
    monkeypatch.setattr(
        web_server,
        "_find_latest_initial_asr_candidate",
        lambda: {
            "run_id": "20260101-010203",
            "video": str(video),
            "srt": str(srt),
            "terms": str(terms),
            "srt_name": srt.name,
        },
    )

    result = asyncio.run(
        web_server.resume_local_initial_asr(
            {
                "model": "qwen",
                "base_url": "http://127.0.0.1:1234",
                "api_key": "",
                "srt_only": True,
                "render_preset": "用户预设",
                "remove_fillers": "嗯\n呃\n啊\n呢",
                "remove_punctuation": "。",
                "comma_as_space": True,
                "subtitle_min_chars": 5,
                "subtitle_max_chars": 20,
            }
        )
    )

    args = captured["args"]
    assert result["status"] == "running"
    assert result["source_srt"] == str(srt)
    assert result["terms"] == str(terms)
    assert captured["target"] == web_server._run_pipeline_thread
    assert captured["started"] is True
    assert args[0] == video
    assert args[1] == srt
    assert args[8] is None
    assert args[9] == terms
    assert args[12] is False
    assert args[13] is False
    assert args[17] is True
    assert args[18] == "用户预设"
    assert args[23] is True


def test_resume_local_initial_asr_upload_uses_current_video(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    input_dir = tmp_path / "input"
    log_dir = tmp_path / "logs"
    srt = work_dir / "02-黄绮芳-高血压诊断标准_20260624-213510.bcut.srt"
    work_dir.mkdir()
    input_dir.mkdir()
    log_dir.mkdir()
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n初版字幕\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    class FakeUpload:
        filename = "02-黄绮芳 高血压诊断标准.mp4"

        async def read(self):
            return b"fake video"

    monkeypatch.setattr(web_server, "WORK_DIR", work_dir)
    monkeypatch.setattr(web_server, "INPUT_DIR", input_dir)
    monkeypatch.setattr(web_server, "LOG_DIR", log_dir)
    monkeypatch.setattr(web_server.threading, "Thread", FakeThread)
    monkeypatch.setattr(web_server, "_validate_llm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(web_server, "_current_job", {"status": "idle"})

    result = asyncio.run(
        web_server.resume_local_initial_asr_upload(
            video=FakeUpload(),
            model="qwen",
            base_url="http://127.0.0.1:1234",
            api_key="",
            srt_only="1",
            render_preset="用户预设",
            remove_fillers="嗯\n呃\n啊\n呢",
            remove_punctuation="。",
            comma_as_space="1",
            subtitle_min_chars=5,
            subtitle_max_chars=20,
        )
    )

    args = captured["args"]
    assert result["status"] == "running"
    assert result["source_srt"] == str(srt)
    assert Path(result["matched_video"]).name == "02-黄绮芳 高血压诊断标准.mp4"
    assert captured["target"] == web_server._run_pipeline_thread
    assert captured["started"] is True
    assert Path(args[0]).name == "02-黄绮芳 高血压诊断标准.mp4"
    assert args[1] == srt
    assert args[13] is False
    assert args[17] is True


def test_revise_srt_can_start_revision_output_thread(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    log_dir = tmp_path / "logs"
    video = tmp_path / "input" / "course.mp4"
    source = work_dir / "course-20260101-010203.optimized.srt"
    work_dir.mkdir()
    log_dir.mkdir()
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video")
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n这里是错词\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(web_server, "WORK_DIR", work_dir)
    monkeypatch.setattr(web_server, "LOG_DIR", log_dir)
    monkeypatch.setattr(web_server.threading, "Thread", FakeThread)
    monkeypatch.setattr(web_server, "_current_job", {"status": "idle"})

    result = asyncio.run(
        web_server.revise_srt(
            {
                "srt": str(source),
                "video": str(video),
                "replacements": [{"wrong": "错词", "correct": "正词"}],
                "render_after": True,
                "render_preset": "用户预设",
                "srt_only": False,
            }
        )
    )

    args = captured["args"]
    assert result["status"] == "running"
    assert result["video"] == str(video)
    assert result["render_preset"] == "用户预设"
    assert Path(result["output_srt"]).read_text(encoding="utf-8").count("正词") == 1
    assert captured["target"] == web_server._run_revision_output_thread
    assert captured["started"] is True
    assert args[0] == video
    assert Path(args[1]) == Path(result["output_srt"])
    assert args[4] == "用户预设"
    assert args[5] is False


def test_llm_test_does_not_overwrite_saved_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "web_config.json"
    monkeypatch.setattr(web_server, "CONFIG_PATH", config_path)
    web_server._write_web_config(
        {
            "model": "deepseek-v4-pro",
            "base_url": "https://api.deepseek.com",
            "render_preset": "ffpg-fast-23",
        }
    )
    monkeypatch.setattr(web_server, "_validate_llm", lambda *_args, **_kwargs: None)

    result = asyncio.run(
        web_server.test_llm(
            {
                "model": "qwen/qwen3-8b",
                "base_url": "http://127.0.0.1:1234",
                "api_key": "",
            }
        )
    )
    loaded = web_server._read_web_config()

    assert result["ok"] is True
    assert loaded["model"] == "deepseek-v4-pro"
    assert loaded["base_url"] == "https://api.deepseek.com"
    assert loaded["render_preset"] == "ffpg-fast-23"


def test_knowledge_status_reports_read_only_database(tmp_path: Path, monkeypatch) -> None:
    knowledge_file = tmp_path / "medical_knowledge.json"
    knowledge_file.write_text(
        json.dumps(
            {
                "entries": [
                    {"term": "mmHg", "note": "血压单位"},
                    {"term": "ACEI", "note": "药物类别缩写"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "KNOWLEDGE_FILES", [knowledge_file])
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", tmp_path / "missing")

    result = asyncio.run(web_server.knowledge_status())

    assert result["knowledge"]["status"] == "ok"
    assert result["knowledge"]["entry_count"] == 2
    assert "人名" in result["knowledge"]["boundary"]


def test_knowledge_entries_returns_editable_rows(tmp_path: Path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_file = knowledge_dir / "medical_terms.json"
    knowledge_dir.mkdir()
    knowledge_file.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "wrong": "儿查酚胺",
                        "correct": "儿茶酚胺",
                        "aliases": ["儿茶酚安"],
                        "confidence": "high",
                        "note": "专业名词",
                        "updated_at": "2026-06-25T12:00:00",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_AUTO_FILE", knowledge_file)
    monkeypatch.setattr(web_server, "KNOWLEDGE_FILES", [])

    result = asyncio.run(web_server.knowledge_entries())

    assert result["knowledge"]["entry_count"] == 1
    assert result["entries"][0]["index"] == 0
    assert result["entries"][0]["wrong"] == "儿查酚胺"
    assert result["entries"][0]["aliases"] == ["儿茶酚安"]
    assert result["entries"][0]["enabled"] is True


def test_update_knowledge_entry_saves_user_edit(tmp_path: Path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_file = knowledge_dir / "medical_terms.json"
    knowledge_dir.mkdir()
    knowledge_file.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "wrong": "ACEI",
                        "correct": "ACEI（血管紧张素转换酶抑制剂）",
                        "aliases": [],
                        "patterns": [],
                        "confidence": "high",
                        "note": "英文缩写",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_AUTO_FILE", knowledge_file)
    monkeypatch.setattr(web_server, "KNOWLEDGE_FILES", [])

    result = asyncio.run(
        web_server.update_knowledge_entry(
            0,
            {
                "wrong": "ACE-I",
                "correct": "血管紧张素转换酶抑制剂",
                "note": "英文缩写，字幕需保留缩写",
                "enabled": False,
            },
        )
    )

    data = json.loads(knowledge_file.read_text(encoding="utf-8"))
    assert result["status"] == "done"
    assert data["entries"][0]["wrong"] == "ACE-I"
    assert data["entries"][0]["correct"] == "ACE-I（血管紧张素转换酶抑制剂）"
    assert data["entries"][0]["enabled"] is False


def test_import_knowledge_pdf_merges_reusable_terms(tmp_path: Path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "knowledge"
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_AUTO_FILE", knowledge_dir / "medical_terms.json")
    monkeypatch.setattr(web_server, "KNOWLEDGE_FILES", [])
    monkeypatch.setattr(web_server, "_validate_llm", lambda *_args, **_kwargs: None)

    def fake_generate_terms_from_context(context, output, **_kwargs):
        payload = {
            "replacements": [
                {"wrong": "儿查酚胺", "correct": "儿茶酚胺", "confidence": "high", "evidence": "PDF", "note": "专业名词"},
                {"wrong": "陈欣", "correct": "陈歆", "confidence": "high", "evidence": "PDF", "note": "人名"},
            ]
        }
        output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    import term_mapper

    monkeypatch.setattr(term_mapper, "generate_terms_from_context", fake_generate_terms_from_context)

    class FakeUpload:
        filename = "course.pdf"

        async def read(self):
            return b"%PDF-1.4 fake"

    result = asyncio.run(
        web_server.import_knowledge_pdf(
            files=[FakeUpload()],
            model="model",
            base_url="http://127.0.0.1:1234",
            api_key="",
            system_prompt="",
        )
    )

    assert result["status"] == "done"
    assert result["added_count"] == 1
    assert result["skipped_count"] == 1
    assert result["imported"][0]["candidate_count"] == 2
    data = json.loads((knowledge_dir / "medical_terms.json").read_text(encoding="utf-8"))
    assert data["entries"][0]["correct"] == "儿茶酚胺"


def test_diagnose_pdf_upload_returns_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(web_server, "WORK_DIR", tmp_path / "work")

    import term_mapper

    monkeypatch.setattr(
        term_mapper,
        "diagnose_pdf",
        lambda path: {
            "file": path.name,
            "status": "ok",
            "message": "PDF 文字层可用",
            "best_char_count": 300,
            "methods": [{"name": "pypdf", "ok": True, "char_count": 300}],
            "preview": "课程术语",
        },
    )

    class FakeUpload:
        filename = "course.pdf"

        async def read(self):
            return b"%PDF-1.4 fake"

    result = asyncio.run(web_server.diagnose_pdf_upload(file=FakeUpload()))

    assert result["status"] == "done"
    assert result["diagnostic"]["status"] == "ok"
    assert Path(result["report"]).exists()


def test_import_knowledge_pdf_reports_zero_terms_as_skipped(tmp_path: Path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "knowledge"
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_AUTO_FILE", knowledge_dir / "medical_terms.json")
    monkeypatch.setattr(web_server, "KNOWLEDGE_FILES", [])
    monkeypatch.setattr(web_server, "_validate_llm", lambda *_args, **_kwargs: None)

    def fake_generate_terms_from_context(context, output, **_kwargs):
        payload = {
            "replacements": [],
            "dropped_replacements": [{"wrong": "错误", "correct": "正确", "drop_reason": "correct_not_in_pdf_context"}],
            "context_char_count": 1200,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    import term_mapper

    monkeypatch.setattr(term_mapper, "generate_terms_from_context", fake_generate_terms_from_context)

    class FakeUpload:
        filename = "course.pdf"

        async def read(self):
            return b"%PDF-1.4 fake"

    result = asyncio.run(
        web_server.import_knowledge_pdf(
            files=[FakeUpload()],
            model="model",
            base_url="http://127.0.0.1:1234",
            api_key="",
            system_prompt="",
        )
    )

    assert result["status"] == "done"
    assert result["added_count"] == 0
    assert result["updated_count"] == 0
    assert result["skipped_count"] == 0
    assert result["imported"][0]["status"] == "skipped"
    assert "没有得到可入库术语" in result["imported"][0]["error"]
    assert result["imported"][0]["dropped_count"] == 1


def test_merge_terms_into_knowledge_keeps_only_stable_medical_terms(tmp_path: Path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_file = knowledge_dir / "medical_terms.json"
    source_terms = tmp_path / "course.terms.approved.json"
    source_terms.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_AUTO_FILE", knowledge_file)

    result = web_server._merge_terms_into_knowledge(
        [
            {
                "wrong": "儿查酚胺",
                "correct": "儿茶酚胺",
                "aliases": ["儿茶酚安"],
                "patterns": [],
                "confidence": "high",
                "evidence": "PDF第6页",
                "note": "专业名词，同音字容易误识别",
            },
            {
                "wrong": "陈欣",
                "correct": "陈歆",
                "confidence": "high",
                "evidence": "PDF第1页作者署名：陈歆",
                "note": "人名，ASR易将歆误识为欣",
            },
            {
                "wrong": "上海交大医学院附属瑞金医院",
                "correct": "上海交通大学医学院附属瑞金医院",
                "confidence": "medium",
                "evidence": "PDF第1页",
                "note": "机构全称",
            },
        ],
        "run-knowledge",
        source_terms,
    )

    assert result["added_count"] == 1
    assert result["skipped_count"] == 2
    data = json.loads(knowledge_file.read_text(encoding="utf-8"))
    assert data["entry_count"] == 1
    assert data["entries"][0]["correct"] == "儿茶酚胺"
    assert data["entries"][0]["source_run_ids"] == ["run-knowledge"]


def test_merge_terms_into_knowledge_keeps_abbreviation_with_translation(tmp_path: Path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_file = knowledge_dir / "medical_terms.json"
    source_terms = tmp_path / "course.terms.approved.json"
    source_terms.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_AUTO_FILE", knowledge_file)

    result = web_server._merge_terms_into_knowledge(
        [
            {
                "wrong": "ACEI",
                "correct": "血管紧张素转换酶抑制剂",
                "confidence": "high",
                "evidence": "PDF",
                "note": "英文缩写",
            }
        ],
        "run-acei",
        source_terms,
    )

    assert result["added_count"] == 1
    data = json.loads(knowledge_file.read_text(encoding="utf-8"))
    assert data["entries"][0]["correct"] == "ACEI（血管紧张素转换酶抑制剂）"


def test_course_code_candidate_is_not_merged_into_knowledge(tmp_path: Path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_file = knowledge_dir / "medical_terms.json"
    source_terms = tmp_path / "course.terms.approved.json"
    source_terms.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_AUTO_FILE", knowledge_file)

    result = web_server._merge_terms_into_knowledge(
        [
            {
                "kind": "course_code",
                "wrong": "2026-03-08-018（国）",
                "correct": "2026-03-08-018（国）",
                "confidence": "high",
                "evidence": "PDF文件名",
                "note": "本课程编号元数据",
            }
        ],
        "run-course-code",
        source_terms,
    )

    assert result["added_count"] == 0
    assert result["skipped_count"] == 1


def test_merge_terms_into_knowledge_updates_existing_entry(tmp_path: Path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_file = knowledge_dir / "medical_terms.json"
    source_terms = tmp_path / "course.terms.approved.json"
    source_terms.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(web_server, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(web_server, "KNOWLEDGE_AUTO_FILE", knowledge_file)

    row = {
        "wrong": "先兆子闲",
        "correct": "先兆子痫",
        "confidence": "medium",
        "evidence": "PDF第31页",
        "note": "专业名词，ASR易混淆",
    }

    first = web_server._merge_terms_into_knowledge([row], "run-1", source_terms)
    second = web_server._merge_terms_into_knowledge([row], "run-2", source_terms)

    data = json.loads(knowledge_file.read_text(encoding="utf-8"))
    assert first["added_count"] == 1
    assert second["added_count"] == 0
    assert second["updated_count"] == 1
    assert data["entry_count"] == 1
    assert data["entries"][0]["occurrence_count"] == 2
    assert data["entries"][0]["source_run_ids"] == ["run-1", "run-2"]
