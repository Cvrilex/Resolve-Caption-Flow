import asyncio
import json
import sys
from pathlib import Path


MVP_DIR = Path(__file__).resolve().parents[1] / "mvp_pipeline"
_INSERTED_MVP_DIR = str(MVP_DIR) not in sys.path
if _INSERTED_MVP_DIR:
    sys.path.insert(0, str(MVP_DIR))

import web_server  # noqa: E402

if _INSERTED_MVP_DIR:
    sys.path.remove(str(MVP_DIR))


def test_confirm_terms_preserves_segment_settings(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "course.mp4"
    video.write_bytes(b"fake")
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
                "engine": "local",
                "segmented_asr": True,
                "asr_max_workers": 2,
                "asr_segment_minutes": 8.0,
                "asr_max_segment_minutes": 15.5,
                "srt_only": True,
                "render_preset": "my-resolve-preset",
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
    assert args[13] is True
    assert args[14] == 2
    assert args[15] == 8.0
    assert args[16] == 15.5
    assert args[17] is True
    assert args[18] == "my-resolve-preset"
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


def test_web_config_preserves_render_preset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(web_server, "CONFIG_PATH", tmp_path / "web_config.json")

    saved = web_server._write_web_config(
        {
            "model": "deepseek-v4-pro",
            "base_url": "https://api.deepseek.com",
            "render_preset": "用户输出预设A",
            "api_key": "should-not-save",
        }
    )
    loaded = web_server._read_web_config()

    assert saved["render_preset"] == "用户输出预设A"
    assert loaded["render_preset"] == "用户输出预设A"
    assert "api_key" not in loaded


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
