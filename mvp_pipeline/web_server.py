#!/usr/bin/env python3
"""Web panel for the medical captioning pipeline — FastAPI + SSE progress."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
WORK_DIR = ROOT / "work"
LOG_DIR = ROOT / "logs"
WEB_DIR = ROOT / "web"

for d in [INPUT_DIR, OUTPUT_DIR, WORK_DIR, LOG_DIR, WEB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="医疗字幕自动化")

# In-memory state for the running job
_job_lock = threading.Lock()
_current_job: dict[str, Any] = {"status": "idle", "log_path": None, "output_path": None}
_active_thread: Optional[threading.Thread] = None

DEFAULT_LLM_MODEL = "deepseek-v4-pro"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEMPLATE_PROJECT = ROOT / "sub.drp"


# ── SSE helpers ────────────────────────────────────────────────────────────────

def _tail_log(log_path: Path):
    """Generator that yields SSE events by tailing a JSONL log file."""
    yield f"data: {json.dumps({'step': 'connected', 'status': 'running', 'message': '流水线启动中...'})}\n\n"

    last_pos = 0
    max_idle = 300  # 5 min timeout
    idle = 0

    while idle < max_idle:
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            if event.get("status") == "review":
                                for _ in range(30):
                                    with _job_lock:
                                        current_status = _current_job.get("status")
                                    if current_status != "running":
                                        break
                                    time.sleep(0.1)
                                with _job_lock:
                                    current_status = _current_job.get("status")
                                if current_status != "awaiting_terms":
                                    continue
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                            if event.get("status") == "complete":
                                return
                            if event.get("status") == "review":
                                return
                            if event.get("status") == "failed":
                                yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': event.get('message', '流水线失败')})}\n\n"
                                return
                        except json.JSONDecodeError:
                            pass
                last_pos = f.tell()
            idle = 0
        else:
            idle += 1
        time.sleep(1)

    yield f"data: {json.dumps({'step': 'error', 'status': 'timeout', 'message': '等待流水线启动超时'})}\n\n"


def _safe_filename(filename: str, fallback: str) -> str:
    name = Path(filename or fallback).name
    name = re.sub(r"[\x00-\x1f]", "", name).strip()
    return name or fallback


def _is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _normalize_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if _is_local_base_url(normalized) and not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized


def _sanitize_llm_error(text: str) -> str:
    text = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer ***", text)
    text = re.sub(r"sk-[A-Za-z0-9._\-]+", "sk-***", text)
    return text.strip()[:800]


def _validate_llm(model: str, base_url: str, api_key: str) -> Optional[str]:
    """Fail fast on bad LLM settings before the expensive ASR/render pipeline starts."""
    if not model:
        return "请填写模型名"
    if not base_url:
        return "请填写 API 地址"

    local = _is_local_base_url(base_url)
    effective_key = api_key or ("local" if local else "")
    if not effective_key:
        return "请填写 API Key，或使用本地 LM Studio 地址"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "只输出JSON。"},
            {"role": "user", "content": '{"ok": true}'},
        ],
        "temperature": 0.1,
        "max_tokens": 32,
    }
    if not local:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {effective_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return f"LLM 预检失败：HTTP {exc.code} {_sanitize_llm_error(detail)}"
    except urllib.error.URLError as exc:
        return f"LLM 预检失败：无法连接到 {base_url}，{_sanitize_llm_error(str(exc.reason))}"
    except Exception as exc:
        return f"LLM 预检失败：{_sanitize_llm_error(str(exc))}"

    try:
        content = data["choices"][0]["message"].get("content", "")
    except Exception:
        return f"LLM 预检失败：返回格式不符合 OpenAI chat/completions：{_sanitize_llm_error(json.dumps(data, ensure_ascii=False))}"
    if content is None:
        return "LLM 预检失败：模型没有返回内容"
    return None


def _load_terms_for_review(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("replacements", []) if isinstance(data, dict) else []
    result: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        wrong = str(item.get("wrong", "")).strip()
        correct = str(item.get("correct", "")).strip()
        if not wrong or not correct or wrong == correct:
            continue
        result.append(
            {
                "id": index,
                "enabled": True,
                "wrong": wrong,
                "correct": correct,
                "confidence": item.get("confidence", ""),
                "evidence": str(item.get("evidence", "")),
                "note": str(item.get("note", "")),
            }
        )
    return result


def _write_approved_terms(source: Path, rows: list[dict[str, Any]], run_id: str) -> tuple[Path, int]:
    replacements: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not row.get("enabled", True):
            continue
        wrong = str(row.get("wrong", "")).strip()
        correct = str(row.get("correct", "")).strip()
        if not wrong or not correct or wrong == correct:
            continue
        key = (wrong, correct)
        if key in seen:
            continue
        seen.add(key)
        replacements.append(
            {
                "wrong": wrong,
                "correct": correct,
                "confidence": row.get("confidence", ""),
                "evidence": str(row.get("evidence", "")),
                "note": str(row.get("note", "")),
            }
        )
    output = source.with_name(f"{source.stem}.approved-{run_id}.json")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_terms": str(source),
        "replacements": replacements,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, len(replacements)


def _read_log_events(log_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    except OSError:
        pass
    return events


def _candidate_from_log(log_path: Path) -> dict[str, Any] | None:
    events = _read_log_events(log_path)
    if not events:
        return None
    video_path = ""
    pdf_path = ""
    srt_path = ""
    cue_count: int | None = None
    status = "unknown"
    message = ""
    run_id = ""

    match = re.search(r"-(\d{8}-\d{6})\.jsonl$", log_path.name)
    if match:
        run_id = match.group(1)

    for event in events:
        step = str(event.get("step") or "")
        event_status = str(event.get("status") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if step == "pipeline" and event_status == "start":
            video_path = str(data.get("video") or video_path)
        if step == "term_map":
            pdf_path = str(data.get("context") or pdf_path)
        if step in {"asr", "srt"}:
            srt_path = str(data.get("srt") or srt_path)
            if data.get("cue_count") is not None:
                try:
                    cue_count = int(data.get("cue_count"))
                except (TypeError, ValueError):
                    pass
        if event_status in {"failed", "complete", "review"}:
            status = event_status
            message = str(event.get("message") or message)

    if status == "complete" or not srt_path:
        return None
    video = Path(video_path) if video_path else None
    srt = Path(srt_path)
    pdf = Path(pdf_path) if pdf_path else None
    if not srt.exists() or not video or not video.exists():
        return None
    return {
        "run_id": run_id,
        "status": status,
        "message": message,
        "video": str(video),
        "video_name": video.name,
        "pdf": str(pdf) if pdf and pdf.exists() else None,
        "pdf_name": pdf.name if pdf and pdf.exists() else None,
        "srt": str(srt),
        "srt_name": srt.name,
        "cue_count": cue_count,
        "log": str(log_path),
        "updated_at": datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def _list_resume_candidates(limit: int = 8) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for log_path in sorted(LOG_DIR.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True):
        candidate = _candidate_from_log(log_path)
        if candidate:
            candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def _find_resume_candidate(source_run_id: str) -> dict[str, Any] | None:
    for candidate in _list_resume_candidates(limit=50):
        if candidate.get("run_id") == source_run_id:
            return candidate
    return None


def _run_pipeline_thread(video_path: Path, srt_path: Optional[Path], engine: str,
                          log_path: Path, output_name: str, run_id: str,
                          api_key: str = "", system_prompt: str = "",
                          pdf_path: Optional[Path] = None,
                          terms_path: Optional[Path] = None,
                          model: str = "", base_url: str = "",
                          review_terms: bool = False,
                          segmented_asr: bool = False,
                          asr_max_workers: int = 1,
                          asr_segment_minutes: float = 10.0,
                          asr_max_segment_minutes: float = 12.0) -> None:
    """Run the MVP pipeline in a background thread."""
    global _current_job
    try:
        sys.path.insert(0, str(ROOT))
        from mvp_pipeline import PipelineError, run_pipeline, stem_for  # type: ignore

        class WebArgs:
            pass

        args = WebArgs()
        args.video = str(video_path)
        args.engine = engine
        args.srt = str(srt_path) if srt_path else None
        args.segmented_asr = bool(segmented_asr and not srt_path)
        args.asr_max_workers = max(1, int(asr_max_workers or 1))
        args.asr_segment_minutes = max(1.0, float(asr_segment_minutes or 10.0))
        args.asr_max_segment_minutes = max(args.asr_segment_minutes, float(asr_max_segment_minutes or 12.0))
        args.terms = str(terms_path) if terms_path else None
        args.context = str(pdf_path) if pdf_path and not terms_path else None
        args.llm_model = model or os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)
        args.llm_base_url = _normalize_base_url(base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL))
        args.llm_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not args.llm_api_key and _is_local_base_url(args.llm_base_url):
            args.llm_api_key = "local"
        args.llm_system_prompt = system_prompt
        args.term_map_timeout = 45
        args.term_map_retries = 1
        args.review_terms = review_terms
        args.optimize_subtitles = True
        args.subtitle_max_chars = 20
        args.remove_punctuation = "，,"
        args.no_subtitle_llm = False
        args.subtitle_llm_timeout = 30
        args.allow_neighbor_rewrite = False
        args.prepare_only = False
        args.render_current = False
        args.project_name = None
        args.template_project = str(DEFAULT_TEMPLATE_PROJECT) if DEFAULT_TEMPLATE_PROJECT.exists() else None
        args.use_template_timeline = bool(args.template_project)
        args.subtitle_preset = "sub01"
        args.allow_subtitle_preset_fallback = not args.use_template_timeline
        args.render_type = "x264 8-bit 4:2:0(FFmpeg)"
        args.render_preset = "ffpg-fast-23"
        args.no_render_preset = False
        args.list_resolve_presets = False
        args.allow_render_type_fallback = True
        args.run_id = run_id

        # Monkey-patch the DEFAULT_LOG_DIR so the pipeline writes exactly where we expect
        import mvp_pipeline as mp
        saved = mp.DEFAULT_LOG_DIR
        mp.DEFAULT_LOG_DIR = LOG_DIR

        try:
            result = run_pipeline(args)
        finally:
            mp.DEFAULT_LOG_DIR = saved
        if result.get("needs_term_review"):
            with _job_lock:
                _current_job["status"] = "awaiting_terms"
                _current_job["output_path"] = None
                _current_job["result"] = result
                _current_job["terms_path"] = result.get("terms")
                _current_job["srt_path"] = result.get("srt")
                _current_job["pending"] = {
                    "video_path": str(video_path),
                    "engine": engine,
                    "api_key": api_key,
                    "system_prompt": system_prompt,
                    "model": model,
                    "base_url": base_url,
                    "segmented_asr": bool(segmented_asr),
                    "asr_max_workers": max(1, int(asr_max_workers or 1)),
                    "asr_segment_minutes": max(1.0, float(asr_segment_minutes or 10.0)),
                }
            return
        with _job_lock:
            _current_job["status"] = "done"
            _current_job["output_path"] = result.get("rendered")
            _current_job["result"] = result
    except Exception as exc:
        with _job_lock:
            _current_job["status"] = "failed"
            _current_job["error"] = str(exc)
        # Write error to pipeline's log (reconstruct expected path)
        stem = video_path.stem.replace(" ", "_")
        actual_log_path = LOG_DIR / f"{stem}-{run_id}.jsonl"
        with open(actual_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "step": "pipeline", "status": "failed", "message": str(exc),
            }, ensure_ascii=False) + "\n")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the frontend page."""
    html_path = WEB_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Web panel not found. Create web/index.html</h1>")


@app.post("/api/upload")
async def upload(video: UploadFile = File(...), pdf: Optional[UploadFile] = File(None),
                 engine: str = Form("bcut"), api_key: str = Form(""),
                 system_prompt: str = Form(""), model: str = Form(""),
                 base_url: str = Form(""),
                 segmented_asr: str = Form(""),
                 asr_max_workers: int = Form(1),
                 asr_segment_minutes: float = Form(10.0)):
    """Upload video and optional PDF, then start the pipeline."""
    global _current_job

    if video.filename is None:
        return {"error": "请选择视频文件"}
    if engine != "bcut":
        return {"error": "ASR 引擎参数不正确"}
    effective_model = (model or os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)).strip()
    effective_base_url = _normalize_base_url(base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL))
    effective_api_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not (effective_api_key or _is_local_base_url(effective_base_url)):
        return {"error": "请填写 API Key，或先在环境变量里设置 OPENAI_API_KEY"}

    with _job_lock:
        if _current_job.get("status") == "running":
            return {"error": "已有任务正在运行，请等待当前任务结束"}
        if _current_job.get("status") == "awaiting_terms":
            return {"error": "当前任务正在等待术语审核，请先确认或刷新服务后再开始新任务"}

    llm_error = _validate_llm(effective_model, effective_base_url, effective_api_key)
    if llm_error:
        return {"error": llm_error}

    # Generate run ID first so uploads are isolated and do not overwrite each other
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    upload_dir = INPUT_DIR / "uploads" / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save video
    video_name = _safe_filename(video.filename, "input.mp4")
    video_path = upload_dir / video_name
    content = await video.read()
    video_path.write_bytes(content)

    # Save PDF if provided
    pdf_path: Optional[Path] = None
    if pdf and pdf.filename:
        pdf_name = _safe_filename(pdf.filename, "context.pdf")
        pdf_path = upload_dir / pdf_name
        pdf_content = await pdf.read()
        pdf_path.write_bytes(pdf_content)

    stem = video_path.stem.replace(" ", "_")
    log_path = LOG_DIR / f"{stem}-{run_id}.jsonl"
    use_segmented_asr = segmented_asr in {"1", "true", "on", "yes"}
    safe_asr_max_workers = max(1, int(asr_max_workers or 1))
    safe_asr_segment_minutes = max(1.0, float(asr_segment_minutes or 10.0))

    with _job_lock:
        _current_job = {
            "status": "running",
            "log_path": str(log_path),
            "output_path": None,
            "run_id": run_id,
            "video": str(video_path),
            "template_project": str(DEFAULT_TEMPLATE_PROJECT) if DEFAULT_TEMPLATE_PROJECT.exists() else None,
            "segmented_asr": use_segmented_asr,
            "asr_max_workers": safe_asr_max_workers,
            "asr_segment_minutes": safe_asr_segment_minutes,
        }

    # Start pipeline in background
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(
            video_path,
            None,
            engine,
            log_path,
            f"{stem}_web_{run_id}",
            run_id,
            effective_api_key,
            system_prompt,
            pdf_path,
            None,
            effective_model,
            effective_base_url,
            bool(pdf_path),
            use_segmented_asr,
            safe_asr_max_workers,
            safe_asr_segment_minutes,
            max(12.0, safe_asr_segment_minutes),
        ),
        daemon=True,
    )
    global _active_thread
    _active_thread = thread
    thread.start()

    return {"run_id": run_id, "log_path": str(log_path), "status": "running"}


@app.get("/api/resumable-runs")
async def resumable_runs():
    """List previous jobs that have an SRT and can continue without re-running ASR."""
    return {"runs": _list_resume_candidates()}


@app.post("/api/resume/{source_run_id}")
async def resume_from_srt(source_run_id: str, payload: dict[str, Any] = Body(...)):
    """Resume from a previous job's generated SRT, skipping ASR."""
    global _current_job, _active_thread
    candidate = _find_resume_candidate(source_run_id)
    if not candidate:
        return {"error": "没有找到可恢复的 SRT 结果"}

    effective_model = str(payload.get("model") or os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)).strip()
    effective_base_url = _normalize_base_url(str(payload.get("base_url") or os.environ.get("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL)))
    effective_api_key = str(payload.get("api_key") or os.environ.get("OPENAI_API_KEY") or "").strip()
    system_prompt = str(payload.get("system_prompt") or "")
    if not (effective_api_key or _is_local_base_url(effective_base_url)):
        return {"error": "请填写 API Key，或先在环境变量里设置 OPENAI_API_KEY"}

    with _job_lock:
        if _current_job.get("status") == "running":
            return {"error": "已有任务正在运行，请等待当前任务结束"}
        if _current_job.get("status") == "awaiting_terms":
            return {"error": "当前任务正在等待术语审核，请先确认或刷新服务后再恢复任务"}

    llm_error = _validate_llm(effective_model, effective_base_url, effective_api_key)
    if llm_error:
        return {"error": llm_error}

    video_path = Path(str(candidate["video"]))
    srt_path = Path(str(candidate["srt"]))
    pdf_value = candidate.get("pdf")
    pdf_path = Path(str(pdf_value)) if pdf_value else None
    if not video_path.exists():
        return {"error": "原视频不存在，无法恢复"}
    if not srt_path.exists():
        return {"error": "原 SRT 不存在，无法恢复"}

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = video_path.stem.replace(" ", "_")
    log_path = LOG_DIR / f"{stem}-{run_id}.jsonl"

    with _job_lock:
        _current_job = {
            "status": "running",
            "log_path": str(log_path),
            "output_path": None,
            "run_id": run_id,
            "source_run_id": source_run_id,
            "video": str(video_path),
            "srt_path": str(srt_path),
            "template_project": str(DEFAULT_TEMPLATE_PROJECT) if DEFAULT_TEMPLATE_PROJECT.exists() else None,
            "resume": True,
        }

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(
            video_path,
            srt_path,
            "bcut",
            log_path,
            f"{stem}_web_{run_id}",
            run_id,
            effective_api_key,
            system_prompt,
            pdf_path if pdf_path and pdf_path.exists() else None,
            None,
            effective_model,
            effective_base_url,
            bool(pdf_path and pdf_path.exists()),
            False,
            1,
            10.0,
            12.0,
        ),
        daemon=True,
    )
    _active_thread = thread
    thread.start()
    return {"run_id": run_id, "source_run_id": source_run_id, "log_path": str(log_path), "status": "running"}


@app.get("/api/terms/{run_id}")
async def terms_for_review(run_id: str):
    """Return generated terminology candidates for manual approval."""
    with _job_lock:
        job = dict(_current_job)
    if job.get("run_id") != run_id:
        return {"error": "没有找到对应任务"}
    if job.get("status") != "awaiting_terms":
        return {"error": "当前任务还没有进入术语审核阶段"}
    terms_path = Path(str(job.get("terms_path") or ""))
    if not terms_path.exists():
        return {"error": "术语候选文件不存在"}
    return {
        "run_id": run_id,
        "terms_path": str(terms_path),
        "srt_path": job.get("srt_path"),
        "replacements": _load_terms_for_review(terms_path),
    }


@app.post("/api/terms/{run_id}/confirm")
async def confirm_terms(run_id: str, payload: dict[str, Any] = Body(...)):
    """Persist approved terms and resume the pipeline from the existing SRT."""
    global _active_thread, _current_job
    with _job_lock:
        job = dict(_current_job)
        if job.get("run_id") != run_id:
            return {"error": "没有找到对应任务"}
        if job.get("status") != "awaiting_terms":
            return {"error": "当前任务不在术语审核阶段"}
        pending = dict(job.get("pending") or {})
        terms_path = Path(str(job.get("terms_path") or ""))
        srt_path = Path(str(job.get("srt_path") or ""))
        log_path = Path(str(job.get("log_path") or ""))

    rows = payload.get("replacements", [])
    if not isinstance(rows, list):
        return {"error": "术语确认数据格式不正确"}
    if not terms_path.exists():
        return {"error": "术语候选文件不存在"}
    if not srt_path.exists():
        return {"error": "待续跑 SRT 不存在"}

    approved_terms, approved_count = _write_approved_terms(terms_path, rows, run_id)
    video_path = Path(str(pending.get("video_path") or job.get("video") or ""))
    if not video_path.exists():
        return {"error": "待续跑视频不存在"}

    stem = video_path.stem.replace(" ", "_")
    with _job_lock:
        _current_job["status"] = "running"
        _current_job["approved_terms_path"] = str(approved_terms)

    continuation_terms = approved_terms if approved_count else None
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(
            video_path,
            srt_path,
            str(pending.get("engine") or "bcut"),
            log_path,
            f"{stem}_web_{run_id}",
            run_id,
            str(pending.get("api_key") or ""),
            str(pending.get("system_prompt") or ""),
            None,
            continuation_terms,
            str(pending.get("model") or ""),
            str(pending.get("base_url") or ""),
            False,
            False,
            int(pending.get("asr_max_workers") or 1),
            float(pending.get("asr_segment_minutes") or 10.0),
            max(12.0, float(pending.get("asr_segment_minutes") or 10.0)),
        ),
        daemon=True,
    )
    _active_thread = thread
    thread.start()
    return {"status": "running", "run_id": run_id, "approved_terms_path": str(approved_terms)}


@app.get("/api/progress/{run_id}")
async def progress(run_id: str):
    """SSE endpoint streaming pipeline progress."""
    matches = sorted(LOG_DIR.glob(f"*{run_id}*.jsonl"))
    if not matches:
        # Wait briefly for the log file to be created
        for _ in range(30):
            time.sleep(1)
            matches = sorted(LOG_DIR.glob(f"*{run_id}*.jsonl"))
            if matches:
                break
    if not matches:
        return StreamingResponse(
            iter([f"data: {json.dumps({'step': 'error', 'status': 'error', 'message': '日志文件未创建，请检查Resolve是否运行'})}\n\n"]),
            media_type="text/event-stream",
        )
    return StreamingResponse(
        _tail_log(matches[0]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/status")
async def status():
    """Get current job status."""
    with _job_lock:
        return dict(_current_job)


@app.get("/api/open-folder/{run_id}")
async def open_folder(run_id: str):
    """Open the output folder in Finder."""
    import subprocess
    matches = sorted(OUTPUT_DIR.glob(f"*{run_id}*"))
    folder = str(OUTPUT_DIR.resolve())
    if matches:
        folder = str(matches[0].parent.resolve())
    subprocess.run(["open", folder])
    return {"folder": folder}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="医疗字幕自动化 Web 面板")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8742, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    args = parser.parse_args()

    print(f"\n  🔤 医疗字幕自动化 Web 面板")
    print(f"  {'─' * 40}")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print()

    uvicorn.run("web_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
