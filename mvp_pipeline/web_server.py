#!/usr/bin/env python3
"""Web panel for the medical captioning pipeline — FastAPI + SSE progress."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
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
REPO_ROOT = ROOT.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
WORK_DIR = ROOT / "work"
LOG_DIR = ROOT / "logs"
WEB_DIR = ROOT / "web"
CONFIG_PATH = WORK_DIR / "web_config.json"
KNOWLEDGE_DIR = WORK_DIR / "knowledge_base"
KNOWLEDGE_AUTO_FILE = KNOWLEDGE_DIR / "medical_terms.json"
KNOWLEDGE_FILES = [
    WORK_DIR / "medical_knowledge.json",
    ROOT / "medical_knowledge.json",
]

for d in [INPUT_DIR, OUTPUT_DIR, WORK_DIR, LOG_DIR, WEB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="医疗字幕自动化")

# In-memory state for the running job
_job_lock = threading.Lock()
_current_job: dict[str, Any] = {"status": "idle", "log_path": None, "output_path": None}
_active_thread: Optional[threading.Thread] = None

DEFAULT_LLM_MODEL = "deepseek-v4-pro"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_REMOVE_FILLERS = "嗯,呃,啊"
DEFAULT_TEMPLATE_PROJECT = ROOT / "sub.drp"
BUNDLED_PYTHON = Path(
    "/Users/x/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
)
ASR_ENGINES: dict[str, dict[str, Any]] = {
    "bcut": {
        "id": "bcut",
        "label": "必剪 B 接口",
        "status": "ok",
        "available": True,
        "message": "必剪在线 ASR 可用",
    },
    "jianying": {
        "id": "jianying",
        "label": "剪映 J 接口",
        "status": "manual",
        "available": bool(os.environ.get("JIANYING_SIGN_SERVICE_URL")),
        "message": "需要可用签名服务；当前默认第三方签名服务测试返回 500",
    },
    "kuaishou": {
        "id": "kuaishou",
        "label": "快手 K 接口",
        "status": "disabled",
        "available": False,
        "message": "接口实测返回 code=501：效果 subtitle_generate 禁用",
    },
}


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
                            if event.get("status") in {"aborted", "cancel_requested"}:
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
    text = re.sub(r"(?i)(api\s*key\s*:\s*)[^,，\"'}\s]+", r"\1***", text)
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


def _read_web_config() -> dict[str, Any]:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    data.pop("api_key", None)
    data.pop("llm_api_key", None)
    return data


def _write_web_config(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "model",
        "base_url",
        "engine",
        "segmented_asr",
        "asr_max_workers",
        "asr_segment_minutes",
        "srt_only",
        "render_preset",
        "remove_fillers",
        "subtitle_min_chars",
        "subtitle_max_chars",
    }
    config: dict[str, Any] = _read_web_config()
    for key in allowed:
        if key in payload:
            config[key] = payload[key]
    config["updated_at"] = datetime.now().isoformat(timespec="seconds")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def _check_writable(path: Path) -> dict[str, Any]:
    exists = path.exists()
    writable = exists and os.access(path, os.W_OK)
    return {
        "path": str(path),
        "exists": exists,
        "writable": writable,
        "status": "ok" if writable else "error",
        "message": "可写" if writable else "目录不可写或不存在",
    }


def _command_status(name: str, label: str) -> dict[str, Any]:
    found = shutil.which(name)
    return {
        "id": name,
        "label": label,
        "status": "ok" if found else "missing",
        "available": bool(found),
        "path": found or "",
        "message": "已安装" if found else f"未找到 {label}",
    }


def _count_knowledge_entries(path: Path) -> int:
    try:
        if path.suffix.lower() == ".jsonl":
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if isinstance(data, list):
        return len(data)
    if not isinstance(data, dict):
        return 0
    for key in ("entries", "terms", "replacements", "knowledge", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    return len(data)


COURSE_SPECIFIC_KNOWLEDGE_KEYWORDS = (
    "人名",
    "姓名",
    "讲者",
    "作者",
    "署名",
    "老师",
    "教授",
    "主任",
    "机构",
    "医院",
    "大学",
    "学院",
    "学校",
    "地名",
    "项目名",
    "项目名称",
    "课程名",
    "课程名称",
)


STABLE_MEDICAL_KNOWLEDGE_KEYWORDS = (
    "专业名词",
    "医学",
    "药",
    "病名",
    "疾病",
    "术式",
    "检查",
    "检验",
    "指标",
    "单位",
    "缩写",
    "指南",
    "ASR",
    "同音",
    "易错",
    "误识别",
    "误听",
)


def _term_blob(row: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("wrong", "correct", "confidence", "evidence", "note"):
        values.append(str(row.get(key) or ""))
    for key in ("aliases", "patterns"):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
    return " ".join(values)


def _is_course_specific_term(row: dict[str, Any]) -> bool:
    blob = _term_blob(row)
    if any(keyword in blob for keyword in COURSE_SPECIFIC_KNOWLEDGE_KEYWORDS):
        return True
    correct = str(row.get("correct") or "")
    return any(keyword in correct for keyword in ("医院", "大学", "学院", "学校"))


def _is_stable_medical_term(row: dict[str, Any]) -> bool:
    if _is_course_specific_term(row):
        return False
    wrong = str(row.get("wrong") or "").strip()
    correct = str(row.get("correct") or "").strip()
    if not wrong or not correct or wrong == correct or len(correct) < 2:
        return False
    has_supporting_context = any(str(row.get(key) or "").strip() for key in ("confidence", "evidence", "note"))
    aliases = row.get("aliases", []) if isinstance(row.get("aliases"), list) else []
    patterns = row.get("patterns", []) if isinstance(row.get("patterns"), list) else []
    if not has_supporting_context and not aliases and not patterns:
        return False
    blob = _term_blob(row)
    if any(keyword in blob for keyword in STABLE_MEDICAL_KNOWLEDGE_KEYWORDS):
        return True
    if patterns:
        return True
    if re.search(r"[A-Za-z]{2,}|\d", correct) and (has_supporting_context or aliases):
        return True
    confidence = str(row.get("confidence") or "").lower()
    if confidence in {"high", "medium"} and str(row.get("evidence") or "").strip() and len(correct) >= 3:
        return True
    return False


def _load_knowledge_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "entries": []}
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return data
    if isinstance(data, list):
        return {"version": 1, "entries": data}
    return {"version": 1, "entries": []}


def _merge_terms_into_knowledge(rows: list[dict[str, Any]], run_id: str, source_terms: Path) -> dict[str, Any]:
    stable_rows = [row for row in rows if _is_stable_medical_term(row)]
    skipped_count = len(rows) - len(stable_rows)
    if not stable_rows:
        return {
            "path": str(KNOWLEDGE_AUTO_FILE),
            "added_count": 0,
            "updated_count": 0,
            "skipped_count": skipped_count,
            "entry_count": _count_knowledge_entries(KNOWLEDGE_AUTO_FILE) if KNOWLEDGE_AUTO_FILE.exists() else 0,
        }

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    payload = _load_knowledge_payload(KNOWLEDGE_AUTO_FILE)
    entries = payload.setdefault("entries", [])
    if not isinstance(entries, list):
        entries = []
        payload["entries"] = entries

    existing: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("wrong") or ""), str(entry.get("correct") or ""))
        existing[key] = entry

    now = datetime.now().isoformat(timespec="seconds")
    added_count = 0
    updated_count = 0
    for row in stable_rows:
        wrong = str(row.get("wrong") or "").strip()
        correct = str(row.get("correct") or "").strip()
        key = (wrong, correct)
        aliases = row.get("aliases", []) if isinstance(row.get("aliases"), list) else []
        patterns = row.get("patterns", []) if isinstance(row.get("patterns"), list) else []
        entry = existing.get(key)
        if entry is None:
            entry = {
                "wrong": wrong,
                "correct": correct,
                "aliases": aliases,
                "patterns": patterns,
                "confidence": str(row.get("confidence") or ""),
                "evidence": str(row.get("evidence") or ""),
                "note": str(row.get("note") or ""),
                "source": "approved_course_terms",
                "source_terms": [str(source_terms)],
                "source_run_ids": [run_id],
                "occurrence_count": 1,
                "created_at": now,
                "updated_at": now,
            }
            entries.append(entry)
            existing[key] = entry
            added_count += 1
            continue

        entry["updated_at"] = now
        entry["occurrence_count"] = int(entry.get("occurrence_count") or 0) + 1
        entry["aliases"] = sorted(set([*([str(v) for v in entry.get("aliases", [])] if isinstance(entry.get("aliases"), list) else []), *map(str, aliases)]))
        entry["patterns"] = sorted(set([*([str(v) for v in entry.get("patterns", [])] if isinstance(entry.get("patterns"), list) else []), *map(str, patterns)]))
        source_terms_list = entry.get("source_terms") if isinstance(entry.get("source_terms"), list) else []
        if str(source_terms) not in source_terms_list:
            source_terms_list.append(str(source_terms))
        entry["source_terms"] = source_terms_list
        source_runs = entry.get("source_run_ids") if isinstance(entry.get("source_run_ids"), list) else []
        if run_id not in source_runs:
            source_runs.append(run_id)
        entry["source_run_ids"] = source_runs
        updated_count += 1

    payload["updated_at"] = now
    payload["entry_count"] = len(entries)
    KNOWLEDGE_AUTO_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "path": str(KNOWLEDGE_AUTO_FILE),
        "added_count": added_count,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "entry_count": len(entries),
    }


def _knowledge_status() -> dict[str, Any]:
    candidates: list[Path] = []
    for path in KNOWLEDGE_FILES:
        if path.exists() and path.is_file():
            candidates.append(path)
    if KNOWLEDGE_DIR.exists() and KNOWLEDGE_DIR.is_dir():
        candidates.extend(
            sorted(
                item for item in KNOWLEDGE_DIR.glob("*")
                if item.is_file() and item.suffix.lower() in {".json", ".jsonl"}
            )
        )

    entry_count = sum(_count_knowledge_entries(path) for path in candidates)
    latest_mtime = 0.0
    for path in candidates:
        try:
            latest_mtime = max(latest_mtime, path.stat().st_mtime)
        except OSError:
            pass

    if candidates:
        status = "ok"
        message = "跨课程医疗知识库可用；课程人名、学校名、机构名不会写入这里"
    else:
        status = "missing"
        message = "暂未建立跨课程医疗知识库；当前任务只使用本课程术语表"

    return {
        "status": status,
        "available": bool(candidates),
        "entry_count": entry_count,
        "paths": [str(path) for path in candidates],
        "directory": str(KNOWLEDGE_DIR),
        "updated_at": datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds") if latest_mtime else None,
        "message": message,
        "boundary": "人名、学校名、医院名、机构名等当次课程强相关信息只进入本课程术语表，不进入跨项目医疗知识库。",
    }


def _resolve_status() -> dict[str, Any]:
    app_path = Path("/Applications/DaVinci Resolve/DaVinci Resolve.app")
    template_ok = DEFAULT_TEMPLATE_PROJECT.exists()
    app_ok = app_path.exists()
    if app_ok and template_ok:
        status = "ok"
        message = "Resolve 和字幕模板已找到"
    elif not app_ok:
        status = "manual"
        message = "未检测到 DaVinci Resolve，需要用户安装"
    else:
        status = "missing"
        message = "未找到字幕模板 sub.drp"
    return {
        "status": status,
        "available": app_ok and template_ok,
        "app_path": str(app_path),
        "template_project": str(DEFAULT_TEMPLATE_PROJECT),
        "template_available": template_ok,
        "message": message,
    }


def _llm_config_status(config: dict[str, Any]) -> dict[str, Any]:
    model = str(config.get("model") or os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)).strip()
    base_url = _normalize_base_url(str(config.get("base_url") or os.environ.get("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL)))
    has_key = bool(os.environ.get("OPENAI_API_KEY")) or _is_local_base_url(base_url)
    if not model or not base_url:
        message = "请填写模型名和 API 地址"
    elif not has_key:
        message = "请填写 API Key，或使用本地模型服务地址"
    else:
        message = "已填写基础配置，请点击测试连接"
    return {
        "status": "configured" if model and base_url and has_key else "not_configured",
        "model": model,
        "base_url": base_url,
        "has_api_key": has_key,
        "is_local": _is_local_base_url(base_url),
        "message": message,
    }


def _asr_status_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for engine_id, info in ASR_ENGINES.items():
        item = dict(info)
        if engine_id == "jianying":
            sign_service_url = os.environ.get("JIANYING_SIGN_SERVICE_URL", "").strip()
            item["available"] = bool(sign_service_url)
            item["status"] = "ok" if sign_service_url else "manual"
            item["message"] = (
                f"已配置签名服务：{sign_service_url}"
                if sign_service_url
                else "需要配置 JIANYING_SIGN_SERVICE_URL；默认第三方签名服务测试返回 500"
            )
        payload[engine_id] = item
    return payload


def _system_status_payload() -> dict[str, Any]:
    config = _read_web_config()
    ffmpeg = _command_status("ffmpeg", "FFmpeg")
    ffprobe = _command_status("ffprobe", "FFprobe")
    with _job_lock:
        job = dict(_current_job)
    return {
        "service": {
            "status": "ok",
            "message": "Web 后端正在运行",
            "app_url": "http://127.0.0.1:8742/",
            "local_llm_note": "127.0.0.1:1234 通常是本地 LLM API 地址，不是医疗字幕 App 页面。",
        },
        "job": {
            "status": job.get("status", "idle"),
            "run_id": job.get("run_id"),
            "message": job.get("error") or "",
        },
        "dependencies": {
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
            "resolve": _resolve_status(),
        },
        "llm": _llm_config_status(config),
        "asr": _asr_status_payload(),
        "knowledge": _knowledge_status(),
        "storage": {
            "input": _check_writable(INPUT_DIR),
            "work": _check_writable(WORK_DIR),
            "output": _check_writable(OUTPUT_DIR),
            "logs": _check_writable(LOG_DIR),
        },
        "config": config,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }


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
                "aliases": item.get("aliases", []) if isinstance(item.get("aliases"), list) else [],
                "patterns": item.get("patterns", []) if isinstance(item.get("patterns"), list) else [],
                "confidence": item.get("confidence", ""),
                "evidence": str(item.get("evidence", "")),
                "note": str(item.get("note", "")),
            }
        )
    return result


def _count_dropped_terms(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("dropped_replacements", []) if isinstance(data, dict) else []
    return len(rows) if isinstance(rows, list) else 0


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
                "aliases": row.get("aliases", []) if isinstance(row.get("aliases"), list) else [],
                "patterns": row.get("patterns", []) if isinstance(row.get("patterns"), list) else [],
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


def _video_from_log(log_path: Path) -> Path | None:
    video: Path | None = None
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = event.get("data") or {}
                if not isinstance(data, dict):
                    continue
                value = data.get("video")
                if value:
                    candidate = Path(str(value))
                    if candidate.exists():
                        video = candidate
    except OSError:
        return None
    return video


def _run_id_from_srt_name(srt_path: Path) -> str:
    for pattern in (
        r"-(\d{8}-\d{6})\.optimized\.srt$",
        r"\.revised-(\d{8}-\d{6})\.srt$",
        r"-(\d{8}-\d{6}).*\.srt$",
    ):
        match = re.search(pattern, srt_path.name)
        if match:
            return match.group(1)
    return ""


def _find_video_for_run_id(run_id: str) -> Path | None:
    if not run_id:
        return None
    for log_path in sorted(LOG_DIR.glob(f"*{run_id}*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True):
        video = _video_from_log(log_path)
        if video:
            return video
    return None


def _find_video_for_srt(srt_path: Path) -> Path | None:
    run_id = _run_id_from_srt_name(srt_path)
    video = _find_video_for_run_id(run_id)
    if video:
        return video
    for terms_path in sorted(WORK_DIR.glob(f"{srt_path.stem}*.terms.json"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(terms_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source_value = payload.get("source_srt") if isinstance(payload, dict) else None
        if source_value:
            source = _safe_work_path(str(source_value))
            if source and source.exists() and source != srt_path:
                video = _find_video_for_srt(source)
                if video:
                    return video
    return None


def _list_final_srt_candidates(limit: int = 8) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    paths = list(WORK_DIR.glob("*.optimized.srt")) + list(WORK_DIR.glob("*.revised*.srt"))
    for srt_path in sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True):
        if srt_path in seen:
            continue
        seen.add(srt_path)
        run_id = _run_id_from_srt_name(srt_path)
        video = _find_video_for_srt(srt_path)
        candidates.append(
            {
                "run_id": run_id,
                "srt": str(srt_path),
                "srt_name": srt_path.name,
                "video": str(video) if video else None,
                "video_name": video.name if video else None,
                "is_revision": ".revised-" in srt_path.stem,
                "updated_at": datetime.fromtimestamp(srt_path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def _safe_work_path(value: str) -> Path | None:
    try:
        path = Path(value).expanduser().resolve()
        path.relative_to(WORK_DIR.resolve())
    except Exception:
        return None
    return path


def _write_log_event(log_path: Path, step: str, status: str, message: str, **data: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "step": step,
            "status": status,
            "message": message,
            "data": data,
        }, ensure_ascii=False) + "\n")


def _run_pdf_terms_thread(video_path: Path, pdf_path: Path, engine: str,
                          log_path: Path, run_id: str,
                          api_key: str = "", system_prompt: str = "",
                          model: str = "", base_url: str = "",
                          segmented_asr: bool = False,
                          asr_max_workers: int = 1,
                          asr_segment_minutes: float = 10.0,
                          asr_max_segment_minutes: float = 12.0,
                          srt_only: bool = False,
                          render_preset: str = "ffpg-fast-23",
                          remove_fillers: str = DEFAULT_REMOVE_FILLERS,
                          subtitle_min_chars: int = 5,
                          subtitle_max_chars: int = 20) -> None:
    """Generate a PDF-first terminology table, then wait for user confirmation."""
    global _current_job
    try:
        stem = video_path.stem.replace(" ", "_")
        terms_path = WORK_DIR / f"{stem}-{run_id}.terms.preflight.json"
        effective_model = model or os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)
        effective_base_url = _normalize_base_url(base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL))
        effective_key = api_key or os.environ.get("OPENAI_API_KEY") or ("local" if _is_local_base_url(effective_base_url) else "")

        _write_log_event(log_path, "pipeline", "start", "PDF terminology preflight started",
                         video=str(video_path), context=str(pdf_path), run_id=run_id)
        _write_log_event(log_path, "term_map", "running", "extracting terminology from PDF",
                         context=str(pdf_path), model=effective_model, base_url=effective_base_url, output=str(terms_path))

        python_executable = BUNDLED_PYTHON if BUNDLED_PYTHON.exists() else Path(sys.executable)
        cmd = [
            str(python_executable),
            str(ROOT / "term_mapper.py"),
            "--context",
            str(pdf_path),
            "--context-only",
            "--output",
            str(terms_path),
            "--model",
            effective_model,
            "--base-url",
            effective_base_url,
            "--system-prompt",
            system_prompt,
            "--timeout",
            "45",
            "--retries",
            "1",
            "--progress-jsonl",
        ]
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = effective_key
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        stderr_lines: list[str] = []
        assert proc.stderr is not None
        while True:
            line = proc.stderr.readline()
            if line:
                stderr_lines.append(line)
                stripped = line.strip()
                if stripped.startswith("{"):
                    try:
                        event = json.loads(stripped)
                        if isinstance(event, dict):
                            status = str(event.pop("status", "progress"))
                            message = str(event.pop("message", "PDF terminology progress"))
                            _write_log_event(log_path, "term_map", status, message, **event)
                    except json.JSONDecodeError:
                        pass
                continue
            if proc.poll() is not None:
                break
        stdout = proc.stdout.read() if proc.stdout is not None else ""
        return_code = proc.wait()
        stderr_text = "".join(stderr_lines)
        if return_code != 0:
            raise RuntimeError(stderr_text.strip()[-2000:] or stdout.strip()[-2000:] or f"term_mapper exited with {return_code}")
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse term_mapper output: {stdout[-2000:]}") from exc

        with _job_lock:
            _current_job["status"] = "awaiting_terms"
            _current_job["terms_path"] = str(terms_path)
            _current_job["srt_path"] = None
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
                "asr_max_segment_minutes": max(12.0, float(asr_max_segment_minutes or 12.0)),
                "srt_only": bool(srt_only),
                "render_preset": str(render_preset or "ffpg-fast-23").strip() or "ffpg-fast-23",
                "remove_fillers": str(remove_fillers),
                "subtitle_min_chars": max(1, int(subtitle_min_chars or 5)),
                "subtitle_max_chars": max(8, int(subtitle_max_chars or 20)),
            }
        _write_log_event(log_path, "term_review", "review", "PDF terminology candidates ready for review",
                         terms=str(terms_path), replacement_count=len(result.get("replacements", [])))
    except Exception as exc:
        with _job_lock:
            _current_job["status"] = "failed"
            _current_job["error"] = str(exc)
        _write_log_event(log_path, "pipeline", "failed", str(exc))


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
                          asr_max_segment_minutes: float = 12.0,
                          srt_only: bool = False,
                          render_preset: str = "ffpg-fast-23",
                          remove_fillers: str = DEFAULT_REMOVE_FILLERS,
                          subtitle_min_chars: int = 5,
                          subtitle_max_chars: int = 20) -> None:
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
        args.no_asr_boundary_repair = False
        args.asr_boundary_llm_timeout = 30
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
        args.llm_term_review = True
        args.no_llm_term_review = False
        args.llm_term_review_batch_size = 100
        args.llm_term_review_overlap = 5
        args.llm_term_review_timeout = 45
        args.optimize_subtitles = True
        args.subtitle_min_chars = max(1, int(subtitle_min_chars or 5))
        args.subtitle_max_chars = max(args.subtitle_min_chars + 1, int(subtitle_max_chars or 20))
        args.remove_punctuation = "，,"
        args.remove_fillers = str(remove_fillers)
        args.no_subtitle_llm = False
        args.subtitle_llm_timeout = 30
        args.allow_neighbor_rewrite = False
        args.prepare_only = False
        args.srt_only = bool(srt_only)
        args.render_current = False
        args.project_name = None
        args.template_project = str(DEFAULT_TEMPLATE_PROJECT) if DEFAULT_TEMPLATE_PROJECT.exists() else None
        args.use_template_timeline = bool(args.template_project)
        args.subtitle_preset = "sub01"
        args.allow_subtitle_preset_fallback = not args.use_template_timeline
        args.render_type = "x264 8-bit 4:2:0(FFmpeg)"
        args.render_preset = str(render_preset or "ffpg-fast-23").strip() or "ffpg-fast-23"
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
        with _job_lock:
            if _current_job.get("run_id") == run_id and _current_job.get("cancel_requested"):
                _current_job["status"] = "aborted"
                _current_job["output_path"] = None
                return
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
                    "asr_max_segment_minutes": max(12.0, float(asr_max_segment_minutes or 12.0)),
                    "srt_only": bool(srt_only),
                    "render_preset": str(render_preset or "ffpg-fast-23").strip() or "ffpg-fast-23",
                    "remove_fillers": str(remove_fillers),
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


def _run_revision_output_thread(video_path: Path, srt_path: Path, log_path: Path,
                                run_id: str, render_preset: str = "ffpg-fast-23",
                                srt_only: bool = False) -> None:
    """Import an already-final revised SRT with its source video, then optionally render."""
    global _current_job
    try:
        sys.path.insert(0, str(ROOT))
        from mvp_pipeline import run_pipeline  # type: ignore

        class WebArgs:
            pass

        args = WebArgs()
        args.video = str(video_path)
        args.engine = "bcut"
        args.srt = str(srt_path)
        args.segmented_asr = False
        args.asr_max_workers = 1
        args.asr_segment_minutes = 10.0
        args.asr_max_segment_minutes = 12.0
        args.no_asr_boundary_repair = True
        args.asr_boundary_llm_timeout = 30
        args.terms = None
        args.context = None
        args.llm_model = DEFAULT_LLM_MODEL
        args.llm_base_url = DEFAULT_LLM_BASE_URL
        args.llm_api_key = ""
        args.llm_system_prompt = ""
        args.term_map_timeout = 45
        args.term_map_retries = 1
        args.review_terms = False
        args.llm_term_review = False
        args.no_llm_term_review = True
        args.llm_term_review_batch_size = 100
        args.llm_term_review_overlap = 5
        args.llm_term_review_timeout = 45
        args.optimize_subtitles = False
        args.subtitle_min_chars = 5
        args.subtitle_max_chars = 20
        args.remove_punctuation = ""
        args.remove_fillers = ""
        args.no_subtitle_llm = True
        args.subtitle_llm_timeout = 30
        args.allow_neighbor_rewrite = False
        args.prepare_only = False
        args.srt_only = bool(srt_only)
        args.render_current = False
        args.project_name = None
        args.template_project = str(DEFAULT_TEMPLATE_PROJECT) if DEFAULT_TEMPLATE_PROJECT.exists() else None
        args.use_template_timeline = bool(args.template_project)
        args.subtitle_preset = "sub01"
        args.allow_subtitle_preset_fallback = not args.use_template_timeline
        args.render_type = "x264 8-bit 4:2:0(FFmpeg)"
        args.render_preset = str(render_preset or "ffpg-fast-23").strip() or "ffpg-fast-23"
        args.no_render_preset = False
        args.list_resolve_presets = False
        args.allow_render_type_fallback = True
        args.run_id = run_id

        import mvp_pipeline as mp
        saved = mp.DEFAULT_LOG_DIR
        mp.DEFAULT_LOG_DIR = LOG_DIR
        try:
            result = run_pipeline(args)
        finally:
            mp.DEFAULT_LOG_DIR = saved
        with _job_lock:
            _current_job["status"] = "done"
            _current_job["output_path"] = result.get("rendered")
            _current_job["result"] = result
    except Exception as exc:
        with _job_lock:
            _current_job["status"] = "failed"
            _current_job["error"] = str(exc)
        _write_log_event(log_path, "pipeline", "failed", str(exc))


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the frontend page."""
    html_path = WEB_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Web panel not found. Create web/index.html</h1>")


@app.get("/api/system/status")
async def system_status():
    """Return user-facing environment readiness without mutating the system."""
    return _system_status_payload()


@app.get("/api/knowledge/status")
async def knowledge_status():
    """Return read-only status for the cross-course medical knowledge base."""
    return {"knowledge": _knowledge_status()}


@app.get("/api/config")
async def get_config():
    """Return non-sensitive saved Web UI configuration."""
    return {"config": _read_web_config()}


@app.post("/api/config")
async def save_config(payload: dict[str, Any] = Body(...)):
    """Persist non-sensitive Web UI configuration."""
    return {"config": _write_web_config(payload)}


@app.post("/api/llm/test")
async def test_llm(payload: dict[str, Any] = Body(...)):
    """Test OpenAI-compatible LLM settings before a pipeline run."""
    model = str(payload.get("model") or os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)).strip()
    raw_base_url = str(payload.get("base_url") or os.environ.get("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL))
    base_url = _normalize_base_url(raw_base_url)
    try:
        api_key = str(payload.get("api_key") or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key and _is_local_base_url(base_url):
            api_key = "local"
        error = _validate_llm(model, base_url, api_key)
        if error:
            return {
                "status": "failed",
                "ok": False,
                "error": error,
                "model": model,
                "base_url": base_url,
                "is_local": _is_local_base_url(base_url),
            }
        return {
            "status": "ok",
            "ok": True,
            "message": "LLM 连接正常，模型可以使用",
            "model": model,
            "base_url": base_url,
            "is_local": _is_local_base_url(base_url),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "ok": False,
            "error": f"LLM 预检异常：{_sanitize_llm_error(str(exc))}",
            "model": model,
            "base_url": base_url,
            "is_local": _is_local_base_url(base_url),
        }


@app.post("/api/job/cancel")
async def cancel_job():
    """Record a user cancellation request for the active job."""
    with _job_lock:
        if _current_job.get("status") not in {"running", "awaiting_terms"}:
            return {"status": _current_job.get("status", "idle"), "message": "当前没有正在运行的任务"}
        _current_job["cancel_requested"] = True
        _current_job["status"] = "aborted" if _current_job.get("status") == "awaiting_terms" else "cancel_requested"
        log_path = Path(str(_current_job.get("log_path") or ""))
        run_id = _current_job.get("run_id")
    if log_path:
        try:
            _write_log_event(log_path, "pipeline", "aborted", "用户请求取消任务", run_id=run_id)
        except OSError:
            pass
    return {"status": "cancel_requested", "message": "已请求取消任务，已生成的文件会保留"}


@app.post("/api/upload")
async def upload(video: UploadFile = File(...), pdf: Optional[UploadFile] = File(None),
                 engine: str = Form("bcut"), api_key: str = Form(""),
                 system_prompt: str = Form(""), model: str = Form(""),
                 base_url: str = Form(""),
                 segmented_asr: str = Form(""),
                 asr_max_workers: int = Form(1),
                 asr_segment_minutes: float = Form(10.0),
                 srt_only: str = Form(""),
                 render_preset: str = Form("ffpg-fast-23"),
                 remove_fillers: str = Form(DEFAULT_REMOVE_FILLERS),
                 subtitle_min_chars: int = Form(5),
                 subtitle_max_chars: int = Form(20)):
    """Upload video and optional PDF, then start the pipeline."""
    global _current_job

    if video.filename is None:
        return {"error": "请选择视频文件"}
    asr_status = _asr_status_payload()
    if engine not in asr_status:
        return {"error": "ASR 接口参数不正确"}
    if asr_status[engine].get("available") is not True:
        return {"error": asr_status[engine].get("message") or "所选 ASR 接口当前不可用"}
    effective_model = (model or os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)).strip()
    effective_base_url = _normalize_base_url(base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL))
    effective_api_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not (effective_api_key or _is_local_base_url(effective_base_url)):
        return {"error": "请填写 API Key，或先在环境变量里设置 OPENAI_API_KEY"}

    with _job_lock:
        if _current_job.get("status") == "running":
            return {"error": "已有任务正在运行，请等待当前任务结束"}
        if _current_job.get("status") == "awaiting_terms":
            return {"error": "当前任务正在等待术语审核，请先确认或终止后再开始新任务"}

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
    use_srt_only = srt_only in {"1", "true", "on", "yes"}
    safe_asr_max_workers = max(1, int(asr_max_workers or 1))
    safe_asr_segment_minutes = max(1.0, float(asr_segment_minutes or 10.0))
    safe_render_preset = str(render_preset or "ffpg-fast-23").strip() or "ffpg-fast-23"
    safe_remove_fillers = str(remove_fillers)
    safe_subtitle_min_chars = max(1, int(subtitle_min_chars or 5))
    safe_subtitle_max_chars = max(safe_subtitle_min_chars + 1, int(subtitle_max_chars or 20))

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
            "srt_only": use_srt_only,
            "render_preset": safe_render_preset,
            "remove_fillers": safe_remove_fillers,
            "subtitle_min_chars": safe_subtitle_min_chars,
            "subtitle_max_chars": safe_subtitle_max_chars,
        }

    global _active_thread
    if pdf_path:
        thread = threading.Thread(
            target=_run_pdf_terms_thread,
            args=(
                video_path,
                pdf_path,
                engine,
                log_path,
                run_id,
                effective_api_key,
                system_prompt,
                effective_model,
                effective_base_url,
                use_segmented_asr,
                safe_asr_max_workers,
                safe_asr_segment_minutes,
                max(12.0, safe_asr_segment_minutes),
                use_srt_only,
                safe_render_preset,
                safe_remove_fillers,
                safe_subtitle_min_chars,
                safe_subtitle_max_chars,
            ),
            daemon=True,
        )
        _active_thread = thread
        thread.start()
        return {"run_id": run_id, "log_path": str(log_path), "status": "running", "preflight": "terms"}

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
            False,
            use_segmented_asr,
            safe_asr_max_workers,
            safe_asr_segment_minutes,
            max(12.0, safe_asr_segment_minutes),
            use_srt_only,
            safe_render_preset,
            safe_remove_fillers,
            safe_subtitle_min_chars,
            safe_subtitle_max_chars,
        ),
        daemon=True,
    )
    _active_thread = thread
    thread.start()

    return {"run_id": run_id, "log_path": str(log_path), "status": "running"}


@app.get("/api/resumable-runs")
async def resumable_runs():
    """List previous jobs that have an SRT and can continue without re-running ASR."""
    return {"runs": _list_resume_candidates()}


@app.get("/api/final-srts")
async def final_srts():
    """List final optimized SRT files available for quick second-pass revision."""
    return {"srts": _list_final_srt_candidates()}


@app.post("/api/revise-srt")
async def revise_srt(payload: dict[str, Any] = Body(...)):
    """Apply user replacement rules and built-in unit normalization to an existing final SRT."""
    global _active_thread, _current_job
    source = _safe_work_path(str(payload.get("srt") or ""))
    if not source or not source.exists() or source.suffix.lower() != ".srt":
        return {"error": "请选择有效的最终 SRT"}

    rows = payload.get("replacements", [])
    if not isinstance(rows, list):
        return {"error": "替换规则格式不正确"}

    replacements: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        wrong = str(row.get("wrong", "")).strip()
        correct = str(row.get("correct", "")).strip()
        if not wrong or not correct or wrong == correct:
            continue
        replacements.append({"wrong": wrong, "correct": correct, "note": "用户二次修改"})

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    source_base_stem = re.sub(r"(?:\.revised-\d{8}-\d{6})+$", "", source.stem)
    terms_path = WORK_DIR / f"{source_base_stem}.revision-{run_id}.terms.json"
    output = WORK_DIR / f"{source_base_stem}.revised-{run_id}.srt"
    report = LOG_DIR / f"{source_base_stem}.revision-{run_id}.correction-report.json"
    terms_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_srt": str(source),
                "revision_mode": "quick_final_srt",
                "replacements": replacements,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        sys.path.insert(0, str(ROOT))
        from term_corrector import correct_srt  # type: ignore

        result = correct_srt(source, terms_path, output, report)
    except Exception as exc:
        return {"error": f"二次修改失败：{exc}"}

    response = {
        "status": "done",
        "source_srt": str(source),
        "output_srt": str(output),
        "terms": str(terms_path),
        "report": str(report),
        "changed_cue_count": result.get("changed_cue_count"),
        "replacement_count": result.get("replacement_count"),
    }
    if not payload.get("render_after"):
        return response

    video_value = str(payload.get("video") or "").strip()
    video_path = Path(video_value).expanduser().resolve() if video_value else _find_video_for_srt(source)
    if not video_path or not video_path.exists():
        response["render_error"] = "已生成新 SRT，但没有找到原视频，无法自动输出"
        return response

    with _job_lock:
        if _current_job.get("status") == "running":
            response["render_error"] = "已生成新 SRT，但当前已有任务正在运行，暂未启动自动输出"
            return response
        if _current_job.get("status") == "awaiting_terms":
            response["render_error"] = "已生成新 SRT，但当前任务正在等待术语确认，暂未启动自动输出"
            return response

    render_preset = str(payload.get("render_preset") or "ffpg-fast-23").strip() or "ffpg-fast-23"
    srt_only = bool(payload.get("srt_only"))
    output_run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = video_path.stem.replace(" ", "_")
    output_log_path = LOG_DIR / f"{stem}-{output_run_id}.jsonl"
    with _job_lock:
        _current_job = {
            "status": "running",
            "log_path": str(output_log_path),
            "output_path": None,
            "run_id": output_run_id,
            "video": str(video_path),
            "srt_path": str(output),
            "revision_source_srt": str(source),
            "revision_terms": str(terms_path),
            "render_preset": render_preset,
            "srt_only": srt_only,
            "revision_output": True,
        }
    thread = threading.Thread(
        target=_run_revision_output_thread,
        args=(video_path, output, output_log_path, output_run_id, render_preset, srt_only),
        daemon=True,
    )
    _active_thread = thread
    thread.start()
    response.update({
        "status": "running",
        "run_id": output_run_id,
        "log_path": str(output_log_path),
        "video": str(video_path),
        "render_preset": render_preset,
        "srt_only": srt_only,
    })
    return response


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
    render_preset = str(payload.get("render_preset") or "ffpg-fast-23").strip() or "ffpg-fast-23"
    remove_fillers = str(payload.get("remove_fillers") if payload.get("remove_fillers") is not None else DEFAULT_REMOVE_FILLERS)
    subtitle_min_chars = max(1, int(payload.get("subtitle_min_chars") or 5))
    subtitle_max_chars = max(subtitle_min_chars + 1, int(payload.get("subtitle_max_chars") or 20))
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
            "render_preset": render_preset,
            "remove_fillers": remove_fillers,
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
            bool(payload.get("srt_only")),
            render_preset,
            remove_fillers,
            subtitle_min_chars,
            subtitle_max_chars,
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
        "dropped_replacement_count": _count_dropped_terms(terms_path),
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
        srt_value = job.get("srt_path")
        srt_path = Path(str(srt_value)) if srt_value else None
        log_path = Path(str(job.get("log_path") or ""))

    rows = payload.get("replacements", [])
    if not isinstance(rows, list):
        return {"error": "术语确认数据格式不正确"}
    if not terms_path.exists():
        return {"error": "术语候选文件不存在"}
    if srt_path is not None and not srt_path.exists():
        return {"error": "待续跑 SRT 不存在"}

    approved_terms, approved_count = _write_approved_terms(terms_path, rows, run_id)
    knowledge_result = _merge_terms_into_knowledge(
        _load_terms_for_review(approved_terms),
        run_id,
        approved_terms,
    )
    video_path = Path(str(pending.get("video_path") or job.get("video") or ""))
    if not video_path.exists():
        return {"error": "待续跑视频不存在"}

    stem = video_path.stem.replace(" ", "_")
    with _job_lock:
        _current_job["status"] = "running"
        _current_job["approved_terms_path"] = str(approved_terms)
        _current_job["knowledge_result"] = knowledge_result

    _write_log_event(
        log_path,
        "knowledge",
        "updated",
        "stable medical terms merged into cross-course knowledge base",
        **knowledge_result,
    )

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
            bool(pending.get("segmented_asr")) and srt_path is None,
            int(pending.get("asr_max_workers") or 1),
            float(pending.get("asr_segment_minutes") or 10.0),
            float(pending.get("asr_max_segment_minutes") or max(12.0, float(pending.get("asr_segment_minutes") or 10.0))),
            bool(pending.get("srt_only")),
            str(pending.get("render_preset") or "ffpg-fast-23"),
            str(pending.get("remove_fillers") if pending.get("remove_fillers") is not None else DEFAULT_REMOVE_FILLERS),
            int(pending.get("subtitle_min_chars") or 5),
            int(pending.get("subtitle_max_chars") or 20),
        ),
        daemon=True,
    )
    _active_thread = thread
    thread.start()
    return {
        "status": "running",
        "run_id": run_id,
        "approved_terms_path": str(approved_terms),
        "knowledge": knowledge_result,
    }


@app.post("/api/terms/{run_id}/abort")
async def abort_terms(run_id: str):
    """Abort a job while it is waiting for terminology review."""
    global _current_job
    with _job_lock:
        job = dict(_current_job)
        if job.get("run_id") != run_id:
            return {"error": "没有找到对应任务"}
        if job.get("status") != "awaiting_terms":
            return {"error": "当前任务不在术语审核阶段"}
        log_path = Path(str(job.get("log_path") or ""))
        terms_path = job.get("terms_path")
        srt_path = job.get("srt_path")
        _current_job["status"] = "aborted"
        _current_job["output_path"] = None

    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "step": "term_review",
                    "status": "aborted",
                    "message": "用户终止术语审核，流程已停止",
                    "data": {"terms": terms_path, "srt": srt_path},
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass

    return {"status": "aborted", "run_id": run_id, "terms_path": terms_path, "srt_path": srt_path}


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
