#!/usr/bin/env python3
"""MVP: video -> online ASR -> Resolve import -> rendered output.

This script intentionally keeps the first MVP narrow: prove whether an external
SRT can be produced by online ASR, placed on a Resolve timeline, and burned into
a render without manual clicks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_TOOL_DIR = ROOT / "tool"
DEFAULT_WORK_DIR = ROOT / "work"
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_LOG_DIR = ROOT / "logs"
BUNDLED_PYTHON = Path(
    "/Users/x/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
)

RESOLVE_MODULES = Path(
    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"
)
RESOLVE_SCRIPT_API = Path(
    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
)
RESOLVE_SCRIPT_LIB = Path(
    "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
)

DEFAULT_SUBTITLE_PRESET = "sub01"
DEFAULT_RENDER_TYPE = "x264 8-bit 4:2:0(FFmpeg)"
DEFAULT_RENDER_PRESET = "ffpg-fast-23"
RENDER_TYPE_KEYS = (
    "Type",
    "CodecType",
    "VideoCodecType",
    "RenderType",
    "Encoder",
    "VideoEncoder",
    "H264EncodingType",
    "EncodingProfile",
)


@dataclass
class VideoInfo:
    width: int | None
    height: int | None
    fps: float | None
    duration: float | None


class PipelineError(RuntimeError):
    pass


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, step: str, status: str, message: str, **data: Any) -> None:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "step": step,
            "status": status,
            "message": message,
            "data": data,
        }
        line = json.dumps(payload, ensure_ascii=False)
        print(f"[{step}] {status}: {message}", flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def run_cmd(args: list[str], logger: Logger, step: str) -> subprocess.CompletedProcess[str]:
    logger.event(step, "running", " ".join(args))
    proc = subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        logger.event(step, "failed", proc.stderr.strip() or proc.stdout.strip(), returncode=proc.returncode)
        raise PipelineError(f"{step} failed: {' '.join(args)}")
    if proc.stdout.strip():
        logger.event(step, "stdout", proc.stdout.strip()[-1000:])
    return proc


def require_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise PipelineError(f"Missing required command: {name}")
    return found


def stem_for(video: Path) -> str:
    return video.stem.replace(" ", "_")


def probe_video(video: Path, logger: Logger) -> VideoInfo:
    ffprobe = require_tool("ffprobe")
    proc = run_cmd(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        logger,
        "probe_video",
    )
    data = json.loads(proc.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fps_text = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/0"
    fps = None
    if "/" in fps_text:
        num, den = fps_text.split("/", 1)
        if float(den or 0):
            fps = float(num) / float(den)
    duration = data.get("format", {}).get("duration")
    info = VideoInfo(
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        fps=fps,
        duration=float(duration) if duration is not None else None,
    )
    logger.event("probe_video", "done", "video metadata read", **info.__dict__)
    return info


def probe_rendered_file(video: Path, logger: Logger) -> dict[str, Any]:
    ffprobe = require_tool("ffprobe")
    proc = run_cmd(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=format_name,duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,avg_frame_rate,bit_rate",
            "-of",
            "json",
            str(video),
        ],
        logger,
        "probe_rendered",
    )
    data = json.loads(proc.stdout)
    logger.event("probe_rendered", "done", "rendered file metadata read", file=str(video), metadata=data)
    return data


def extract_audio(video: Path, audio_path: Path, logger: Logger) -> Path:
    ffmpeg = require_tool("ffmpeg")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            str(audio_path),
        ],
        logger,
        "extract_audio",
    )
    logger.event("extract_audio", "done", "audio extracted", audio=str(audio_path))
    return audio_path


def run_asr(audio_path: Path, srt_path: Path, engine: str, logger: Logger) -> Path:
    sys.path.insert(0, str(DEFAULT_TOOL_DIR))
    try:
        import online_asr  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise PipelineError(f"Failed to import online_asr from {DEFAULT_TOOL_DIR}: {exc}") from exc

    def progress(percent: int, message: str) -> None:
        logger.event("asr", "progress", message, percent=percent)

    logger.event("asr", "running", f"starting {engine} ASR", audio=str(audio_path))
    if engine == "bcut":
        result = online_asr.BcutASR(str(audio_path)).run(callback=progress)
    elif engine == "jianying":
        sign_service_url = os.environ.get("JIANYING_SIGN_SERVICE_URL", "").strip()
        if sign_service_url:
            online_asr.JianYingASR.SIGN_SERVICE_URL = sign_service_url
        result = online_asr.JianYingASR(str(audio_path)).run(callback=progress)
    else:
        raise PipelineError(f"Unknown ASR engine: {engine}")
    if not result.has_data():
        raise PipelineError("ASR returned no subtitle segments")
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(srt_path), fmt="srt")
    logger.event("asr", "done", "SRT generated", srt=str(srt_path), cue_count=len(result))
    return srt_path


def run_segmented_online_asr(video: Path, srt_path: Path, engine: str, args: argparse.Namespace, logger: Logger) -> Path:
    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from drautocut.pipeline.long_asr import run_online_long_video_asr
    except Exception as exc:
        raise PipelineError(f"Failed to import segmented ASR pipeline: {exc}") from exc

    def progress(event: dict[str, Any]) -> None:
        payload = dict(event)
        step = str(payload.pop("step", "asr"))
        status = str(payload.pop("status", "progress"))
        message = str(payload.pop("message", ""))
        logger.event(step, status, message, **payload)

    max_workers = max(1, int(getattr(args, "asr_max_workers", 1) or 1))
    target_minutes = max(1.0, float(getattr(args, "asr_segment_minutes", 10.0) or 10.0))
    max_minutes = max(target_minutes, float(getattr(args, "asr_max_segment_minutes", 12.0) or 12.0))
    logger.event(
        "asr_prepare",
        "running",
        "segmented online ASR enabled",
        engine=engine,
        max_workers=max_workers,
        target_segment_minutes=target_minutes,
        max_segment_minutes=max_minutes,
    )
    sign_service_url = (
        getattr(args, "jianying_sign_service_url", "") or os.environ.get("JIANYING_SIGN_SERVICE_URL", "")
    ).strip()
    try:
        result = run_online_long_video_asr(
            video_path=video,
            output_srt=srt_path,
            work_dir=DEFAULT_WORK_DIR,
            tool_dir=DEFAULT_TOOL_DIR,
            engine=engine,
            max_workers=max_workers,
            target_segment_ms=int(target_minutes * 60 * 1000),
            max_segment_ms=int(max_minutes * 60 * 1000),
            jianying_sign_service_url=sign_service_url,
            progress=progress,
        )
    except Exception as exc:
        failures = getattr(exc, "failures", None)
        logger.event(
            "asr",
            "failed",
            "segmented ASR failed",
            error=str(exc),
            failures=failures if isinstance(failures, list) else None,
        )
        raise
    logger.event(
        "asr",
        "done",
        "segmented SRT generated",
        srt=str(result.srt_path),
        cue_count=len(result.cues),
        segment_count=len(result.segments),
    )
    return result.srt_path


def correct_srt_with_terms(srt: Path, terms: Path, base: str, logger: Logger) -> tuple[Path, Path, dict[str, Any]]:
    try:
        from term_corrector import correct_srt  # type: ignore
    except Exception as exc:
        raise PipelineError(f"Failed to import term_corrector: {exc}") from exc
    output = DEFAULT_WORK_DIR / f"{base}.corrected.srt"
    report = DEFAULT_LOG_DIR / f"{base}.correction-report.json"
    logger.event("term_correct", "running", "applying terminology map", srt=str(srt), terms=str(terms))
    result = correct_srt(srt, terms, output, report)
    logger.event(
        "term_correct",
        "done",
        "terminology correction complete",
        output_srt=str(output),
        report=str(report),
        changed_cue_count=result.get("changed_cue_count"),
        replacement_count=result.get("replacement_count"),
    )
    return output, report, result


def optimize_subtitles(srt: Path, base: str, args: argparse.Namespace, logger: Logger) -> tuple[Path, Path, dict[str, Any]]:
    try:
        from subtitle_optimizer import optimize_srt  # type: ignore
    except Exception as exc:
        raise PipelineError(f"Failed to import subtitle_optimizer: {exc}") from exc
    api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
    output = DEFAULT_WORK_DIR / f"{base}.optimized.srt"
    report = DEFAULT_LOG_DIR / f"{base}.subtitle-optimization-report.json"
    logger.event(
        "subtitle_optimize",
        "running",
        "cleaning punctuation and optimizing overlong subtitles",
        srt=str(srt),
        output=str(output),
        max_chars=args.subtitle_max_chars,
        remove_punctuation=args.remove_punctuation,
        model=args.llm_model,
        base_url=args.llm_base_url,
        use_llm=not args.no_subtitle_llm,
    )

    def log_progress(payload: dict[str, Any]) -> None:
        data = dict(payload)
        status = str(data.pop("status", "progress"))
        message = str(data.pop("message", "subtitle optimization progress"))
        logger.event("subtitle_optimize", status, message, **data)

    result = optimize_srt(
        srt=srt,
        output=output,
        report_path=report,
        max_chars=args.subtitle_max_chars,
        punctuation=args.remove_punctuation,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=api_key,
        use_llm=not args.no_subtitle_llm,
        allow_neighbor_rewrite=args.allow_neighbor_rewrite,
        llm_timeout=args.subtitle_llm_timeout,
        progress_callback=log_progress,
    )
    logger.event(
        "subtitle_optimize",
        "done",
        "subtitle optimization complete",
        output_srt=str(output),
        report=str(report),
        punctuation_changed_cue_count=result.get("punctuation_changed_cue_count"),
        overlong_detected_count=result.get("overlong_detected_count"),
        overlong_changed_cue_count=result.get("overlong_changed_cue_count"),
    )
    return output, report, result


def generate_terms_from_context(context: Path, srt: Path, base: str, args: argparse.Namespace, logger: Logger) -> Path:
    api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise PipelineError("Missing LLM API key. Set OPENAI_API_KEY or pass --llm-api-key.")
    output = DEFAULT_WORK_DIR / f"{base}.terms.generated.json"
    python = BUNDLED_PYTHON if BUNDLED_PYTHON.exists() else Path(sys.executable)
    term_mapper = ROOT / "term_mapper.py"
    logger.event(
        "term_map",
        "running",
        "generating terminology map from course context",
        context=str(context),
        srt=str(srt),
        model=args.llm_model,
        base_url=args.llm_base_url,
        output=str(output),
    )
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    cmd = [
        str(python),
        str(term_mapper),
        "--context",
        str(context),
        "--srt",
        str(srt),
        "--output",
        str(output),
        "--model",
        args.llm_model,
        "--base-url",
        args.llm_base_url,
        "--api-key",
        api_key,
        "--progress-jsonl",
        "--timeout",
        str(getattr(args, "term_map_timeout", 45)),
        "--retries",
        str(getattr(args, "term_map_retries", 1)),
    ]
    if getattr(args, "llm_system_prompt", ""):
        cmd.extend(["--system-prompt", args.llm_system_prompt])
    proc = subprocess.Popen(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    stderr_lines: list[str] = []
    assert proc.stderr is not None
    for line in proc.stderr:
        stripped = line.strip()
        if not stripped:
            continue
        stderr_lines.append(stripped)
        try:
            progress = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(progress, dict):
            status = str(progress.pop("status", "progress"))
            message = str(progress.pop("message", "terminology map progress"))
            logger.event("term_map", status, message, **progress)
    stdout, stderr_remainder = proc.communicate()
    if stderr_remainder:
        stderr_lines.extend(line for line in stderr_remainder.splitlines() if line.strip())
    if proc.returncode != 0:
        stderr_text = "\n".join(stderr_lines)
        logger.event("term_map", "failed", stderr_text.strip()[-2000:] or stdout.strip()[-2000:])
        raise PipelineError("Terminology map generation failed")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        result = json.loads(output.read_text(encoding="utf-8"))
    logger.event(
        "term_map",
        "done",
        "terminology map generated",
        terms=str(output),
        replacement_count=len(result.get("replacements", [])),
    )
    return output


def count_srt_cues(path: Path) -> int:
    text = path.read_text(encoding="utf-8-sig")
    return sum(1 for block in text.replace("\r\n", "\n").split("\n\n") if "-->" in block)


def setup_resolve_import_path() -> None:
    os.environ.setdefault("RESOLVE_SCRIPT_API", str(RESOLVE_SCRIPT_API))
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", str(RESOLVE_SCRIPT_LIB))
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    modules_text = str(RESOLVE_MODULES)
    if modules_text not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [current_pythonpath, modules_text]))
    if modules_text not in sys.path:
        sys.path.insert(0, modules_text)


def connect_resolve(logger: Logger):
    setup_resolve_import_path()
    try:
        import DaVinciResolveScript as dvr_script  # type: ignore
    except Exception as exc:
        raise PipelineError(
            "Cannot import DaVinciResolveScript. Check Resolve scripting install paths."
        ) from exc
    resolve = dvr_script.scriptapp("Resolve")
    if not resolve:
        raise PipelineError("Cannot connect to Resolve. Open DaVinci Resolve Studio and enable External scripting: Local.")
    logger.event(
        "resolve",
        "connected",
        "connected to DaVinci Resolve",
        product=resolve.GetProductName(),
        version=resolve.GetVersionString(),
    )
    return resolve


def create_or_load_project(resolve, project_name: str, logger: Logger):
    pm = resolve.GetProjectManager()
    project = pm.LoadProject(project_name)
    if project:
        logger.event("resolve_project", "loaded", "existing project loaded", project=project_name)
        return project
    project = pm.CreateProject(project_name)
    if not project:
        raise PipelineError(f"Failed to create Resolve project: {project_name}")
    logger.event("resolve_project", "created", "new project created", project=project_name)
    return project


def import_template_project(resolve, template_path: Path, project_name: str, logger: Logger):
    if not template_path.exists():
        raise PipelineError(f"Resolve template project not found: {template_path}")
    pm = resolve.GetProjectManager()
    existing = pm.LoadProject(project_name)
    if existing:
        logger.event("resolve_project", "loaded", "existing imported template project loaded", project=project_name)
        return existing
    ok = bool(pm.ImportProject(str(template_path), project_name))
    logger.event(
        "resolve_project",
        "template_imported" if ok else "template_import_failed",
        "template project import attempted",
        template=str(template_path),
        project=project_name,
    )
    if not ok:
        raise PipelineError(f"Resolve failed to import template project: {template_path}")
    project = pm.LoadProject(project_name)
    if not project:
        raise PipelineError(f"Resolve imported template but could not load project: {project_name}")
    logger.event("resolve_project", "loaded", "imported template project loaded", project=project_name)
    return project


def create_pipeline_project(resolve, project_name: str, template_path: Path | None, logger: Logger):
    if template_path:
        return import_template_project(resolve, template_path, project_name, logger)
    return create_or_load_project(resolve, project_name, logger)


def project_for_listing(resolve, project_name: str | None, fallback_name: str, logger: Logger):
    if project_name:
        return create_or_load_project(resolve, project_name, logger)
    pm = resolve.GetProjectManager()
    project = pm.GetCurrentProject()
    if project:
        logger.event("resolve_project", "current", "using current project", project=project.GetName())
        return project
    return create_or_load_project(resolve, fallback_name, logger)


def set_project_timing(project, info: VideoInfo, logger: Logger) -> None:
    if info.fps:
        fps_text = str(round(info.fps, 3)).rstrip("0").rstrip(".")
        for key in ("timelineFrameRate", "videoFrameRate"):
            try:
                project.SetSetting(key, fps_text)
            except Exception:
                pass
        logger.event("resolve_project", "timing", "requested project frame rate", fps=fps_text)


def import_video_create_timeline(project, video: Path, timeline_name: str, logger: Logger):
    media_pool = project.GetMediaPool()
    imported = media_pool.ImportMedia([str(video)])
    if not imported:
        raise PipelineError(f"Resolve failed to import video: {video}")
    clip = imported[0]
    logger.event("resolve_import_video", "done", "video imported", clip=clip.GetName())
    timeline = media_pool.CreateTimelineFromClips(timeline_name, [clip])
    if not timeline:
        raise PipelineError("Resolve failed to create timeline from video")
    project.SetCurrentTimeline(timeline)
    logger.event("resolve_timeline", "created", "timeline created", timeline=timeline.GetName())
    return media_pool, timeline


def timeline_items(timeline, track_type: str) -> list[Any]:
    items: list[Any] = []
    for track_index in range(1, int(timeline.GetTrackCount(track_type) or 0) + 1):
        items.extend(timeline.GetItemListInTrack(track_type, track_index) or [])
    return items


def clear_template_timeline_items(timeline, logger: Logger) -> None:
    delete_items: list[Any] = []
    counts: dict[str, int] = {}
    for track_type in ("video", "audio", "subtitle"):
        items = timeline_items(timeline, track_type)
        counts[track_type] = len(items)
        delete_items.extend(items)
    if not delete_items:
        logger.event("resolve_timeline", "template_clear_skipped", "template timeline has no placeholder items")
        return
    ok = bool(timeline.DeleteClips(delete_items, False))
    logger.event(
        "resolve_timeline",
        "template_cleared" if ok else "template_clear_failed",
        "placeholder timeline items removed while preserving tracks",
        counts=counts,
        deleted_count=len(delete_items),
    )
    if not ok:
        raise PipelineError("Resolve failed to clear placeholder items from the template timeline.")


def import_video_into_template_timeline(
    project,
    video: Path,
    info: VideoInfo,
    timeline_name: str,
    logger: Logger,
):
    media_pool = project.GetMediaPool()
    timeline = project.GetCurrentTimeline() or project.GetTimelineByIndex(1)
    if not timeline:
        raise PipelineError("Template project has no timeline to reuse.")
    project.SetCurrentTimeline(timeline)
    try:
        timeline.SetName(timeline_name)
    except Exception:
        logger.event("resolve_timeline", "rename_skipped", "template timeline rename is unavailable")

    logger.event(
        "resolve_timeline",
        "template_reuse",
        "reusing template timeline so subtitle track style is preserved",
        timeline=timeline.GetName(),
        video_tracks=timeline.GetTrackCount("video"),
        audio_tracks=timeline.GetTrackCount("audio"),
        subtitle_tracks=timeline.GetTrackCount("subtitle"),
    )
    if int(timeline.GetTrackCount("subtitle") or 0) < 1:
        raise PipelineError("Template timeline has no subtitle track; it cannot carry the saved subtitle style.")

    clear_template_timeline_items(timeline, logger)

    imported = media_pool.ImportMedia([str(video)])
    if not imported:
        raise PipelineError(f"Resolve failed to import video: {video}")
    clip = imported[0]
    logger.event("resolve_import_video", "done", "video imported", clip=clip.GetName())

    append_payload: list[Any] | list[dict[str, Any]]
    if info.duration and info.fps:
        append_payload = [
            {
                "mediaPoolItem": clip,
                "startFrame": 0,
                "endFrame": int(round(info.duration * info.fps)),
                "recordFrame": 0,
                "trackIndex": 1,
            }
        ]
    else:
        append_payload = [clip]
    appended = media_pool.AppendToTimeline(append_payload)
    if not appended:
        raise PipelineError("Resolve failed to append video to the template timeline.")
    logger.event(
        "resolve_timeline",
        "template_video_appended",
        "video appended to reused template timeline",
        appended_count=len(appended or []),
    )
    return media_pool, timeline


def timeline_item_count(timeline, track_type: str) -> int:
    total = 0
    for index in range(1, int(timeline.GetTrackCount(track_type) or 0) + 1):
        total += len(timeline.GetItemListInTrack(track_type, index) or [])
    return total


def try_import_srt(media_pool, timeline, srt: Path, logger: Logger) -> bool:
    attempts: list[dict[str, Any]] = []

    before_count = timeline_item_count(timeline, "subtitle")
    if int(timeline.GetTrackCount("subtitle") or 0) < 1:
        try:
            ok = bool(timeline.AddTrack("subtitle"))
            attempts.append({"strategy": "AddTrack(subtitle)", "success": ok})
        except Exception as exc:
            attempts.append({"strategy": "AddTrack(subtitle)", "success": False, "error": repr(exc)})

    try:
        imported = media_pool.ImportMedia([str(srt)])
        attempts.append({"strategy": "MediaPool.ImportMedia(srt)", "success": bool(imported), "count": len(imported or [])})
        if imported:
            appended = media_pool.AppendToTimeline(imported)
            after_count = timeline_item_count(timeline, "subtitle")
            attempts.append(
                {
                    "strategy": "MediaPool.AppendToTimeline(imported_srt)",
                    "success": bool(appended) or after_count > before_count,
                    "appended_count": len(appended or []),
                    "subtitle_items_before": before_count,
                    "subtitle_items_after": after_count,
                }
            )
            if after_count > before_count:
                logger.event("resolve_import_srt", "done", "SRT imported via MediaPool", attempts=attempts)
                return True
    except Exception as exc:
        attempts.append({"strategy": "ImportMedia/AppendToTimeline", "success": False, "error": repr(exc)})

    before_count = timeline_item_count(timeline, "subtitle")
    try:
        ok = bool(timeline.ImportIntoTimeline(str(srt), {}))
        after_count = timeline_item_count(timeline, "subtitle")
        attempts.append(
            {
                "strategy": "Timeline.ImportIntoTimeline(srt)",
                "success": ok or after_count > before_count,
                "api_return": ok,
                "subtitle_items_before": before_count,
                "subtitle_items_after": after_count,
            }
        )
        if ok or after_count > before_count:
            logger.event("resolve_import_srt", "done", "SRT imported via timeline import", attempts=attempts)
            return True
    except Exception as exc:
        attempts.append({"strategy": "Timeline.ImportIntoTimeline(srt)", "success": False, "error": repr(exc)})

    logger.event("resolve_import_srt", "failed", "all SRT import strategies failed", attempts=attempts)
    return False


def apply_subtitle_preset(
    project,
    timeline,
    preset_name: str | None,
    logger: Logger,
    allow_fallback: bool = False,
) -> None:
    if not preset_name:
        logger.event("resolve_subtitle_preset", "skipped", "no subtitle preset requested")
        return

    # Resolve scripting API does not expose subtitle track style settings.
    # The user should set the track style manually in Resolve.
    logger.event(
        "resolve_subtitle_preset",
        "skipped",
        f"subtitle preset {preset_name!r} not scriptable — set track style manually in Resolve",
    )
    return

    attempts: list[dict[str, Any]] = []
    success = False
    if hasattr(project, "LoadBurnInPreset"):
        try:
            ok = bool(project.LoadBurnInPreset(preset_name))
            attempts.append({"target": "project", "method": "LoadBurnInPreset", "success": ok})
            success = success or ok
        except Exception as exc:
            attempts.append({"target": "project", "method": "LoadBurnInPreset", "success": False, "error": repr(exc)})

    items = timeline_items(timeline, "subtitle")
    probe_items = items[:1]
    working_item_methods: list[tuple[str, str | None]] = []
    for item_index, item in enumerate(probe_items):
        if hasattr(item, "LoadBurnInPreset"):
            try:
                ok = bool(item.LoadBurnInPreset(preset_name))
                attempts.append(
                    {
                        "target": "subtitle_item",
                        "index": item_index,
                        "method": "LoadBurnInPreset",
                        "success": ok,
                    }
                )
                success = success or ok
                if ok:
                    working_item_methods.append(("LoadBurnInPreset", None))
            except Exception as exc:
                attempts.append(
                    {
                        "target": "subtitle_item",
                        "index": item_index,
                        "method": "LoadBurnInPreset",
                        "success": False,
                        "error": repr(exc),
                    }
                )

        for key in ("CaptionPreset", "SubtitlePreset", "Preset", "StylePreset"):
            try:
                ok = bool(item.SetProperty(key, preset_name))
                attempts.append(
                    {
                        "target": "subtitle_item",
                        "index": item_index,
                        "method": "SetProperty",
                        "key": key,
                        "success": ok,
                    }
                )
                success = success or ok
                if ok:
                    working_item_methods.append(("SetProperty", key))
            except Exception as exc:
                attempts.append(
                    {
                        "target": "subtitle_item",
                        "index": item_index,
                        "method": "SetProperty",
                        "key": key,
                        "success": False,
                        "error": repr(exc),
                    }
                )

    applied_count = len(probe_items)
    if working_item_methods and len(items) > len(probe_items):
        for item in items[len(probe_items):]:
            for method, key in working_item_methods:
                try:
                    if method == "LoadBurnInPreset":
                        item.LoadBurnInPreset(preset_name)
                    elif method == "SetProperty" and key:
                        item.SetProperty(key, preset_name)
                    applied_count += 1
                except Exception:
                    pass

    logger.event(
        "resolve_subtitle_preset",
        "done" if success else "failed",
        f"subtitle preset {preset_name!r} application attempted",
        preset=preset_name,
        subtitle_item_count=len(items),
        applied_count=applied_count if success else 0,
        attempts=attempts,
    )
    if not success:
        if allow_fallback:
            logger.event(
                "resolve_subtitle_preset",
                "fallback",
                "continuing without verified subtitle preset",
                preset=preset_name,
            )
            return
        raise PipelineError(
            f"Resolve did not accept subtitle preset {preset_name!r} through the available scripting paths."
        )


def validate_timeline_bounds(timeline, info: VideoInfo, logger: Logger) -> None:
    if not info.duration or not info.fps:
        logger.event("resolve_timeline", "warn", "cannot validate timeline duration without source duration/fps")
        return
    expected_frames = int(round(info.duration * info.fps))
    start = int(timeline.GetStartFrame() or 0)
    end = int(timeline.GetEndFrame() or 0)
    actual_frames = max(0, end - start)
    tolerance = max(25, int(round(info.fps * 2)))
    logger.event(
        "resolve_timeline",
        "bounds",
        "timeline duration checked",
        expected_frames=expected_frames,
        actual_frames=actual_frames,
        start_frame=start,
        end_frame=end,
        tolerance=tolerance,
    )
    if actual_frames > expected_frames + tolerance:
        raise PipelineError(
            "Resolve timeline is longer than the source after SRT import. "
            "This usually means the SRT was appended after the video instead of aligned at 0. "
            f"Expected about {expected_frames} frames, got {actual_frames}."
        )


def choose_format_codec(project, logger: Logger) -> tuple[str | None, str | None]:
    formats = project.GetRenderFormats() or {}
    chosen_format = None
    for label, value in formats.items():
        text = f"{label} {value}".lower()
        if "mp4" in text:
            chosen_format = value if str(value).lower() == "mp4" else label
            break
    if not chosen_format:
        logger.event("resolve_render", "warn", "MP4 render format not found; keeping current format")
        return None, None

    codecs = project.GetRenderCodecs(chosen_format) or {}
    chosen_codec = None
    for label, value in codecs.items():
        text = f"{label} {value}".lower().replace(".", "")
        if "h264" in text or "h 264" in text:
            chosen_codec = value
            break
    if not chosen_codec:
        logger.event("resolve_render", "warn", "H.264 codec not found; keeping current codec", format=chosen_format)
        return chosen_format, None
    logger.event("resolve_render", "codec", "selected render codec", format=chosen_format, codec=chosen_codec)
    return chosen_format, chosen_codec


def get_render_settings(project) -> dict[str, Any]:
    if not hasattr(project, "GetRenderSettings"):
        return {}
    try:
        settings = project.GetRenderSettings()
    except Exception:
        return {}
    return settings if isinstance(settings, dict) else {}


def render_type_matches(settings: dict[str, Any], render_type: str) -> bool:
    if not settings:
        return False
    wanted = render_type.strip().lower()
    for value in settings.values():
        if isinstance(value, str) and value.strip().lower() == wanted:
            return True
    return False


def apply_render_type(
    project,
    render_type: str | None,
    logger: Logger,
    allow_fallback: bool = False,
) -> None:
    if not render_type:
        logger.event("resolve_render", "type_skipped", "no render type requested")
        return

    before = get_render_settings(project)
    attempts: list[dict[str, Any]] = []
    verified = False
    accepted = False
    for key in RENDER_TYPE_KEYS:
        try:
            ok = bool(project.SetRenderSettings({key: render_type}))
        except Exception as exc:
            attempts.append({"key": key, "success": False, "error": repr(exc)})
            continue
        after = get_render_settings(project)
        key_verified = after.get(key) == render_type or render_type_matches(after, render_type)
        attempts.append({"key": key, "success": ok, "verified": key_verified, "readback": after.get(key)})
        accepted = accepted or ok
        verified = verified or key_verified
        if key_verified:
            break

    logger.event(
        "resolve_render",
        "type_set" if verified else "type_unverified",
        f"render type requested: {render_type}",
        render_type=render_type,
        attempts=attempts,
        before=before,
        after=get_render_settings(project),
    )
    if not accepted:
        if allow_fallback:
            logger.event(
                "resolve_render",
                "type_fallback",
                "continuing without verified render type",
                render_type=render_type,
            )
            return
        raise PipelineError(f"Resolve rejected render type {render_type!r} for all tested setting keys.")


def list_render_presets(project) -> list[str]:
    if not hasattr(project, "GetRenderPresetList"):
        return []
    try:
        presets = project.GetRenderPresetList() or []
    except Exception:
        return []
    return [str(preset) for preset in presets]


def apply_render_preset(project, preset_name: str | None, logger: Logger) -> bool:
    if not preset_name:
        return False
    try:
        ok = bool(project.LoadRenderPreset(preset_name))
    except Exception as exc:
        logger.event(
            "resolve_render",
            "preset_failed",
            f"render preset {preset_name!r} raised an error",
            preset=preset_name,
            error=repr(exc),
            available_presets=list_render_presets(project),
        )
        raise PipelineError(f"Resolve failed while loading render preset {preset_name!r}: {exc}") from exc
    logger.event(
        "resolve_render",
        "preset_loaded" if ok else "preset_missing",
        f"render preset {preset_name!r} load attempted",
        preset=preset_name,
        available_presets=list_render_presets(project),
    )
    if not ok:
        raise PipelineError(
            f"Resolve could not load render preset {preset_name!r}. "
            "Run with --list-resolve-presets to see available preset names."
        )
    return True


def render_timeline(
    project,
    output_dir: Path,
    output_name: str,
    info: VideoInfo,
    logger: Logger,
    render_preset: str | None,
    render_type: str | None,
    allow_render_type_fallback: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    preset_loaded = apply_render_preset(project, render_preset, logger)
    if preset_loaded:
        settings: dict[str, Any] = {
            "TargetDir": str(output_dir),
            "CustomName": output_name,
            "SelectAllFrames": True,
            "ExportSubtitle": True,
            "SubtitleFormat": "BurnIn",
        }
        logger.event(
            "resolve_render",
            "preset_preserved",
            "using render preset settings; only output path/name/range/subtitles are overridden",
            render_preset=render_preset,
        )
    else:
        fmt, codec = choose_format_codec(project, logger)
        if fmt and codec:
            ok = project.SetCurrentRenderFormatAndCodec(fmt, codec)
            logger.event(
                "resolve_render",
                "format_set",
                "render format/codec requested",
                success=bool(ok),
                format=fmt,
                codec=codec,
            )
        settings = {
            "TargetDir": str(output_dir),
            "CustomName": output_name,
            "SelectAllFrames": True,
            "ExportVideo": True,
            "ExportAudio": True,
            "ExportSubtitle": True,
            "SubtitleFormat": "BurnIn",
        }
        if info.width:
            settings["FormatWidth"] = int(info.width)
        if info.height:
            settings["FormatHeight"] = int(info.height)
        if info.fps:
            settings["FrameRate"] = float(info.fps)

    if not project.SetRenderSettings(settings):
        raise PipelineError(f"Resolve failed to apply render settings: {settings}")
    if preset_loaded:
        logger.event(
            "resolve_render",
            "type_skipped",
            "render type is expected to come from the loaded render preset",
            render_preset=render_preset,
        )
    else:
        apply_render_type(project, render_type, logger, allow_fallback=allow_render_type_fallback)
    job_id = project.AddRenderJob()
    if not job_id:
        raise PipelineError("Resolve failed to add render job")
    logger.event("resolve_render", "queued", "render job queued", job_id=job_id, settings=settings)
    if not project.StartRendering([job_id], False):
        raise PipelineError(f"Resolve failed to start render job: {job_id}")
    logger.event("resolve_render", "running", "render started", job_id=job_id)

    while project.IsRenderingInProgress():
        render_status = project.GetRenderJobStatus(job_id)
        logger.event("resolve_render", "progress", "rendering", render_status=render_status)
        time.sleep(5)

    render_status = project.GetRenderJobStatus(job_id)
    logger.event("resolve_render", "done", "render finished", render_status=render_status)
    candidates = sorted(output_dir.glob(output_name + "*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise PipelineError(f"Render finished but no output file matching {output_name}* was found in {output_dir}")
    return candidates[0]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.render_current:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        run_id = getattr(args, "run_id", None) or datetime.now().strftime("%Y%m%d-%H%M%S")
        logger = Logger(DEFAULT_LOG_DIR / f"render-current-{run_id}.jsonl")
        logger.event("pipeline", "start", "rendering current Resolve timeline")
        resolve = connect_resolve(logger)
        project = resolve.GetProjectManager().GetCurrentProject()
        if not project:
            raise PipelineError("Resolve has no current project to render")
        timeline = project.GetCurrentTimeline()
        if not timeline:
            raise PipelineError("Resolve has no current timeline to render")
        render_preset = None if args.no_render_preset else args.render_preset
        rendered = render_timeline(
            project,
            DEFAULT_OUTPUT_DIR.resolve(),
            f"{timeline.GetName()}_styled_{run_id}",
            VideoInfo(None, None, None, None),
            logger,
            render_preset,
            args.render_type,
            args.allow_render_type_fallback,
        )
        rendered_metadata = probe_rendered_file(rendered, logger)
        result = {
            "project": project.GetName(),
            "timeline": timeline.GetName(),
            "rendered": str(rendered),
            "rendered_metadata": rendered_metadata,
            "log": str(logger.path),
        }
        logger.event("pipeline", "complete", "current Resolve timeline rendered", **result)
        return result

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise PipelineError(f"Video not found: {video}")

    DEFAULT_WORK_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_id = getattr(args, "run_id", None) or datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stem_for(video)}-{run_id}"
    logger = Logger(DEFAULT_LOG_DIR / f"{base}.jsonl")
    logger.event("pipeline", "start", "MVP pipeline started", video=str(video), engine=args.engine)
    render_preset = None if args.no_render_preset else args.render_preset
    template_path = Path(args.template_project).expanduser().resolve() if args.template_project else None
    if args.use_template_timeline and not template_path:
        raise PipelineError("--use-template-timeline requires --template-project.")

    if args.list_resolve_presets:
        resolve = connect_resolve(logger)
        project = project_for_listing(resolve, args.project_name, f"DRautocut_MVP_{run_id}", logger)
        presets = list_render_presets(project)
        logger.event("resolve_render", "preset_list", "available render presets listed", presets=presets)
        return {
            "project": project.GetName(),
            "render_presets": presets,
            "log": str(logger.path),
        }

    info = probe_video(video, logger)
    audio = DEFAULT_WORK_DIR / f"{base}.mp3"
    srt = Path(args.srt).expanduser().resolve() if args.srt else DEFAULT_WORK_DIR / f"{base}.{args.engine}.srt"
    original_srt: Path | None = None
    correction_report: Path | None = None
    correction_result: dict[str, Any] | None = None
    subtitle_optimization_report: Path | None = None
    subtitle_optimization_result: dict[str, Any] | None = None

    if args.srt:
        logger.event("asr", "skipped", "using provided SRT", srt=str(srt), cue_count=count_srt_cues(srt))
    elif getattr(args, "segmented_asr", False):
        run_segmented_online_asr(video, srt, args.engine, args, logger)
    else:
        extract_audio(video, audio, logger)
        run_asr(audio, srt, args.engine, logger)

    logger.event("srt", "ready", "SRT ready for Resolve", srt=str(srt), cue_count=count_srt_cues(srt))
    terms: Path | None = None
    if args.context:
        context = Path(args.context).expanduser().resolve()
        if not context.exists():
            raise PipelineError(f"Course context file not found: {context}")
        terms = generate_terms_from_context(context, srt, base, args, logger)
    elif args.terms:
        terms = Path(args.terms).expanduser().resolve()

    if terms and getattr(args, "review_terms", False):
        result = {
            "video": str(video),
            "srt": str(srt),
            "terms": str(terms),
            "original_srt": str(original_srt) if original_srt else None,
            "needs_term_review": True,
            "log": str(logger.path),
        }
        logger.event(
            "term_review",
            "review",
            "terminology candidates ready for review",
            terms=str(terms),
            srt=str(srt),
            cue_count=count_srt_cues(srt),
        )
        return result

    if terms:
        if not terms.exists():
            raise PipelineError(f"Terms file not found: {terms}")
        original_srt = srt
        srt, correction_report, correction_result = correct_srt_with_terms(srt, terms, base, logger)
        logger.event("srt", "corrected", "corrected SRT ready for Resolve", srt=str(srt), cue_count=count_srt_cues(srt))

    if args.optimize_subtitles:
        if original_srt is None:
            original_srt = srt
        srt, subtitle_optimization_report, subtitle_optimization_result = optimize_subtitles(srt, base, args, logger)
        logger.event("srt", "optimized", "optimized SRT ready for Resolve", srt=str(srt), cue_count=count_srt_cues(srt))

    resolve = connect_resolve(logger)
    project_name = args.project_name or f"DRautocut_MVP_{run_id}"
    project = create_pipeline_project(resolve, project_name, template_path, logger)
    set_project_timing(project, info, logger)
    timeline_name = f"{stem_for(video)}_{run_id}"
    if args.use_template_timeline:
        media_pool, timeline = import_video_into_template_timeline(project, video, info, timeline_name, logger)
    else:
        media_pool, timeline = import_video_create_timeline(project, video, timeline_name, logger)

    if not try_import_srt(media_pool, timeline, srt, logger):
        raise PipelineError(
            "Resolve could not import the external SRT through the tested scripting paths. "
            f"Generated SRT is ready at: {srt}"
        )
    if args.use_template_timeline:
        logger.event(
            "resolve_subtitle_preset",
            "template",
            "using subtitle track style stored in the template timeline",
            template_project=str(template_path),
            subtitle_item_count=timeline_item_count(timeline, "subtitle"),
        )
    else:
        apply_subtitle_preset(
            project,
            timeline,
            args.subtitle_preset,
            logger,
            allow_fallback=args.allow_subtitle_preset_fallback,
        )
    validate_timeline_bounds(timeline, info, logger)
    if args.prepare_only:
        result = {
            "video": str(video),
            "srt": str(srt),
            "original_srt": str(original_srt) if original_srt else None,
            "correction_report": str(correction_report) if correction_report else None,
            "subtitle_optimization_report": str(subtitle_optimization_report) if subtitle_optimization_report else None,
            "template_project": str(template_path) if template_path else None,
            "used_template_timeline": bool(args.use_template_timeline),
            "project": project.GetName(),
            "timeline": timeline.GetName(),
            "log": str(logger.path),
        }
        logger.event("pipeline", "prepared", "timeline prepared; render skipped", **result)
        return result

    output_name = f"{stem_for(video)}_burned_{run_id}"
    rendered = render_timeline(
        project,
        DEFAULT_OUTPUT_DIR.resolve(),
        output_name,
        info,
        logger,
        render_preset,
        args.render_type,
        args.allow_render_type_fallback,
    )
    rendered_metadata = probe_rendered_file(rendered, logger)
    result = {
        "video": str(video),
        "srt": str(srt),
        "original_srt": str(original_srt) if original_srt else None,
        "correction_report": str(correction_report) if correction_report else None,
        "correction_result": correction_result,
        "subtitle_optimization_report": str(subtitle_optimization_report) if subtitle_optimization_report else None,
        "subtitle_optimization_result": subtitle_optimization_result,
        "template_project": str(template_path) if template_path else None,
        "used_template_timeline": bool(args.use_template_timeline),
        "rendered": str(rendered),
        "rendered_metadata": rendered_metadata,
        "log": str(logger.path),
    }
    logger.event("pipeline", "complete", "MVP pipeline complete", **result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MVP video -> ASR -> Resolve render pipeline.")
    parser.add_argument("--video", default=str(ROOT / "input" / "3min.mp4"), help="Input video path.")
    parser.add_argument("--engine", choices=["bcut"], default="bcut", help="Online ASR engine.")
    parser.add_argument("--srt", help="Skip ASR and use this existing SRT.")
    parser.add_argument(
        "--segmented-asr",
        action="store_true",
        help="Split long videos into audio segments and transcribe each segment independently.",
    )
    parser.add_argument(
        "--asr-max-workers",
        type=int,
        default=1,
        help="Segmented ASR concurrency. Keep 1 for online ASR unless the provider tolerates parallel jobs.",
    )
    parser.add_argument(
        "--asr-segment-minutes",
        type=float,
        default=10.0,
        help="Target length for each ASR segment when --segmented-asr is enabled.",
    )
    parser.add_argument(
        "--asr-max-segment-minutes",
        type=float,
        default=12.0,
        help="Hard maximum segment length for --segmented-asr.",
    )
    parser.add_argument("--terms", help="JSON terminology replacement map to apply before Resolve import.")
    parser.add_argument(
        "--context",
        help="Course context .txt/.md file. When set, an LLM-generated terms JSON is created before correction.",
    )
    parser.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--llm-system-prompt", default="", help="Custom system prompt for LLM terminology mapping.")
    parser.add_argument(
        "--optimize-subtitles",
        action="store_true",
        help="After terminology correction, remove configured punctuation and split cues over the max length.",
    )
    parser.add_argument("--subtitle-max-chars", type=int, default=20, help="Cue text length threshold for cue splitting.")
    parser.add_argument(
        "--remove-punctuation",
        default="，,",
        help="Characters to remove from subtitle text before overlong detection. Defaults to Chinese/English commas.",
    )
    parser.add_argument(
        "--no-subtitle-llm",
        action="store_true",
        help="Use local greedy cue splitting instead of LLM for overlong subtitles.",
    )
    parser.add_argument(
        "--subtitle-llm-timeout",
        type=int,
        default=30,
        help="Per-cue LLM timeout in seconds for subtitle splitting before local fallback.",
    )
    parser.add_argument(
        "--allow-neighbor-rewrite",
        action="store_true",
        help="Allow subtitle LLM to redistribute text across previous/current/next cues. Experimental; review required.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare Resolve timeline with video and SRT, then stop before rendering.",
    )
    parser.add_argument(
        "--render-current",
        action="store_true",
        help="Render the current Resolve project/timeline without importing video or SRT.",
    )
    parser.add_argument("--project-name", help="Resolve project name. Defaults to DRautocut_MVP_<run_id>.")
    parser.add_argument(
        "--template-project",
        help="Path to a Resolve .drp template project to import as the per-run project.",
    )
    parser.add_argument(
        "--use-template-timeline",
        action="store_true",
        help="Reuse the imported template project's existing timeline so saved subtitle track styling is preserved.",
    )
    parser.add_argument("--subtitle-preset", default=DEFAULT_SUBTITLE_PRESET, help="Subtitle style preset to apply.")
    parser.add_argument(
        "--allow-subtitle-preset-fallback",
        action="store_true",
        help="Continue if Resolve does not expose the requested subtitle preset to scripting.",
    )
    parser.add_argument("--render-type", default=DEFAULT_RENDER_TYPE, help="Resolve render type/encoder variant.")
    parser.add_argument(
        "--render-preset",
        default=DEFAULT_RENDER_PRESET,
        help="Resolve render preset name to load before render settings. Defaults to ffpg-fast-23.",
    )
    parser.add_argument(
        "--no-render-preset",
        action="store_true",
        help="Do not load the default render preset; use script-selected MP4/H.264 settings instead.",
    )
    parser.add_argument(
        "--list-resolve-presets",
        action="store_true",
        help="Connect to Resolve, load/create the target project, print available render presets, then exit.",
    )
    parser.add_argument(
        "--allow-render-type-fallback",
        action="store_true",
        help="Continue if Resolve rejects the requested render type through scripting.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        result = run_pipeline(parse_args())
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    print("\nDONE")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
