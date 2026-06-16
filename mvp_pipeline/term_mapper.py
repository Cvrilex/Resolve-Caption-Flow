#!/usr/bin/env python3
"""Generate an auditable terminology replacement map from course context and SRT."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from term_corrector import parse_srt


ROOT = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = ROOT / "work"


class TermMapperError(RuntimeError):
    pass


def read_context(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return extract_pdf_text(path)
    raise TermMapperError(f"Unsupported context file type: {path.suffix}")


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise TermMapperError(
            "PDF parsing requires pypdf. Run term_mapper.py with the bundled Codex Python runtime "
            "or install pypdf for the active Python."
        ) from exc

    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(f"## PDF Page {page_index}\n{text}")
    if not chunks:
        raise TermMapperError(
            "No text could be extracted from the PDF. This may be an image-only PDF; OCR/MinerU is needed."
        )
    return "\n\n".join(chunks)


def compact_srt_text(srt: Path, max_chars: int = 14000) -> str:
    cues = parse_srt(srt)
    rows = [f"{cue.index} {cue.timing} {' '.join(cue.lines)}" for cue in cues]
    text = "\n".join(rows)
    return text[:max_chars]


def chunk_srt_text(srt: Path, max_chars: int = 8000) -> list[str]:
    cues = parse_srt(srt)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for cue in cues:
        row = f"{cue.index} {cue.timing} {' '.join(cue.lines)}"
        row_len = len(row) + 1
        if current and current_len + row_len > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(row)
        current_len += row_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def compact_context(text: str, max_chars: int = 18000) -> str:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    return normalized[:max_chars]


def compaction_limits(base_url: str) -> tuple[int, int]:
    if is_local_base_url(base_url):
        return (
            int(os.environ.get("TERM_MAPPER_LOCAL_CONTEXT_CHARS", "1800")),
            int(os.environ.get("TERM_MAPPER_LOCAL_SRT_CHUNK_CHARS", "1400")),
        )
    return (
        int(os.environ.get("TERM_MAPPER_CONTEXT_CHARS", "18000")),
        int(os.environ.get("TERM_MAPPER_SRT_CHUNK_CHARS", "8000")),
    )


def chat_completion(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    api_key: str,
    timeout: int = 45,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    if not is_local_base_url(base_url):
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TermMapperError(f"LLM request failed: {exc.code} {detail[:1000]}") from exc
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        raise TermMapperError(f"LLM request failed: {exc}") from exc
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise TermMapperError(f"Unexpected LLM response shape: {data}") from exc


def is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def normalize_terms_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(extract_json_object(raw))
    except json.JSONDecodeError as exc:
        raise TermMapperError(f"LLM did not return valid JSON: {raw[:1000]}") from exc
    replacements = payload.get("replacements")
    if not isinstance(replacements, list):
        raise TermMapperError("LLM JSON must contain a replacements array")

    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in replacements:
        if not isinstance(item, dict):
            continue
        wrong = str(item.get("wrong", "")).strip()
        correct = str(item.get("correct", "")).strip()
        if not wrong or not correct or wrong == correct:
            continue
        key = (wrong, correct)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "wrong": wrong,
                "correct": correct,
                "confidence": item.get("confidence", "medium"),
                "evidence": str(item.get("evidence", ""))[:300],
                "note": str(item.get("note", ""))[:300],
            }
        )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replacements": cleaned,
    }


def merge_replacements(term_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for terms in term_lists:
        for item in terms:
            wrong = str(item.get("wrong", "")).strip()
            correct = str(item.get("correct", "")).strip()
            if not wrong or not correct or wrong == correct:
                continue
            key = (wrong, correct)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def build_messages(context_text: str, srt_text: str, system_prompt: str = "") -> list[dict[str, str]]:
    system = system_prompt.strip() or (
        "你是医疗课程中文字幕校对助手。任务是基于课程资料和ASR字幕，"
        "生成精简、可审核的术语替换表。只输出JSON。不要改写整句，不要翻译，"
        "只提出高置信的错词->正确术语替换。遇到不确定的内容不要输出。"
    )
    user = f"""
请从下面两部分内容中找出 ASR 字幕里的高价值识别错误，
输出 JSON：

{{
  "replacements": [
    {{
      "wrong": "字幕里的错误写法",
      "correct": "标准写法",
      "confidence": "high|medium|low",
      "evidence": "来自课程资料或字幕上下文的简短依据",
      "note": "为什么替换"
    }}
  ]
}}

约束：
- 只输出可以批量替换的短词/短语。
- 不要输出低置信替换。
- 不要输出会改变医学含义的猜测。
- wrong 必须出现在 ASR 字幕中。
- 术语表要尽量精简，不要把课程资料里的名词原样整理成词库。
- 优先保留人名、地名、学校名、医院/机构名、会议/指南名、英文缩写、药物名、检查/术式名、罕见病名、冷门或容易被 ASR 误识别的专业名词。
- 学科中常见且 ASR 不容易混淆的普通术语不要输出，例如常见疾病大类、常规检查、普通症状、通用动词和泛化概念。
- 如果 ASR 已经识别正确，不要输出。
- correct 可以来自课程资料，也可以基于相关医学领域专业知识进行高置信校正；但必须能用课程资料、字幕上下文或医学常识给出简短依据。
- 对英文缩写、药物名、指南名、人名和机构名保持标准大小写和标准写法。
- 每个字幕块只输出最值得审核的少量候选，宁缺毋滥。

课程资料：
{context_text}

ASR字幕：
{srt_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_terms(
    context: Path,
    srt: Path,
    output: Path,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str = "",
    timeout: int = 45,
    retries: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    context_limit, srt_chunk_limit = compaction_limits(base_url)
    context_text = compact_context(read_context(context), max_chars=context_limit)
    srt_chunks = chunk_srt_text(srt, max_chars=srt_chunk_limit)
    if progress_callback:
        progress_callback({
            "status": "planned",
            "message": "terminology mapping chunks planned",
            "chunk_count": len(srt_chunks),
            "context_chars": len(context_text),
            "srt_chunk_chars": srt_chunk_limit,
        })

    term_lists: list[list[dict[str, Any]]] = []
    chunk_errors: list[dict[str, Any]] = []
    for index, srt_text in enumerate(srt_chunks, start=1):
        if progress_callback:
            progress_callback({
                "status": "progress",
                "message": "generating terminology map chunk",
                "current": index,
                "total": len(srt_chunks),
                "percent": round(index * 100 / max(1, len(srt_chunks))),
            })
        chunk_error: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                raw = chat_completion(
                    build_messages(context_text, srt_text, system_prompt),
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout=timeout,
                )
                chunk_payload = normalize_terms_payload(raw)
                replacements = chunk_payload.get("replacements", [])
                term_lists.append(replacements)
                if progress_callback:
                    progress_callback({
                        "status": "chunk_done",
                        "message": "terminology map chunk complete",
                        "current": index,
                        "total": len(srt_chunks),
                        "attempt": attempt,
                        "replacement_count": len(replacements),
                    })
                chunk_error = None
                break
            except Exception as exc:
                chunk_error = exc
                if progress_callback:
                    progress_callback({
                        "status": "chunk_retry" if attempt <= retries else "chunk_failed",
                        "message": (
                            "terminology map chunk failed; retrying"
                            if attempt <= retries
                            else "terminology map chunk failed; skipping"
                        ),
                        "current": index,
                        "total": len(srt_chunks),
                        "attempt": attempt,
                        "max_attempts": retries + 1,
                        "error": str(exc),
                    })
        if chunk_error is not None:
            chunk_errors.append({"chunk": index, "error": str(chunk_error)})

    if chunk_errors and len(chunk_errors) == len(srt_chunks):
        raise TermMapperError(f"All terminology map chunks failed: {chunk_errors[-1]['error']}")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replacements": merge_replacements(term_lists),
        "chunk_count": len(srt_chunks),
        "chunk_error_count": len(chunk_errors),
        "chunk_errors": chunk_errors,
    }
    payload["source_context"] = str(context)
    payload["source_srt"] = str(srt)
    payload["model"] = model
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a terminology replacement JSON from course context and SRT.")
    parser.add_argument("--context", required=True, help="Course context file path (.txt/.md for this MVP).")
    parser.add_argument("--srt", required=True, help="ASR SRT path.")
    parser.add_argument("--output", default=str(DEFAULT_WORK_DIR / "terms.generated.json"), help="Output terms JSON path.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--system-prompt", default="", help="Custom system prompt for the LLM.")
    parser.add_argument("--print-prompt", action="store_true", help="Print the LLM messages without making a request.")
    parser.add_argument("--progress-jsonl", action="store_true", help="Write progress events as JSON lines to stderr.")
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("TERM_MAPPER_TIMEOUT", "45")))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("TERM_MAPPER_RETRIES", "1")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_prompt:
        context_text = compact_context(read_context(Path(args.context).expanduser().resolve()))
        srt_text = compact_srt_text(Path(args.srt).expanduser().resolve())
        print(json.dumps(build_messages(context_text, srt_text, args.system_prompt), ensure_ascii=False, indent=2))
        return 0
    if not args.api_key:
        raise TermMapperError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    progress_callback = None
    if args.progress_jsonl:
        def emit_progress(payload: dict[str, Any]) -> None:
            print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
        progress_callback = emit_progress
    result = generate_terms(
        context=Path(args.context).expanduser().resolve(),
        srt=Path(args.srt).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
        timeout=args.timeout,
        retries=max(0, args.retries),
        progress_callback=progress_callback,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
