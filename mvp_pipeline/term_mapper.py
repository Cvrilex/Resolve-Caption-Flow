#!/usr/bin/env python3
"""Generate an auditable terminology replacement map from course context and SRT."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
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


def compact_context(text: str, max_chars: int = 18000) -> str:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    return normalized[:max_chars]


def compaction_limits(base_url: str) -> tuple[int, int]:
    if is_local_base_url(base_url):
        return (
            int(os.environ.get("TERM_MAPPER_LOCAL_CONTEXT_CHARS", "1800")),
            int(os.environ.get("TERM_MAPPER_LOCAL_SRT_CHARS", "1400")),
        )
    return (
        int(os.environ.get("TERM_MAPPER_CONTEXT_CHARS", "18000")),
        int(os.environ.get("TERM_MAPPER_SRT_CHARS", "14000")),
    )


def chat_completion(messages: list[dict[str, str]], model: str, base_url: str, api_key: str) -> str:
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
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TermMapperError(f"LLM request failed: {exc.code} {detail[:1000]}") from exc
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


def build_messages(context_text: str, srt_text: str, system_prompt: str = "") -> list[dict[str, str]]:
    system = system_prompt.strip() or (
        "你是医疗课程中文字幕校对助手。任务是基于课程资料和ASR字幕，"
        "生成可审核的术语替换表。只输出JSON。不要改写整句，不要翻译，"
        "只提出高置信的错词->正确术语替换。遇到不确定的内容不要输出。"
    )
    user = f"""
请从下面两部分内容中找出 ASR 字幕里的医学术语、人名、机构名、英文缩写错误，
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
- correct 应优先来自课程资料。
- 保留英文缩写如 TED、MDT、TAO 的标准大写形式。

课程资料：
{context_text}

ASR字幕：
{srt_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_terms(context: Path, srt: Path, output: Path, model: str, base_url: str, api_key: str,
                   system_prompt: str = "") -> dict[str, Any]:
    context_limit, srt_limit = compaction_limits(base_url)
    context_text = compact_context(read_context(context), max_chars=context_limit)
    srt_text = compact_srt_text(srt, max_chars=srt_limit)
    raw = chat_completion(
        build_messages(context_text, srt_text, system_prompt),
        model=model, base_url=base_url, api_key=api_key,
    )
    payload = normalize_terms_payload(raw)
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
    result = generate_terms(
        context=Path(args.context).expanduser().resolve(),
        srt=Path(args.srt).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
