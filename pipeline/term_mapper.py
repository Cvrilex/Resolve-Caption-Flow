#!/usr/bin/env python3
"""Generate an auditable terminology replacement map from course context and SRT."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

try:  # Supports both ``python -m pipeline.term_mapper`` and direct script execution.
    from .term_corrector import parse_srt
except ImportError:  # pragma: no cover - direct script execution path
    from term_corrector import parse_srt


PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
DEFAULT_WORK_DIR = REPO_ROOT / "data" / "work"
BUNDLED_PYTHON = Path(
    os.environ.get(
        "DRAUTOCUT_PDF_PYTHON",
        "/Users/x/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
    )
)


class TermMapperError(RuntimeError):
    pass


def read_context(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return add_source_metadata(path, path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        return add_source_metadata(path, extract_pdf_text(path))
    raise TermMapperError(f"Unsupported context file type: {path.suffix}")


def add_source_metadata(path: Path, text: str) -> str:
    metadata = [
        "## Course Source Metadata",
        f"资料文件名：{path.name}",
        f"资料标题：{path.stem}",
    ]
    return "\n".join(metadata) + "\n\n" + text


def extract_pdf_text(path: Path) -> str:
    extracted: list[tuple[str, str]] = []
    errors: list[str] = []
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        errors.append(f"pypdf unavailable: {exc}")
        try:
            text = extract_pdf_text_with_python(path, exc)
            extracted.append(("pypdf-fallback", text))
        except Exception as fallback_exc:
            errors.append(str(fallback_exc))
    else:
        try:
            reader = PdfReader(str(path))
            extracted.append(("pypdf", extract_pdf_text_from_reader(reader)))
        except Exception as exc:
            errors.append(f"pypdf failed: {exc}")

    try:
        extracted.append(("pymupdf", extract_pdf_text_with_pymupdf(path)))
    except Exception as exc:
        errors.append(f"pymupdf failed: {exc}")

    usable = [(source, text) for source, text in extracted if len(text.strip()) >= 80]
    if usable:
        source, text = max(usable, key=lambda item: len(item[1]))
        return text + f"\n\n## PDF Extraction Metadata\n抽取方式：{source}\n抽取字符数：{len(text)}"

    if extracted:
        source, text = max(extracted, key=lambda item: len(item[1]))
        if text.strip():
            return text + f"\n\n## PDF Extraction Metadata\n抽取方式：{source}\n抽取字符数：{len(text)}\n警告：PDF 可抽取文字很少，术语候选可能为空。"

    detail = "；".join(errors[-3:])
    raise TermMapperError(
        "这个 PDF 没有可抽取的文字层，或文字层过少。它很可能是扫描图片版/课件截图版 PDF，"
        "当前版本需要先对 PDF 做 OCR 后再导入；也可以先换成可复制文字的 PDF。"
        + (f" 解析细节：{detail}" if detail else "")
    )


def extract_pdf_text_from_reader(reader: Any) -> str:
    chunks: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(f"## PDF Page {page_index}\n{text}")
    if not chunks:
        raise TermMapperError(
            "pypdf 没有从 PDF 中抽取到文字。"
        )
    return "\n\n".join(chunks)


def extract_pdf_text_with_pymupdf(path: Path) -> str:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise TermMapperError("PyMuPDF not installed") from exc

    chunks: list[str] = []
    with fitz.open(str(path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = (page.get_text("text") or "").strip()
            if text:
                chunks.append(f"## PDF Page {page_index}\n{text}")
    if not chunks:
        raise TermMapperError("PyMuPDF 没有从 PDF 中抽取到文字。")
    return "\n\n".join(chunks)


def diagnose_pdf(path: Path, preview_chars: int = 800) -> dict[str, Any]:
    """Inspect PDF text extractability without calling the LLM."""
    path = path.expanduser().resolve()
    report: dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "methods": [],
        "best_method": "",
        "best_char_count": 0,
        "preview": "",
        "status": "failed",
        "message": "",
    }

    def add_method(name: str, ok: bool, text: str = "", error: str = "", pages: int | None = None) -> None:
        method = {
            "name": name,
            "ok": bool(ok),
            "char_count": len(text.strip()),
            "error": error,
        }
        if pages is not None:
            method["pages"] = pages
        report["methods"].append(method)
        if ok and len(text.strip()) > int(report["best_char_count"]):
            report["best_method"] = name
            report["best_char_count"] = len(text.strip())
            report["preview"] = text.strip()[:preview_chars]

    try:
        from pypdf import PdfReader  # type: ignore

        try:
            reader = PdfReader(str(path))
            text = extract_pdf_text_from_reader(reader)
            add_method("pypdf", True, text, pages=len(reader.pages))
        except Exception as exc:
            pages = None
            try:
                pages = len(PdfReader(str(path)).pages)
            except Exception:
                pass
            add_method("pypdf", False, error=str(exc), pages=pages)
    except Exception as exc:
        add_method("pypdf", False, error=f"pypdf unavailable: {exc}")

    try:
        import fitz  # type: ignore

        try:
            with fitz.open(str(path)) as doc:
                page_count = len(doc)
            text = extract_pdf_text_with_pymupdf(path)
            add_method("pymupdf", True, text, pages=page_count)
        except Exception as exc:
            pages = None
            try:
                with fitz.open(str(path)) as doc:
                    pages = len(doc)
            except Exception:
                pass
            add_method("pymupdf", False, error=str(exc), pages=pages)
    except Exception as exc:
        add_method("pymupdf", False, error=f"PyMuPDF unavailable: {exc}")

    best = int(report["best_char_count"])
    if best >= 300:
        report["status"] = "ok"
        report["message"] = f"PDF 文字层可用，最佳解析器 {report['best_method']} 抽取 {best} 字。"
    elif best >= 80:
        report["status"] = "weak"
        report["message"] = f"PDF 可抽取文字较少，最佳解析器 {report['best_method']} 仅抽取 {best} 字，术语可能偏少。"
    elif best > 0:
        report["status"] = "weak"
        report["message"] = f"PDF 文字层很少，最佳解析器 {report['best_method']} 仅抽取 {best} 字，建议换可复制文字的 PDF 或先 OCR。"
    else:
        report["status"] = "image_only"
        report["message"] = "没有抽取到可用文字，疑似扫描图片版/课件截图版 PDF；当前需要先 OCR。"
    return report


def extract_pdf_text_with_python(path: Path, import_error: Exception) -> str:
    if not BUNDLED_PYTHON.exists() or BUNDLED_PYTHON.resolve() == Path(sys.executable).resolve():
        raise TermMapperError(
            "PDF parsing requires pypdf. 请双击 start_web.command 启动，让程序自动安装依赖；"
            "如果仍失败，请在项目目录执行 `.venv/bin/pip install -r requirements.txt`。"
        ) from import_error

    code = r"""
import json
import sys
from pypdf import PdfReader

reader = PdfReader(sys.argv[1])
chunks = []
for page_index, page in enumerate(reader.pages, start=1):
    text = (page.extract_text() or "").strip()
    if text:
        chunks.append(f"## PDF Page {page_index}\n{text}")
print(json.dumps({"text": "\n\n".join(chunks)}, ensure_ascii=False))
"""
    try:
        completed = subprocess.run(
            [str(BUNDLED_PYTHON), "-c", code, str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        payload = json.loads(completed.stdout or "{}")
        text = str(payload.get("text") or "")
    except Exception as exc:
        raise TermMapperError(
            "PDF parsing requires pypdf, and the bundled PDF parser fallback failed. "
            f"Active Python: {sys.executable}; fallback Python: {BUNDLED_PYTHON}; error: {exc}"
        ) from exc

    if not text.strip():
        raise TermMapperError(
            "No text could be extracted from the PDF. This may be an image-only PDF; OCR/MinerU is needed."
        )
    return text


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


def context_chunk_limit(base_url: str) -> tuple[int, int]:
    if is_local_base_url(base_url):
        return (
            int(os.environ.get("TERM_MAPPER_LOCAL_CONTEXT_CHUNK_CHARS", "1600")),
            int(os.environ.get("TERM_MAPPER_LOCAL_CONTEXT_MAX_CHUNKS", "10")),
        )
    return (
        int(os.environ.get("TERM_MAPPER_CONTEXT_CHUNK_CHARS", "1800")),
        int(os.environ.get("TERM_MAPPER_CONTEXT_MAX_CHUNKS", "16")),
    )


def chunk_context_text(text: str, max_chars: int = 3000, max_chunks: int = 12) -> list[str]:
    """Split PDF/course context into LLM-sized chunks without truncating early pages only."""
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return [""]

    metadata = ""
    if normalized.startswith("## Course Source Metadata"):
        marker = "\n\n"
        split_at = normalized.find(marker)
        if split_at > 0:
            metadata = normalized[:split_at].strip()
            normalized = normalized[split_at + len(marker) :].strip()

    blocks = re.split(r"(?=## PDF Page \d+\n)", normalized)
    blocks = [block.strip() for block in blocks if block.strip()]
    if not blocks:
        blocks = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        block_len = len(block) + 2
        if current and current_len + block_len > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
            if len(chunks) >= max_chunks:
                break
        if block_len > max_chars and not current:
            for start in range(0, len(block), max_chars):
                chunks.append(block[start : start + max_chars])
                if len(chunks) >= max_chunks:
                    break
            if len(chunks) >= max_chunks:
                break
            continue
        current.append(block)
        current_len += block_len
    if current and len(chunks) < max_chunks:
        chunks.append("\n\n".join(current))

    if metadata:
        chunks = [metadata + "\n\n" + chunk for chunk in chunks]
    return chunks or ([metadata] if metadata else [""])


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
    request_messages = [dict(message) for message in messages]
    if is_local_base_url(base_url) and "qwen" in model.lower():
        for message in reversed(request_messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if "/no_think" not in content:
                    message["content"] = f"{content}\n/no_think"
                break
    payload: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        "temperature": 0.1,
        "max_tokens": int(os.environ.get("TERM_MAPPER_MAX_TOKENS", "4096")),
    }
    if should_disable_thinking(model, base_url):
        payload["thinking"] = {"type": "disabled"}
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
        message = data["choices"][0]["message"]
        content = str(message.get("content") or "").strip()
        if content:
            return content
        reasoning = str(message.get("reasoning_content") or "").strip()
        if reasoning and "{" in reasoning and "}" in reasoning:
            try:
                extract_json_object(reasoning)
                return reasoning
            except Exception:
                pass
        if reasoning:
            raise TermMapperError("LLM returned reasoning content but no final answer; thinking mode should be disabled for JSON tasks.")
        return content
    except TermMapperError:
        raise
    except Exception as exc:
        raise TermMapperError(f"Unexpected LLM response shape: {data}") from exc


def is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def is_deepseek_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host == "api.deepseek.com" or host.endswith(".deepseek.com")


def should_disable_thinking(model: str, base_url: str) -> bool:
    model_name = model.lower()
    return is_deepseek_base_url(base_url) and ("v4" in model_name or "reasoner" in model_name)


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


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def _collapse_cjk_layout_spaces(text: str) -> str:
    previous = None
    current = str(text or "")
    while previous != current:
        previous = current
        current = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", current)
    return current.strip()


def _abbreviation_from_text(text: str) -> str:
    compact_spaced = re.sub(r"\s+", "", text or "")
    if re.fullmatch(r"[A-Za-z]{2,}[A-Za-z0-9+-]*", compact_spaced):
        letters = re.findall(r"[A-Za-z]", compact_spaced)
        if len(letters) >= 2 and (len(compact_spaced) <= 8 or any(ch.isupper() for ch in compact_spaced)):
            return compact_spaced.upper() if compact_spaced.islower() else compact_spaced
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]{1,}", text or ""):
        letters = re.findall(r"[A-Za-z]", token)
        if len(letters) < 2:
            continue
        if token.islower() and len(token) > 6:
            continue
        if any(ch.isupper() for ch in token) or token.islower():
            return token.upper() if token.islower() else token
    return ""


def normalize_abbreviation_translation(row: dict[str, Any]) -> dict[str, Any]:
    """Keep English abbreviations when the model returns only a Chinese expansion."""
    correct = str(row.get("correct") or "").strip()
    if not correct or not _has_cjk(correct) or _has_latin(correct):
        return row
    candidates = [str(row.get("wrong") or "")]
    aliases = row.get("aliases")
    if isinstance(aliases, list):
        candidates.extend(str(value) for value in aliases)
    abbreviation = next((abbr for abbr in (_abbreviation_from_text(value) for value in candidates) if abbr), "")
    if not abbreviation:
        return row
    return {
        **row,
        "correct": f"{abbreviation}（{correct}）",
        "note": (str(row.get("note") or "") + "；英文缩写保留并补充中文解释").strip("；"),
    }


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
        wrong = _collapse_cjk_layout_spaces(wrong)
        correct = _collapse_cjk_layout_spaces(correct)
        if not wrong or not correct or wrong == correct:
            continue
        normalized_item = normalize_abbreviation_translation({**item, "wrong": wrong, "correct": correct})
        correct = str(normalized_item.get("correct", "")).strip()
        correct = _collapse_cjk_layout_spaces(correct)
        if wrong == correct:
            continue
        key = (wrong, correct)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "wrong": wrong,
                "correct": correct,
                "aliases": [
                    str(v).strip()
                    for v in normalized_item.get("aliases", []) or []
                    if str(v).strip() and str(v).strip() not in {wrong, correct}
                ][:8],
                "patterns": [
                    str(v).strip()
                    for v in normalized_item.get("patterns", []) or []
                    if str(v).strip()
                ][:5],
                "confidence": normalized_item.get("confidence", "medium"),
                "evidence": str(normalized_item.get("evidence", ""))[:300],
                "note": str(normalized_item.get("note", ""))[:300],
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


def _compact_for_membership(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _strip_parenthetical_parts(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = [raw]
    for outside, inside in re.findall(r"^(.+?)[（(]([^（）()]+)[）)]$", raw):
        parts.extend([outside.strip(), inside.strip()])
    return [part for part in dict.fromkeys(parts) if part]


def _correct_is_supported_by_context(correct: str, context_compact: str) -> bool:
    correct_compact = _compact_for_membership(correct)
    if correct_compact in context_compact:
        return True
    parts = _strip_parenthetical_parts(correct)
    if len(parts) < 3:
        return False
    outside, inside = parts[1], parts[2]
    outside_compact = _compact_for_membership(outside)
    inside_compact = _compact_for_membership(inside)
    # Abbreviation expansions often do not appear as one literal string in PDFs:
    # "GERD" and "胃食管反流病" may appear separately, while our standard subtitle
    # form is "GERD（胃食管反流病）".
    return bool(outside_compact and inside_compact and outside_compact in context_compact and inside_compact in context_compact)


def filter_context_only_replacements(
    payload: dict[str, Any],
    context_text: str,
    *,
    keep_unverified: bool = False,
) -> dict[str, Any]:
    context_compact = _compact_for_membership(context_text)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in payload.get("replacements", []):
        wrong = str(item.get("wrong", "")).strip()
        correct = str(item.get("correct", "")).strip()
        wrong_compact = _compact_for_membership(wrong)
        correct_compact = _compact_for_membership(correct)
        reason = ""
        if not wrong or not correct or wrong == correct:
            reason = "empty_or_same"
        elif not _correct_is_supported_by_context(correct, context_compact):
            reason = "correct_not_in_pdf_context"

        if reason == "correct_not_in_pdf_context" and keep_unverified:
            item = normalize_abbreviation_translation(item)
            kept.append(
                {
                    **item,
                    "confidence": "low",
                    "review_warning": "correct_not_verified_in_pdf_text",
                    "note": (
                        str(item.get("note") or "")
                        + "；PDF 抽取文本未能逐字验证该标准写法，请人工确认"
                    ).strip("；"),
                }
            )
            dropped.append({**item, "drop_reason": reason, "kept_for_manual_review": True})
        elif reason:
            dropped.append({**item, "drop_reason": reason})
        else:
            item = normalize_abbreviation_translation(item)
            if wrong_compact in context_compact:
                item = {
                    **item,
                    "review_warning": "wrong_also_appears_in_pdf_context",
                }
            kept.append(item)

    return {
        **payload,
        "replacements": kept,
        "dropped_replacements": dropped,
    }


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
      "aliases": ["同一错误的其他写法，可为空数组"],
      "patterns": ["安全正则，可为空数组"],
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
- aliases 只放同一个错误的其他高置信写法，例如空格、大小写、中文单位差异。
- patterns 只用于非常安全的单位或缩写归一化；不确定就留空数组。
- 术语表要尽量精简，不要把课程资料里的名词原样整理成词库。
- 优先保留人名、地名、学校名、医院/机构名、会议/指南名、英文缩写、药物名、检查/术式名、罕见病名、冷门或容易被 ASR 误识别的专业名词。
- 学科中常见且 ASR 不容易混淆的普通术语不要输出，例如常见疾病大类、常规检查、普通症状、通用动词和泛化概念。
- 如果 ASR 已经识别正确，不要输出。
- correct 可以来自课程资料，也可以基于相关医学领域专业知识进行高置信校正；但必须能用课程资料、字幕上下文或医学常识给出简短依据。
- 对英文缩写、药物名、指南名、人名和机构名保持标准大小写和标准写法。
- 英文缩写不要直接替换成中文全称。如果需要补充中文解释，correct 使用“英文缩写（中文翻译）”，例如“ACEI（血管紧张素转换酶抑制剂）”，不要输出“ACEI->血管紧张素转换酶抑制剂”。
- 每个字幕块只输出最值得审核的少量候选，宁缺毋滥。

课程资料：
{context_text}

ASR字幕：
{srt_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_context_only_messages(context_text: str, system_prompt: str = "") -> list[dict[str, str]]:
    system = system_prompt.strip() or (
        "你是医疗课程术语整理助手。任务是只基于课程PDF资料，生成精简、可审核的课程标准术语表。"
        "只输出JSON。不要逐句改写，不要翻译全文。"
    )
    user = f"""
请从下面课程资料中提取高价值课程术语，并为后续ASR字幕校对准备候选替换表。

输出 JSON：
{{
  "replacements": [
    {{
      "wrong": "可能的ASR错误写法或别名",
      "correct": "课程中的标准写法",
      "aliases": ["其他可能错误写法，可为空数组"],
      "patterns": ["非常安全的正则，可为空数组"],
      "confidence": "high|medium|low",
      "evidence": "来自课程资料的简短依据",
      "note": "为什么需要关注"
    }}
  ]
}}

约束：
- 这是 ASR 前的课程术语表，不要依赖字幕实际错词。
- wrong 可以是标准术语的常见同音误写、口语误识别、大小写/空格差异或中文别名。
- 如果无法判断可能错写，wrong 可填写最可能被ASR混淆的短别名，但不要让 wrong 与 correct 完全相同。
- correct 必须是课程资料中真实出现的标准写法，不要编造课程资料中没有的写法。
- 课程资料包括 PDF 正文和上方的课程源文件名/资料标题；讲者姓名、课程名如果只出现在文件名中，也可以作为 correct 来源。
- wrong 不能是课程资料中真实出现的医学同义词、旧称、上位词或另一种正确表达。
- 不要输出“高血压危象->高血压急症”这类概念替换；这不是 ASR 错词。
- 推荐优先输出容易被 ASR 误写的人名和专名，例如讲者姓名错字、医院简称补全、课程强相关英文缩写误写。
- 不要复用本提示词中的说明性示例文字；所有 wrong/correct 都必须来自当前课程资料或基于当前课程资料可高置信推断。
- aliases 只放同一个术语的其他高置信写法。
- patterns 只用于非常安全的单位、缩写、大小写或空格归一化。
- 优先保留人名、地名、学校名、医院/机构名、指南名、英文缩写、药物名、检查/术式名、罕见病名、冷门或容易被ASR误识别的专业名词。
- 英文缩写不要直接替换成中文全称。如果需要补充中文解释，correct 使用“英文缩写（中文翻译）”，例如“ACEI（血管紧张素转换酶抑制剂）”，不要输出“ACEI->血管紧张素转换酶抑制剂”。
- 学科中常见且不容易混淆的普通术语不要输出。
- 不要输出低置信替换。
- 每个术语尽量短，宁缺毋滥。
- 最多输出 8 条 replacements；如果候选很多，只保留最有可能被 ASR 误写、且用户最需要提前审核的 8 条。
- wrong 和 correct 不能完全相同。

课程资料：
{context_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_terms_from_context(
    context: Path,
    output: Path,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str = "",
    timeout: int = 45,
    retries: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    keep_unverified: bool = True,
) -> dict[str, Any]:
    full_context_text = read_context(context)
    chunk_limit, max_chunks = context_chunk_limit(base_url)
    context_chunks = chunk_context_text(full_context_text, max_chars=chunk_limit, max_chunks=max_chunks)
    context_char_count = len(full_context_text)
    if progress_callback:
        progress_callback({
            "status": "planned",
            "message": "PDF terminology extraction planned",
            "context_chars": context_char_count,
            "chunk_count": len(context_chunks),
            "context_chunk_chars": chunk_limit,
        })

    term_lists: list[list[dict[str, Any]]] = []
    dropped_replacements: list[dict[str, Any]] = []
    chunk_errors: list[dict[str, Any]] = []
    for index, context_text in enumerate(context_chunks, start=1):
        if progress_callback:
            progress_callback({
                "status": "progress",
                "message": "extracting terminology from PDF",
                "current": index,
                "total": len(context_chunks),
                "percent": round(index * 100 / max(1, len(context_chunks))),
            })
        chunk_error: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                raw = chat_completion(
                    build_context_only_messages(context_text, system_prompt),
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout=timeout,
                )
                chunk_payload = filter_context_only_replacements(
                    normalize_terms_payload(raw),
                    full_context_text,
                    keep_unverified=keep_unverified,
                )
                replacements = chunk_payload.get("replacements", [])
                term_lists.append(replacements)
                dropped_replacements.extend(
                    {
                        **item,
                        "chunk": index,
                    }
                    for item in chunk_payload.get("dropped_replacements", [])
                    if isinstance(item, dict)
                )
                if progress_callback:
                    progress_callback({
                        "status": "chunk_done",
                        "message": "PDF terminology extraction chunk complete",
                        "current": index,
                        "total": len(context_chunks),
                        "attempt": attempt,
                        "replacement_count": len(replacements),
                        "dropped_replacement_count": len(chunk_payload.get("dropped_replacements", [])),
                    })
                chunk_error = None
                break
            except Exception as exc:
                chunk_error = exc
                if progress_callback:
                    progress_callback({
                        "status": "chunk_retry" if attempt <= retries else "chunk_failed",
                        "message": "PDF terminology extraction chunk failed; retrying" if attempt <= retries else "PDF terminology extraction chunk failed; skipping",
                        "current": index,
                        "total": len(context_chunks),
                        "attempt": attempt,
                        "max_attempts": retries + 1,
                        "error": str(exc),
                    })
        if chunk_error is not None:
            chunk_errors.append({"chunk": index, "error": str(chunk_error)})

    if chunk_errors and len(chunk_errors) == len(context_chunks):
        raise TermMapperError(f"PDF terminology extraction failed: {chunk_errors[-1]['error']}")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replacements": merge_replacements(term_lists),
        "dropped_replacements": dropped_replacements,
        "chunk_count": len(context_chunks),
        "chunk_error_count": len(chunk_errors),
        "chunk_errors": chunk_errors,
    }

    payload["source_context"] = str(context)
    payload["context_char_count"] = context_char_count
    payload["context_chunk_chars"] = chunk_limit
    payload["context_max_chunks"] = max_chunks
    payload["context_preview"] = full_context_text[:600]
    payload["mode"] = "pdf_preflight"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback({
            "status": "done",
            "message": "PDF terminology extraction complete",
            "replacement_count": len(payload.get("replacements", [])),
            "dropped_replacement_count": len(payload.get("dropped_replacements", [])),
            "terms": str(output),
        })
    return payload


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
    parser.add_argument("--srt", help="ASR SRT path.")
    parser.add_argument("--context-only", action="store_true", help="Generate a PDF/course terminology table before ASR.")
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
        if args.context_only:
            print(json.dumps(build_context_only_messages(context_text, args.system_prompt), ensure_ascii=False, indent=2))
        else:
            if not args.srt:
                raise TermMapperError("--srt is required unless --context-only is set.")
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
    if args.context_only:
        result = generate_terms_from_context(
            context=Path(args.context).expanduser().resolve(),
            output=Path(args.output).expanduser().resolve(),
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            system_prompt=args.system_prompt,
            timeout=args.timeout,
            retries=max(0, args.retries),
            progress_callback=progress_callback,
        )
    else:
        if not args.srt:
            raise TermMapperError("--srt is required unless --context-only is set.")
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
