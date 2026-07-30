#!/usr/bin/env python3
"""Batch LLM medical terminology review for SRT files using auditable patches."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

try:  # Supports both package imports and direct script execution.
    from .subtitle_optimizer import (
        build_messages as build_overlong_split_messages,
        build_short_window_messages,
        canonical_content,
        clean_subtitle_text,
        enforce_max_chars,
        greedy_split,
        interpolate_timestamps,
        merge_hanging_value_connector_cues,
        merge_short_segments,
        normalize_segments_payload,
        repair_hanging_value_connectors,
        repair_protected_term_boundaries,
        visible_len,
    )
    from .term_corrector import Cue, parse_srt, write_srt
except ImportError:  # pragma: no cover - direct script execution path
    from subtitle_optimizer import (
        build_messages as build_overlong_split_messages,
        build_short_window_messages,
        canonical_content,
        clean_subtitle_text,
        enforce_max_chars,
        greedy_split,
        interpolate_timestamps,
        merge_hanging_value_connector_cues,
        merge_short_segments,
        normalize_segments_payload,
        repair_hanging_value_connectors,
        repair_protected_term_boundaries,
        visible_len,
    )
    from term_corrector import Cue, parse_srt, write_srt


class LLMTermReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class CuePatch:
    cue: int
    old: str
    new: str
    reason: str = ""
    confidence: str = "medium"


@dataclass(frozen=True)
class ReflowSuggestion:
    cue_ids: tuple[int, ...]
    segments: tuple[str, ...]
    reason: str = ""
    confidence: str = "medium"


def is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def is_deepseek_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host == "api.deepseek.com" or host.endswith(".deepseek.com")


def should_disable_thinking(model: str, base_url: str) -> bool:
    model_name = model.lower()
    return is_deepseek_base_url(base_url) and ("v4" in model_name or "reasoner" in model_name)


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
        "max_tokens": int(os.environ.get("LLM_TERM_REVIEW_MAX_TOKENS", "4096")),
    }
    if should_disable_thinking(model, base_url):
        payload["thinking"] = {"type": "disabled"}
    if not is_local_base_url(base_url):
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
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
        raise LLMTermReviewError(f"LLM request failed: {exc.code} {detail[:1000]}") from exc
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        raise LLMTermReviewError(f"LLM request failed: {exc}") from exc
    try:
        message = data["choices"][0]["message"]
        content = str(message.get("content") or "").strip()
        if content:
            return content
        reasoning = str(message.get("reasoning_content") or "").strip()
        if reasoning and "{" in reasoning and "}" in reasoning:
            return reasoning
        if reasoning:
            raise LLMTermReviewError("LLM returned reasoning content but no final answer.")
        return content
    except LLMTermReviewError:
        raise
    except Exception as exc:
        raise LLMTermReviewError(f"Unexpected LLM response shape: {data}") from exc


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


def load_terms_summary(path: Path | None, max_chars: int = 4000) -> str:
    if not path or not path.exists():
        return "无用户确认课程术语表。"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    rows = data.get("replacements", []) if isinstance(data, dict) else []
    lines: list[str] = []
    for item in rows[:80]:
        if not isinstance(item, dict):
            continue
        wrong = str(item.get("wrong", "")).strip()
        correct = str(item.get("correct", "")).strip()
        note = str(item.get("note", "") or item.get("evidence", "")).strip()
        if wrong and correct:
            lines.append(f"- {wrong} -> {correct}" + (f" ({note[:80]})" if note else ""))
    return "\n".join(lines)[:max_chars] if lines else "用户确认术语表为空。"


def cue_text(cue: Cue) -> str:
    return " ".join(line.strip() for line in cue.lines if line.strip())


def make_batches(cues: list[Cue], batch_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if batch_size <= 0:
        raise LLMTermReviewError("batch_size must be positive")
    batches: list[tuple[int, int, int, int]] = []
    start = 0
    while start < len(cues):
        end = min(len(cues), start + batch_size)
        context_start = max(0, start - overlap)
        context_end = min(len(cues), end + overlap)
        batches.append((start, end, context_start, context_end))
        start = end
    return batches


def build_messages(
    cues: list[Cue],
    body_start: int,
    body_end: int,
    context_start: int,
    context_end: int,
    terms_summary: str,
    min_chars: int,
    max_chars: int,
    system_prompt: str = "",
) -> list[dict[str, str]]:
    system = system_prompt.strip() or (
        "你是医疗课程 ASR 字幕全量审校助手。只输出 JSON。"
        "你的任务是找出字幕中的医学术语、人名、机构名、单位和英文缩写错误，"
        "并顺带给出明显不自然断句的局部重排建议；不要润色，不要扩写。"
    )
    rows: list[str] = []
    allowed_range = f"{int(cues[body_start].index)}-{int(cues[body_end - 1].index)}"
    for idx in range(context_start, context_end):
        cue = cues[idx]
        marker = "EDIT" if body_start <= idx < body_end else "CTX"
        length = visible_len(cue_text(cue))
        flags: list[str] = []
        if length < min_chars:
            flags.append("SHORT")
        if length > max_chars:
            flags.append("LONG")
        flag_text = f" flags={','.join(flags)}" if flags else ""
        rows.append(f"[{marker}] {cue.index} len={length}{flag_text}: {cue_text(cue)}")
    user = f"""
请审校下面字幕，只允许修改标记为 EDIT 的字幕；CTX 只作为上下文。

用户确认课程术语表：
{terms_summary}

输出 JSON：
{{
  "patches": [
    {{
      "cue": 123,
      "old": "字幕中的原文片段",
      "new": "修正后的片段",
      "reason": "为什么这是医学术语/单位/专名修正",
      "confidence": "high|medium|low"
    }}
  ],
  "reflows": [
    {{
      "cue_ids": [123, 124, 125],
      "segments": ["重排后的第一条字幕", "重排后的第二条字幕"],
      "reason": "为什么这里需要语义断句/合并",
      "confidence": "high|medium|low"
    }}
  ]
}}

规则：
- patches 只输出 cue 在 {allowed_range} 范围内的项目。
- old 必须是该 cue 字幕中连续出现的原文片段。
- 不要输出 old 和 new 完全相同的 patch。
- 只处理医学术语、单位、人名、学校名、医院/机构名、药物名、检查/术式名、英文缩写大小写。
- reflows 只用于明显过短、语义断裂或不自然断句；cue_ids 必须连续，且全部在 {allowed_range} 范围内。
- reflows 的 segments 只能重排 cue_ids 范围内原有文字，不要在 reflows 里做术语修正。
- reflows 的目标是让字幕尽量落在 {min_chars}-{max_chars} 个可见字之间；医学缩写、数字单位、人名可略微突破。
- 不要删除口语，不要润色表达。
- 不要做概念替换，例如不要把“高血压危象”改成“高血压急症”，除非字幕显然是 ASR 错字。
- 没有高把握就返回空数组。
- 最多输出 20 条 patches，最多输出 10 条 reflows。

字幕：
{chr(10).join(rows)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_patches(raw: str) -> list[CuePatch]:
    try:
        data = json.loads(extract_json_object(raw))
    except json.JSONDecodeError as exc:
        raise LLMTermReviewError(f"LLM did not return valid JSON: {raw[:1000]}") from exc
    rows = data.get("patches", []) if isinstance(data, dict) else []
    if not isinstance(rows, list):
        raise LLMTermReviewError("LLM JSON must contain a patches array")
    patches: list[CuePatch] = []
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            cue = int(row.get("cue"))
        except (TypeError, ValueError):
            continue
        old = str(row.get("old", "")).strip()
        new = str(row.get("new", "")).strip()
        if not old or not new or old == new:
            continue
        key = (cue, old, new)
        if key in seen:
            continue
        seen.add(key)
        patches.append(
            CuePatch(
                cue=cue,
                old=old,
                new=new,
                reason=str(row.get("reason", ""))[:300],
                confidence=str(row.get("confidence", "medium")).lower(),
            )
        )
    return patches[:30]


def normalize_review_payload(raw: str) -> tuple[list[CuePatch], list[ReflowSuggestion]]:
    try:
        data = json.loads(extract_json_object(raw))
    except json.JSONDecodeError as exc:
        raise LLMTermReviewError(f"LLM did not return valid JSON: {raw[:1000]}") from exc
    if not isinstance(data, dict):
        raise LLMTermReviewError("LLM JSON must be an object")

    patches = normalize_patches(json.dumps({"patches": data.get("patches", [])}, ensure_ascii=False))
    rows = data.get("reflows", [])
    if not isinstance(rows, list):
        raise LLMTermReviewError("LLM JSON reflows must be an array")

    reflows: list[ReflowSuggestion] = []
    seen: set[tuple[tuple[int, ...], tuple[str, ...]]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        cue_ids_raw = row.get("cue_ids", [])
        segments_raw = row.get("segments", [])
        if not isinstance(cue_ids_raw, list) or not isinstance(segments_raw, list):
            continue
        try:
            cue_ids = tuple(int(value) for value in cue_ids_raw)
        except (TypeError, ValueError):
            continue
        cue_ids = tuple(dict.fromkeys(cue_ids))
        segments = tuple(str(segment).strip().replace("\n", " ") for segment in segments_raw if str(segment).strip())
        if len(cue_ids) < 2 or not segments:
            continue
        if tuple(sorted(cue_ids)) != cue_ids:
            continue
        if any(right != left + 1 for left, right in zip(cue_ids, cue_ids[1:])):
            continue
        key = (cue_ids, segments)
        if key in seen:
            continue
        seen.add(key)
        reflows.append(
            ReflowSuggestion(
                cue_ids=cue_ids,
                segments=segments,
                reason=str(row.get("reason", ""))[:300],
                confidence=str(row.get("confidence", "medium")).lower(),
            )
        )
    return patches, reflows[:10]


def apply_patches(cues: list[Cue], patches: list[CuePatch], allowed_cues: set[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cue_by_number: dict[int, Cue] = {}
    for cue in cues:
        try:
            cue_by_number[int(cue.index)] = cue
        except ValueError:
            continue
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    per_cue_count: dict[int, int] = {}
    for patch in patches:
        if patch.confidence not in {"high", "medium"}:
            rejected.append({**patch.__dict__, "reason_rejected": "low_confidence"})
            continue
        if patch.cue not in allowed_cues:
            rejected.append({**patch.__dict__, "reason_rejected": "outside_edit_range"})
            continue
        cue = cue_by_number.get(patch.cue)
        if cue is None:
            rejected.append({**patch.__dict__, "reason_rejected": "cue_not_found"})
            continue
        if per_cue_count.get(patch.cue, 0) >= 5:
            rejected.append({**patch.__dict__, "reason_rejected": "too_many_patches_for_cue"})
            continue
        before = cue_text(cue)
        if patch.old not in before:
            rejected.append({**patch.__dict__, "reason_rejected": "old_text_not_found"})
            continue
        after = before.replace(patch.old, patch.new, 1)
        cue.lines = [after]
        per_cue_count[patch.cue] = per_cue_count.get(patch.cue, 0) + 1
        applied.append({
            **patch.__dict__,
            "before": before,
            "after": after,
        })
    return applied, rejected


UNIT_REVIEW_CONTEXT_RE = re.compile(
    r"(耐药率|敏感率|检出率|死亡率|发生率|有效率|比例|占到|达到|以内|不到|高于|低于|超过|小于|大于|左右)"
)
BARE_PERCENT_NUMBER_RE = re.compile(r"(?<![\d.])([1-9]?\d|100)(?![\d.%％])")
UNIT_REVIEW_SKIP_AFTER_RE = re.compile(r"^(?:年|月|日|例|种|位|个|天|小时|分钟|秒|岁|次|毫克|mg|g|kg|ml|l|mmhg|mmol|μmol|umol|q\d|h)")


def needs_unit_review_window(cues: list[Cue], position: int) -> bool:
    text = cue_text(cues[position])
    if not BARE_PERCENT_NUMBER_RE.search(text):
        return False
    for match in BARE_PERCENT_NUMBER_RE.finditer(text):
        after = text[match.end() : match.end() + 8].strip().lower()
        number = int(match.group(1))
        if number > 100 or UNIT_REVIEW_SKIP_AFTER_RE.match(after):
            continue
        start_idx = max(0, position - 2)
        end_idx = min(len(cues), position + 3)
        context = " ".join(cue_text(cue) for cue in cues[start_idx:end_idx])
        if UNIT_REVIEW_CONTEXT_RE.search(context):
            return True
    return False


def build_unit_review_messages(cues: list[Cue], position: int, terms_summary: str, system_prompt: str = "") -> list[dict[str, str]]:
    system = system_prompt.strip() or (
        "你是医疗课程字幕数值单位复核助手。只输出 JSON。"
        "只检查当前字幕是否漏写明确单位，例如百分号%、mg、mmHg、q12h。"
    )
    start_idx = max(0, position - 2)
    end_idx = min(len(cues), position + 3)
    rows: list[str] = []
    for idx in range(start_idx, end_idx):
        marker = "EDIT" if idx == position else "CTX"
        rows.append(f"[{marker}] {cues[idx].index}: {cue_text(cues[idx])}")
    user = f"""
请只复核标记为 EDIT 的字幕是否漏写数值单位。CTX 只作为上下文。

用户确认课程术语表：
{terms_summary}

输出 JSON：
{{
  "patches": [
    {{
      "cue": {cues[position].index},
      "old": "字幕中的原文片段",
      "new": "补充单位后的片段",
      "reason": "为什么上下文能确定缺少该单位",
      "confidence": "high|medium|low"
    }}
  ]
}}

规则：
- 只能修改 EDIT 这一条字幕。
- old 必须是 EDIT 字幕中连续出现的原文片段。
- 只在上下文明确显示是百分比、剂量、浓度、血压或给药频次时补单位。
- 不要把年份、数量、序号、课程编号、药物种类数误改成百分比。
- 不能改写句子，不能补不存在的数字，不能删除文字。
- 没有高把握就返回空数组。

字幕：
{chr(10).join(rows)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def review_numeric_unit_windows(
    cues: list[Cue],
    terms_summary: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    system_prompt: str = "",
    use_llm: bool = True,
    fallback_on_llm_error: bool = True,
    max_windows: int = 40,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not use_llm or not api_key:
        return [], []
    candidate_positions = [
        idx for idx in range(len(cues)) if needs_unit_review_window(cues, idx)
    ][:max_windows]
    if progress_callback:
        progress_callback({
            "status": "planned",
            "message": "numeric unit review windows planned",
            "window_count": len(candidate_positions),
        })
    applied_all: list[dict[str, Any]] = []
    rejected_all: list[dict[str, Any]] = []
    for window_index, position in enumerate(candidate_positions, start=1):
        cue = cues[position]
        if progress_callback:
            progress_callback({
                "status": "progress",
                "message": "reviewing numeric unit window",
                "current": window_index,
                "total": len(candidate_positions),
                "cue": cue.index,
                "percent": round(window_index * 100 / max(1, len(candidate_positions))),
            })
        try:
            raw = chat_completion(
                build_unit_review_messages(cues, position, terms_summary, system_prompt=system_prompt),
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            patches = normalize_patches(raw)
            allowed = {int(cue.index)} if str(cue.index).isdigit() else set()
            applied, rejected = apply_patches(cues, patches, allowed)
            applied_all.extend(applied)
            rejected_all.extend(rejected)
        except Exception as exc:
            rejected_all.append({
                "cue": cue.index,
                "timing": cue.timing,
                "before": cue_text(cue),
                "reason_rejected": "unit_review_error",
                "error": str(exc),
            })
            if not fallback_on_llm_error:
                raise
    return applied_all, rejected_all


def _cue_start_end(cue: Cue) -> tuple[str, str]:
    start, end = cue.timing.split(" --> ")
    return start.strip(), end.strip()


def _ts_to_ms(ts: str) -> int:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def _segments_preserve_window_text(original_segments: list[str], candidate_segments: list[str]) -> bool:
    original = canonical_content("".join(original_segments), "，,")
    candidate = canonical_content("".join(candidate_segments), "，,")
    return bool(original) and original == candidate


def apply_reflows(
    cues: list[Cue],
    reflows: list[ReflowSuggestion],
    allowed_cues: set[int],
    min_chars: int,
    max_chars: int,
) -> tuple[list[Cue], list[dict[str, Any]], list[dict[str, Any]]]:
    cue_positions: dict[int, int] = {}
    for idx, cue in enumerate(cues):
        try:
            cue_positions[int(cue.index)] = idx
        except ValueError:
            continue

    accepted: list[tuple[int, int, ReflowSuggestion, list[str], list[str]]] = []
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_positions: set[int] = set()

    for reflow in sorted(reflows, key=lambda item: item.cue_ids[0]):
        payload = {
            "cue_ids": list(reflow.cue_ids),
            "segments": list(reflow.segments),
            "reason": reflow.reason,
            "confidence": reflow.confidence,
        }
        if reflow.confidence not in {"high", "medium"}:
            rejected.append({**payload, "reason_rejected": "low_confidence"})
            continue
        if any(cue_id not in allowed_cues for cue_id in reflow.cue_ids):
            rejected.append({**payload, "reason_rejected": "outside_edit_range"})
            continue
        if any(cue_id not in cue_positions for cue_id in reflow.cue_ids):
            rejected.append({**payload, "reason_rejected": "cue_not_found"})
            continue
        positions = [cue_positions[cue_id] for cue_id in reflow.cue_ids]
        if any(right != left + 1 for left, right in zip(positions, positions[1:])):
            rejected.append({**payload, "reason_rejected": "non_contiguous_cues"})
            continue
        if any(position in used_positions for position in positions):
            rejected.append({**payload, "reason_rejected": "overlaps_previous_reflow"})
            continue

        start_idx = positions[0]
        end_idx = positions[-1]
        before_texts = [cue_text(cue) for cue in cues[start_idx : end_idx + 1]]
        segments = [str(segment).strip().replace("\n", " ") for segment in reflow.segments if str(segment).strip()]
        if not _segments_preserve_window_text(before_texts, segments):
            rejected.append({**payload, "before": before_texts, "reason_rejected": "content_not_preserved"})
            continue
        segments = repair_protected_term_boundaries(segments)
        segments = merge_short_segments(segments, min_chars=min_chars, max_chars=max_chars)
        if not _segments_preserve_window_text(before_texts, segments):
            rejected.append({**payload, "before": before_texts, "reason_rejected": "postprocess_changed_content"})
            continue

        for position in positions:
            used_positions.add(position)
        accepted.append((start_idx, end_idx, reflow, before_texts, segments))
        applied.append({
            **payload,
            "before": before_texts,
            "after": segments,
            "segment_lengths": [visible_len(segment) for segment in segments],
        })

    if not accepted:
        return cues, applied, rejected

    rebuilt: list[Cue] = []
    cursor = 0
    for start_idx, end_idx, _reflow, _before_texts, segments in accepted:
        while cursor < start_idx:
            rebuilt.append(cues[cursor])
            cursor += 1
        start_ts, _ = _cue_start_end(cues[start_idx])
        _, end_ts = _cue_start_end(cues[end_idx])
        timings = interpolate_timestamps(start_ts, end_ts, segments)
        for segment, (segment_start, segment_end) in zip(segments, timings):
            rebuilt.append(Cue(index="", timing=f"{segment_start} --> {segment_end}", lines=[segment]))
        cursor = end_idx + 1
    while cursor < len(cues):
        rebuilt.append(cues[cursor])
        cursor += 1

    for index, cue in enumerate(rebuilt, start=1):
        cue.index = str(index)
    return rebuilt, applied, rejected


def split_residual_overlong_cues(
    cues: list[Cue],
    max_chars: int,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    use_llm: bool = True,
    fallback_on_llm_error: bool = True,
) -> tuple[list[Cue], list[dict[str, Any]], list[dict[str, Any]]]:
    rebuilt: list[Cue] = []
    changes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for position, cue in enumerate(cues):
        text = cue_text(cue)
        if visible_len(text) <= max_chars:
            rebuilt.append(cue)
            continue

        method = "local_fallback"
        error_text = ""
        prev_text = cue_text(cues[position - 1]) if position > 0 else ""
        next_text = cue_text(cues[position + 1]) if position + 1 < len(cues) else ""
        if use_llm and api_key:
            try:
                raw = chat_completion(
                    build_overlong_split_messages(prev_text, text, next_text, max_chars, allow_neighbor_rewrite=False),
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout=timeout,
                )
                segments = normalize_segments_payload(raw, text)
                if not _segments_preserve_window_text([text], segments):
                    raise LLMTermReviewError("LLM residual overlong split changed, added, or removed subtitle text")
                segments = enforce_max_chars(repair_protected_term_boundaries(segments), max_chars)
                method = "llm"
            except Exception as exc:
                if not fallback_on_llm_error:
                    raise
                error_text = str(exc)
                errors.append({
                    "cue": cue.index,
                    "timing": cue.timing,
                    "error": error_text,
                    "fallback": "local_split",
                })
                segments = enforce_max_chars(repair_protected_term_boundaries(greedy_split(text, max_chars)), max_chars)
        else:
            segments = enforce_max_chars(repair_protected_term_boundaries(greedy_split(text, max_chars)), max_chars)

        if len(segments) <= 1:
            rebuilt.append(cue)
            continue
        start_ts, end_ts = _cue_start_end(cue)
        timings = interpolate_timestamps(start_ts, end_ts, segments)
        changes.append({
            "cue": cue.index,
            "timing": cue.timing,
            "before": text,
            "after": segments,
            "segment_lengths": [visible_len(segment) for segment in segments],
            "method": method,
            **({"error": error_text} if error_text else {}),
        })
        for segment, (segment_start, segment_end) in zip(segments, timings):
            rebuilt.append(Cue(index="", timing=f"{segment_start} --> {segment_end}", lines=[segment]))

    if changes:
        for index, cue in enumerate(rebuilt, start=1):
            cue.index = str(index)
    return rebuilt, changes, errors


def clean_final_cues(
    cues: list[Cue],
    punctuation: str,
    fillers: tuple[str, ...],
    comma_as_space: bool,
) -> tuple[list[Cue], list[dict[str, Any]]]:
    cleaned: list[Cue] = []
    changes: list[dict[str, Any]] = []
    for cue in cues:
        before = cue_text(cue)
        after = clean_subtitle_text(before, punctuation, fillers, comma_as_space=comma_as_space)
        cleaned.append(Cue(index=cue.index, timing=cue.timing, lines=[after] if after else [""]))
        if before != after:
            changes.append({
                "cue": cue.index,
                "timing": cue.timing,
                "before": before,
                "after": after,
            })
    return cleaned, changes


SYMBOL_END_RE = re.compile(r"[^\w\u4e00-\u9fff）)]$")


def has_symbol_ending(text: str) -> bool:
    return bool(SYMBOL_END_RE.search(text.strip()))


def repair_symbol_ending_windows(
    cues: list[Cue],
    min_chars: int,
    max_chars: int,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    punctuation: str,
    fillers: tuple[str, ...],
    comma_as_space: bool,
    use_llm: bool = True,
    fallback_on_llm_error: bool = True,
) -> tuple[list[Cue], list[dict[str, Any]], list[dict[str, Any]]]:
    if not use_llm or not api_key:
        return cues, [], []

    accepted: list[tuple[int, int, list[str], list[str]]] = []
    changes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    used_positions: set[int] = set()
    issue_positions = [idx for idx, cue in enumerate(cues) if has_symbol_ending(cue_text(cue))]

    for issue_idx in issue_positions:
        start_idx = max(0, issue_idx - 2)
        end_idx = min(len(cues) - 1, issue_idx + 2)
        positions = set(range(start_idx, end_idx + 1))
        if positions & used_positions:
            continue
        window_cues = cues[start_idx : end_idx + 1]
        before_texts = [cue_text(cue) for cue in window_cues]
        try:
            raw = chat_completion(
                build_short_window_messages(
                    [(int(cue.index) if str(cue.index).isdigit() else start_idx + offset + 1, cue_text(cue)) for offset, cue in enumerate(window_cues)],
                    min_chars,
                    max_chars,
                ),
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            segments = normalize_segments_payload(raw, " ".join(before_texts))
            segments = [
                cleaned
                for cleaned in (
                    clean_subtitle_text(segment, punctuation, fillers, comma_as_space=comma_as_space)
                    for segment in segments
                )
                if cleaned
            ]
            segments = repair_hanging_value_connectors(repair_protected_term_boundaries(segments), max_chars)
            segments = merge_short_segments(segments, min_chars=min_chars, max_chars=max_chars)
            if not segments or not _segments_preserve_window_text(before_texts, segments):
                raise LLMTermReviewError("LLM symbol-ending window output changed, added, or removed subtitle text")
        except Exception as exc:
            errors.append({
                "cue": cues[issue_idx].index,
                "timing": cues[issue_idx].timing,
                "window_start_cue": window_cues[0].index,
                "window_end_cue": window_cues[-1].index,
                "error": str(exc),
                "fallback": "keep_window",
            })
            if not fallback_on_llm_error:
                raise
            continue

        if segments == before_texts:
            continue
        used_positions.update(positions)
        accepted.append((start_idx, end_idx, before_texts, segments))
        changes.append({
            "cue": cues[issue_idx].index,
            "timing": cues[issue_idx].timing,
            "window_start_cue": window_cues[0].index,
            "window_end_cue": window_cues[-1].index,
            "before": before_texts,
            "after": segments,
            "segment_lengths": [visible_len(segment) for segment in segments],
        })

    if not accepted:
        return cues, changes, errors

    rebuilt: list[Cue] = []
    cursor = 0
    for start_idx, end_idx, _before_texts, segments in accepted:
        while cursor < start_idx:
            rebuilt.append(cues[cursor])
            cursor += 1
        start_ts, _ = _cue_start_end(cues[start_idx])
        _, end_ts = _cue_start_end(cues[end_idx])
        timings = interpolate_timestamps(start_ts, end_ts, segments)
        for segment, (segment_start, segment_end) in zip(segments, timings):
            rebuilt.append(Cue(index="", timing=f"{segment_start} --> {segment_end}", lines=[segment]))
        cursor = end_idx + 1
    while cursor < len(cues):
        rebuilt.append(cues[cursor])
        cursor += 1

    for number, cue in enumerate(rebuilt, start=1):
        cue.index = str(number)
    return rebuilt, changes, errors


def merge_residual_short_cues(
    cues: list[Cue],
    min_chars: int,
    max_chars: int,
    max_gap_ms: int = 700,
) -> tuple[list[Cue], list[dict[str, Any]]]:
    rebuilt = list(cues)
    changes: list[dict[str, Any]] = []
    index = 0
    while index < len(rebuilt):
        cue = rebuilt[index]
        text = cue_text(cue)
        if not text or visible_len(text) >= min_chars:
            index += 1
            continue

        start_ts, end_ts = _cue_start_end(cue)
        candidates: list[tuple[int, int, str, str, str, str]] = []
        if index > 0:
            prev = rebuilt[index - 1]
            prev_start, prev_end = _cue_start_end(prev)
            gap = _ts_to_ms(start_ts) - _ts_to_ms(prev_end)
            combined = f"{cue_text(prev)}{text}"
            if 0 <= gap <= max_gap_ms and visible_len(combined) <= max_chars:
                candidates.append((gap, 0, "previous", combined, prev_start, end_ts))
        if index + 1 < len(rebuilt):
            next_cue = rebuilt[index + 1]
            next_start, next_end = _cue_start_end(next_cue)
            gap = _ts_to_ms(next_start) - _ts_to_ms(end_ts)
            combined = f"{text}{cue_text(next_cue)}"
            prefer_next = -1 if visible_len(text) <= 1 else 1
            if 0 <= gap <= max_gap_ms and visible_len(combined) <= max_chars:
                candidates.append((gap, prefer_next, "next", combined, start_ts, next_end))
        if not candidates:
            index += 1
            continue

        _gap, _preference, direction, combined_text, merged_start, merged_end = min(candidates, key=lambda item: (item[0], item[1]))
        if direction == "previous":
            prev = rebuilt[index - 1]
            before = [cue_text(prev), text]
            rebuilt[index - 1] = Cue(index="", timing=f"{merged_start} --> {merged_end}", lines=[combined_text])
            del rebuilt[index]
            changed_position = max(0, index - 1)
        else:
            next_cue = rebuilt[index + 1]
            before = [text, cue_text(next_cue)]
            rebuilt[index] = Cue(index="", timing=f"{merged_start} --> {merged_end}", lines=[combined_text])
            del rebuilt[index + 1]
            changed_position = index
        changes.append({
            "cue_position": index + 1,
            "direction": direction,
            "before": before,
            "after": combined_text,
            "timing": f"{merged_start} --> {merged_end}",
            "length": visible_len(combined_text),
        })
        index = changed_position

    if changes:
        for number, cue in enumerate(rebuilt, start=1):
            cue.index = str(number)
    return rebuilt, changes


def review_srt_terms(
    srt: Path,
    output: Path,
    report_path: Path,
    terms: Path | None,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str = "",
    batch_size: int = 100,
    overlap: int = 5,
    min_chars: int = 5,
    max_chars: int = 20,
    timeout: int = 45,
    use_llm: bool = True,
    fallback_on_llm_error: bool = True,
    punctuation: str = "",
    fillers: tuple[str, ...] = (),
    comma_as_space: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cues = parse_srt(srt)
    batches = make_batches(cues, batch_size=batch_size, overlap=overlap)
    terms_summary = load_terms_summary(terms)
    applied_all: list[dict[str, Any]] = []
    rejected_all: list[dict[str, Any]] = []
    pending_reflows: list[ReflowSuggestion] = []
    applied_reflows: list[dict[str, Any]] = []
    rejected_reflows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if progress_callback:
        progress_callback({
            "status": "planned",
            "message": "LLM terminology review batches planned",
            "batch_count": len(batches),
            "cue_count": len(cues),
            "batch_size": batch_size,
            "overlap": overlap,
            "min_chars": min_chars,
            "max_chars": max_chars,
        })

    for batch_index, (start, end, context_start, context_end) in enumerate(batches, start=1):
        if progress_callback:
            progress_callback({
                "status": "progress",
                "message": "reviewing terminology batch",
                "current": batch_index,
                "total": len(batches),
                "cue_start": cues[start].index,
                "cue_end": cues[end - 1].index,
                "percent": round(batch_index * 100 / max(1, len(batches))),
            })
        if not use_llm:
            continue
        try:
            raw = chat_completion(
                build_messages(cues, start, end, context_start, context_end, terms_summary, min_chars, max_chars, system_prompt),
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            allowed = {int(cues[idx].index) for idx in range(start, end) if str(cues[idx].index).isdigit()}
            patches, reflows = normalize_review_payload(raw)
            applied, rejected = apply_patches(cues, patches, allowed)
            applied_all.extend(applied)
            rejected_all.extend(rejected)
            for reflow in reflows:
                if any(cue_id not in allowed for cue_id in reflow.cue_ids):
                    rejected_reflows.append({
                        "cue_ids": list(reflow.cue_ids),
                        "segments": list(reflow.segments),
                        "reason": reflow.reason,
                        "confidence": reflow.confidence,
                        "reason_rejected": "outside_batch_edit_range",
                    })
                    continue
                pending_reflows.append(reflow)
        except Exception as exc:
            error = {
                "batch": batch_index,
                "cue_start": cues[start].index,
                "cue_end": cues[end - 1].index,
                "error": str(exc),
            }
            errors.append(error)
            if progress_callback:
                progress_callback({
                    "status": "fallback",
                    "message": "LLM terminology review failed; keeping batch unchanged",
                    **error,
                })
            if not fallback_on_llm_error:
                raise

    all_cue_ids = {int(cue.index) for cue in cues if str(cue.index).isdigit()}
    cues, applied_reflows, post_rejected_reflows = apply_reflows(
        cues,
        pending_reflows,
        allowed_cues=all_cue_ids,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    rejected_reflows.extend(post_rejected_reflows)
    numeric_unit_patches, numeric_unit_rejected = review_numeric_unit_windows(
        cues,
        terms_summary=terms_summary,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        system_prompt=system_prompt,
        use_llm=use_llm,
        fallback_on_llm_error=fallback_on_llm_error,
        progress_callback=progress_callback,
    )
    applied_all.extend(numeric_unit_patches)
    rejected_all.extend(numeric_unit_rejected)

    cues, residual_overlong_splits, residual_overlong_errors = split_residual_overlong_cues(
        cues,
        max_chars=max_chars,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        use_llm=use_llm,
        fallback_on_llm_error=fallback_on_llm_error,
    )
    errors.extend(residual_overlong_errors)
    cues, final_cleanup_changes = clean_final_cues(
        cues,
        punctuation=punctuation,
        fillers=fillers,
        comma_as_space=comma_as_space,
    )
    cues, hanging_value_merges = merge_hanging_value_connector_cues(cues, max_chars=max_chars)
    cues, symbol_ending_repairs, symbol_ending_errors = repair_symbol_ending_windows(
        cues,
        min_chars=min_chars,
        max_chars=max_chars,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        punctuation=punctuation,
        fillers=fillers,
        comma_as_space=comma_as_space,
        use_llm=use_llm,
        fallback_on_llm_error=fallback_on_llm_error,
    )
    errors.extend(symbol_ending_errors)
    cues, post_symbol_cleanup_changes = clean_final_cues(
        cues,
        punctuation=punctuation,
        fillers=fillers,
        comma_as_space=comma_as_space,
    )
    final_cleanup_changes.extend(post_symbol_cleanup_changes)
    cues, residual_short_merges = merge_residual_short_cues(
        cues,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    residual_short_count = sum(1 for cue in cues if 0 < visible_len(cue_text(cue)) < min_chars)
    residual_overlong_count = sum(1 for cue in cues if visible_len(cue_text(cue)) > max_chars)

    write_srt(cues, output)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_srt": str(srt),
        "output_srt": str(output),
        "terms": str(terms) if terms else None,
        "cue_count": len(cues),
        "batch_count": len(batches),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "removed_punctuation": punctuation,
        "removed_fillers": list(fillers),
        "comma_as_space": comma_as_space,
        "applied_patch_count": len(applied_all),
        "rejected_patch_count": len(rejected_all),
        "applied_reflow_count": len(applied_reflows),
        "rejected_reflow_count": len(rejected_reflows),
        "numeric_unit_patch_count": len(numeric_unit_patches),
        "numeric_unit_rejected_count": len(numeric_unit_rejected),
        "residual_short_count": residual_short_count,
        "residual_overlong_count": residual_overlong_count,
        "hanging_value_merge_count": len(hanging_value_merges),
        "symbol_ending_repair_count": len(symbol_ending_repairs),
        "symbol_ending_error_count": len(symbol_ending_errors),
        "residual_short_merge_count": len(residual_short_merges),
        "final_cleanup_changed_cue_count": len(final_cleanup_changes),
        "error_count": len(errors),
        "applied_patches": applied_all,
        "rejected_patches": rejected_all,
        "numeric_unit_patches": numeric_unit_patches,
        "numeric_unit_rejected": numeric_unit_rejected,
        "applied_reflows": applied_reflows,
        "rejected_reflows": rejected_reflows,
        "residual_overlong_splits": residual_overlong_splits,
        "residual_overlong_error_count": len(residual_overlong_errors),
        "hanging_value_merges": hanging_value_merges,
        "symbol_ending_repairs": symbol_ending_repairs,
        "symbol_ending_errors": symbol_ending_errors,
        "residual_short_merges": residual_short_merges,
        "final_cleanup_changes": final_cleanup_changes,
        "errors": errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback({
            "status": "done",
            "message": "LLM terminology review complete",
            "applied_patch_count": len(applied_all),
            "rejected_patch_count": len(rejected_all),
            "applied_reflow_count": len(applied_reflows),
            "rejected_reflow_count": len(rejected_reflows),
            "numeric_unit_patch_count": len(numeric_unit_patches),
            "numeric_unit_rejected_count": len(numeric_unit_rejected),
            "residual_overlong_split_count": len(residual_overlong_splits),
            "residual_overlong_error_count": len(residual_overlong_errors),
            "hanging_value_merge_count": len(hanging_value_merges),
            "symbol_ending_repair_count": len(symbol_ending_repairs),
            "symbol_ending_error_count": len(symbol_ending_errors),
            "residual_short_merge_count": len(residual_short_merges),
            "final_cleanup_changed_cue_count": len(final_cleanup_changes),
            "residual_short_count": residual_short_count,
            "residual_overlong_count": residual_overlong_count,
            "error_count": len(errors),
            "output_srt": str(output),
            "report": str(report_path),
        })
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch LLM terminology review for SRT files.")
    parser.add_argument("--srt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--terms")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--remove-punctuation", default="")
    parser.add_argument(
        "--remove-fillers",
        default="",
        help="Comma/space separated filler words to remove during final cleanup.",
    )
    parser.add_argument("--comma-as-space", dest="comma_as_space", action="store_true", default=True)
    parser.add_argument("--no-comma-as-space", dest="comma_as_space", action="store_false")
    parser.add_argument("--no-llm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or ("local" if is_local_base_url(args.base_url) else "")
    if not api_key and not args.no_llm:
        raise LLMTermReviewError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    report = review_srt_terms(
        srt=Path(args.srt).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        report_path=Path(args.report).expanduser().resolve(),
        terms=Path(args.terms).expanduser().resolve() if args.terms else None,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
        overlap=args.overlap,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        timeout=args.timeout,
        use_llm=not args.no_llm,
        punctuation=args.remove_punctuation,
        fillers=tuple(word for word in re.split(r"[,，、\s]+", args.remove_fillers.strip()) if word),
        comma_as_space=args.comma_as_space,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
