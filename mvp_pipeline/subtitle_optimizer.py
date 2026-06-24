#!/usr/bin/env python3
"""Clean punctuation and split overlong SRT cues into independent short cues via LLM."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from term_corrector import (
    Cue,
    builtin_medical_term_normalizations,
    builtin_unit_normalizations,
    parse_srt,
    write_srt,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = ROOT / "work"
DEFAULT_REMOVE_PUNCTUATION = "，,"
DEFAULT_REMOVE_FILLER_WORDS = ("嗯", "呃", "啊")
DEFAULT_REMOVE_FILLERS_TEXT = ",".join(DEFAULT_REMOVE_FILLER_WORDS)
PROTECTED_BOUNDARY_TERMS = (
    "高血压",
    "急症",
    "亚急症",
    "靶器官",
    "硝普钠",
    "硝苯地平",
    "美托洛尔",
    "多巴胺",
    "冠脉",
    "综合征",
    "主动脉",
    "夹层",
    "脑出血",
)


class SubtitleOptimizerError(RuntimeError):
    pass


# ── timestamp helpers ──────────────────────────────────────────────────────────

def _ts_to_ms(ts: str) -> int:
    """Convert SRT timestamp 'HH:MM:SS,mmm' to milliseconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def _ms_to_ts(ms: int) -> str:
    """Convert milliseconds to SRT timestamp 'HH:MM:SS,mmm'."""
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_remainder = ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms_remainder:03}"


def interpolate_timestamps(start_ts: str, end_ts: str, segments: list[str]) -> list[tuple[str, str]]:
    """Distribute original time range across segments proportional to text length."""
    start_ms = _ts_to_ms(start_ts)
    end_ms = _ts_to_ms(end_ts)
    total_ms = end_ms - start_ms
    if total_ms <= 0 or not segments:
        return [(start_ts, end_ts)]

    lengths = [max(1, visible_len(s)) for s in segments]
    total_len = sum(lengths)
    if total_len == 0:
        return [(start_ts, end_ts)]

    timings: list[tuple[str, str]] = []
    cursor = start_ms
    min_ms = 500 if total_ms >= len(segments) * 500 else max(1, total_ms // len(segments))
    for i, _seg in enumerate(segments):
        fraction = lengths[i] / total_len
        seg_duration = max(min_ms, int(round(fraction * total_ms)))
        if i == len(segments) - 1:
            seg_end = end_ms
        else:
            latest_end = end_ms - (len(segments) - i - 1) * min_ms
            seg_end = min(cursor + seg_duration, latest_end)
            seg_end = max(cursor + 1, seg_end)
        timings.append((_ms_to_ts(cursor), _ms_to_ts(seg_end)))
        cursor = seg_end
    return timings


def visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def clean_punctuation(text: str, punctuation: str) -> str:
    if not punctuation:
        return text
    table = str.maketrans("", "", punctuation)
    cleaned = text.translate(table)
    cleaned = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n\s+", "\n", cleaned)
    return cleaned.strip()


def clean_filler_words(text: str, fillers: tuple[str, ...] = DEFAULT_REMOVE_FILLER_WORDS) -> str:
    cleaned = text
    for filler in fillers:
        cleaned = cleaned.replace(filler, "")
    return cleaned.strip()


def parse_filler_words(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_REMOVE_FILLER_WORDS
    words: list[str] = []
    seen: set[str] = set()
    for word in re.split(r"[,，、\s]+", value.strip()):
        word = word.strip()
        if word and word not in seen:
            seen.add(word)
            words.append(word)
    return tuple(words)


def clean_subtitle_text(text: str, punctuation: str, fillers: tuple[str, ...] = DEFAULT_REMOVE_FILLER_WORDS) -> str:
    cleaned = clean_filler_words(clean_punctuation(text, punctuation), fillers)
    cleaned, _term_changes = builtin_medical_term_normalizations(cleaned)
    cleaned, _unit_changes = builtin_unit_normalizations(cleaned)
    return cleaned.strip()


_VALIDATION_IGNORED_PUNCTUATION = "，,。.;；:：!?！？、"


def canonical_content(text: str, punctuation: str = "") -> str:
    ignored = "".join(dict.fromkeys(f"{punctuation}{_VALIDATION_IGNORED_PUNCTUATION}"))
    if ignored:
        text = text.translate(str.maketrans("", "", ignored))
    return re.sub(r"\s+", "", text)


def segments_preserve_content(original_segments: list[str], candidate_segments: list[str], punctuation: str) -> bool:
    original = canonical_content("".join(original_segments), punctuation)
    candidate = canonical_content("".join(candidate_segments), punctuation)
    return bool(original) and original == candidate


# ── LLM helpers ────────────────────────────────────────────────────────────────

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
        "max_tokens": int(os.environ.get("SUBTITLE_OPTIMIZER_MAX_TOKENS", "2048")),
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
        raise SubtitleOptimizerError(f"LLM request failed: {exc.code} {detail[:1000]}") from exc
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        raise SubtitleOptimizerError(f"LLM request failed: {exc}") from exc
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
            raise SubtitleOptimizerError("LLM returned reasoning content but no final answer; thinking mode should be disabled for JSON tasks.")
        return content
    except SubtitleOptimizerError:
        raise
    except Exception as exc:
        raise SubtitleOptimizerError(f"Unexpected LLM response shape: {data}") from exc


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


def build_messages(
    prev_text: str,
    current_text: str,
    next_text: str,
    max_chars: int,
    allow_neighbor_rewrite: bool,
) -> list[dict[str, str]]:
    system = (
        "你是中文字幕切分助手。任务是把一条过长的字幕拆分成多条独立短字幕，"
        "每条都是独立的时间轴cue。按语义切分，不要改原文，不新增信息，不删除信息，不输出解释。"
    )
    if allow_neighbor_rewrite:
        user = f"""
请结合前后字幕语境，把"当前字幕"切分成多条独立短字幕（segments）。
每条不超过 {max_chars} 个汉字/字符；医学术语或英文名可略微超过。
必要时可以从相邻字幕借走或归还少量文字以保持语义完整，但不能改变整体内容。

输出 JSON：
{{
  "prev_segments": ["前一条可保持不变"],
  "segments": ["当前第一段", "当前第二段", "当前第三段"],
  "next_segments": ["后一条可保持不变"]
}}

硬性要求：
- segments 里的每条都是独立字幕，不能包含换行符 \\n。
- 不要使用逗号、句号、分号等标点（英文缩写和数字中的标点除外）。
- 保留 TED、TAO、MDT、TRAb、TPOAb、TgAb、FT3、FT4、TSH、131I 等专业写法。
- 如果前后字幕语义完整，原样返回不做改动。
- 时间戳由系统自动分配，你只负责输出文字。

前一条字幕：
{prev_text or "[无]"}

当前字幕：
{current_text}

后一条字幕：
{next_text or "[无]"}
""".strip()
    else:
        user = f"""
请把"当前字幕"切分成多条独立短字幕（segments）。前后字幕仅作为语境参考，不要修改前后字幕。

每条不超过 {max_chars} 个汉字/字符；医学术语或英文名可略微超过。

输出 JSON：
{{
  "segments": ["第一段", "第二段", "第三段"]
}}

硬性要求：
- segments 里的每条都是独立字幕，不能包含换行符 \\n。
- 不要使用逗号、句号、分号等标点（英文缩写和数字中的标点除外）。
- 保留 TED、TAO、MDT、TRAb、TPOAb、TgAb、FT3、FT4、TSH、131I 等专业写法。
- 不要从前后字幕借文字过来。
- 时间戳由系统自动分配，你只负责输出文字。

前一条字幕：
{prev_text or "[无]"}

当前字幕：
{current_text}

后一条字幕：
{next_text or "[无]"}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_segments_payload(raw: str, fallback_text: str) -> list[str]:
    try:
        payload = json.loads(extract_json_object(raw))
    except json.JSONDecodeError as exc:
        raise SubtitleOptimizerError(f"LLM did not return valid JSON: {raw[:1000]}") from exc
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise SubtitleOptimizerError("LLM JSON must contain a segments array")
    cleaned = [str(s).strip().replace("\n", " ") for s in segments if str(s).strip()]
    if not cleaned:
        return [fallback_text.replace("\n", " ")]
    return cleaned


def split_preserving_english_spaces(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if visible_len(text) <= max_chars:
        return [text] if text else []

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+./-]*|[\u4e00-\u9fff]|[^\s]", text)
    total_len = sum(max(1, visible_len(token)) for token in tokens)
    segment_count = max(1, (total_len + max_chars - 1) // max_chars)
    target_len = max(1, (total_len + segment_count - 1) // segment_count)
    segments: list[str] = []
    current = ""
    current_len = 0
    for token in tokens:
        token_len = max(1, visible_len(token))
        joiner = " " if current and re.match(r"^[A-Za-z0-9]", token) and re.search(r"[A-Za-z0-9+./-]$", current) else ""
        candidate = f"{current}{joiner}{token}" if current else token
        if current and current_len + token_len > target_len and len(segments) < segment_count - 1:
            segments.append(current)
            current = token
            current_len = token_len
        else:
            current = candidate
            current_len += token_len
    if current:
        segments.append(current)
    return segments


def enforce_max_chars(segments: list[str], max_chars: int) -> list[str]:
    enforced: list[str] = []
    for segment in segments:
        if visible_len(segment) <= max_chars:
            enforced.append(segment)
        else:
            enforced.extend(split_preserving_english_spaces(segment, max_chars))
    return [segment for segment in enforced if segment]


def repair_protected_term_boundaries(segments: list[str]) -> list[str]:
    repaired = [segment for segment in segments if segment]
    for index in range(len(repaired) - 1):
        left = repaired[index]
        right = repaired[index + 1]
        if not left or not right:
            continue
        joined = left + right
        boundary = len(left)
        for term in PROTECTED_BOUNDARY_TERMS:
            search_from = 0
            while True:
                start = joined.find(term, search_from)
                if start < 0:
                    break
                end = start + len(term)
                if start < boundary < end:
                    move_count = end - boundary
                    repaired[index] = left + right[:move_count]
                    repaired[index + 1] = right[move_count:]
                    left = repaired[index]
                    right = repaired[index + 1]
                    joined = left + right
                    boundary = len(left)
                    break
                search_from = start + 1
    return [segment for segment in repaired if segment]


def join_segment_text(left: str, right: str) -> str:
    joiner = " " if re.search(r"[A-Za-z0-9]$", left) and re.match(r"^[A-Za-z0-9]", right) else ""
    return f"{left}{joiner}{right}"


def merge_short_segments(segments: list[str], min_chars: int, max_chars: int) -> list[str]:
    if not segments:
        return []
    soft_max = max_chars + min_chars

    def merge_once(source: list[str]) -> list[str]:
        merged: list[str] = []
        index = 0
        while index < len(source):
            segment = source[index]
            if visible_len(segment) >= min_chars or len(source) == 1:
                merged.append(segment)
                index += 1
                continue

            if merged:
                candidate = join_segment_text(merged[-1], segment)
                if visible_len(candidate) <= soft_max:
                    merged[-1] = candidate
                    index += 1
                    continue

            if index + 1 < len(source):
                candidate = join_segment_text(segment, source[index + 1])
                if visible_len(candidate) <= soft_max:
                    merged.append(candidate)
                    index += 2
                    continue

            if merged:
                merged[-1] = join_segment_text(merged[-1], segment)
            else:
                merged.append(segment)
            index += 1
        return merged

    current = [segment for segment in segments if segment]
    for _ in range(len(current)):
        updated = merge_once(current)
        if updated == current or not any(0 < visible_len(segment) < min_chars for segment in updated):
            return updated
        current = updated
    return current


def normalize_window_payload(
    raw: str,
    prev_text: str,
    current_text: str,
    next_text: str,
) -> tuple[list[str], list[str], list[str]]:
    try:
        payload = json.loads(extract_json_object(raw))
    except json.JSONDecodeError as exc:
        raise SubtitleOptimizerError(f"LLM did not return valid JSON: {raw[:1000]}") from exc

    def get_segments(key: str, fallback: str) -> list[str]:
        segs = payload.get(key)
        if not isinstance(segs, list):
            return [fallback] if fallback else []
        cleaned = [str(s).strip().replace("\n", " ") for s in segs if str(s).strip()]
        if not cleaned and fallback:
            return [fallback]
        return cleaned

    return (
        get_segments("prev_segments", prev_text),
        get_segments("segments", current_text),
        get_segments("next_segments", next_text),
    )


def build_short_window_messages(window: list[tuple[int, str]], min_chars: int, max_chars: int) -> list[dict[str, str]]:
    system = (
        "你是中文字幕断句助手。任务是把一个局部字幕窗口重新整理成多条独立短字幕，"
        "修复过短字幕和不自然断句。不要新增信息，不删除信息，不输出解释。"
    )
    window_text = "\n".join(f"{cue_id}. {text}" for cue_id, text in window)
    user = f"""
请把下面这个字幕窗口重新断句，输出多条独立字幕 segments。

目标：
- 尽量让每条字幕在 {min_chars}-{max_chars} 个可见字之间。
- 少于 {min_chars} 个字的碎片应优先和前后语义合并。
- 超过 {max_chars} 个字的字幕应按语义拆开。
- 医学缩写、数字单位、人名、药名等可以略微突破长度限制。

输出 JSON：
{{
  "segments": ["第一条字幕", "第二条字幕"]
}}

硬性要求：
- 只能重排窗口内原有文字，不要新增事实，不要删除课程信息。
- 每个 segment 都是独立字幕，不能包含换行符 \\n。
- 不要使用逗号、句号、分号等标点（英文缩写和数字中的标点除外）。
- 保留 TED、TAO、MDT、TRAb、TPOAb、TgAb、FT3、FT4、TSH、131I、mmHg 等专业写法。
- 时间戳由系统自动分配，你只负责输出文字。

字幕窗口：
{window_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_asr_boundary_messages(
    window: list[tuple[int, str]],
    min_chars: int,
    max_chars: int,
    boundary_ts: str,
) -> list[dict[str, str]]:
    system = (
        "你是医疗课程中文字幕断句助手。任务是修复分段 ASR 切点附近的断句问题。"
        "只能重排给定三条字幕里的原始文字，不新增信息，不删除信息，不输出解释。"
    )
    window_text = "\n".join(f"{cue_id}. {text}" for cue_id, text in window)
    user = f"""
下面三条字幕位于 ASR 分段切点 {boundary_ts} 附近，切点可能把一句话截断。
请结合语义把这个小窗口重新断句，输出多条独立字幕 segments。

目标：
- 优先修复切点处上下文被截断、粘连或不自然换句的问题。
- 尽量让每条字幕在 {min_chars}-{max_chars} 个可见字之间。
- 医学缩写、数字单位、人名、药名等可以略微突破长度限制。

输出 JSON：
{{
  "segments": ["第一条字幕", "第二条字幕"]
}}

硬性要求：
- 只能使用窗口内已有文字，不能新增、删除或改写医学事实。
- 每个 segment 都是独立字幕，不能包含换行符 \\n。
- 不要使用逗号、句号、分号等标点（英文缩写和数字中的标点除外）。
- 保留 TED、TAO、MDT、TRAb、TPOAb、TgAb、FT3、FT4、TSH、131I、mmHg 等专业写法。
- 时间戳由系统自动分配，你只负责输出文字。

字幕窗口：
{window_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _cue_text(cue: Cue) -> str:
    return " ".join(cue.lines).strip()


def _cue_start_end(cue: Cue) -> tuple[str, str]:
    start, end = cue.timing.split(" --> ")
    return start.strip(), end.strip()


def _can_merge_windows(
    previous: tuple[int, int],
    current: tuple[int, int],
    cues: list[Cue] | None,
    max_gap_ms: int,
) -> bool:
    if current[0] > previous[1] + 1:
        return False
    if cues is None:
        return True
    previous_end = _ts_to_ms(_cue_start_end(cues[previous[1]])[1])
    current_start = _ts_to_ms(_cue_start_end(cues[current[0]])[0])
    return current_start - previous_end <= max_gap_ms


def _cue_gap_ms(cues: list[Cue], left_idx: int, right_idx: int) -> int:
    left_end = _ts_to_ms(_cue_start_end(cues[left_idx])[1])
    right_start = _ts_to_ms(_cue_start_end(cues[right_idx])[0])
    return right_start - left_end


def _expand_issue_window(
    idx: int,
    cue_count: int,
    radius: int,
    cues: list[Cue] | None,
    max_gap_ms: int,
) -> tuple[int, int]:
    if cues is None:
        return max(0, idx - radius), min(cue_count - 1, idx + radius)

    start = idx
    for _ in range(radius):
        if start <= 0:
            break
        if _cue_gap_ms(cues, start - 1, start) > max_gap_ms:
            break
        start -= 1

    end = idx
    for _ in range(radius):
        if end >= cue_count - 1:
            break
        if _cue_gap_ms(cues, end, end + 1) > max_gap_ms:
            break
        end += 1

    return start, end


def build_issue_windows(
    issue_indices: list[int],
    cue_count: int,
    radius: int = 2,
    cues: list[Cue] | None = None,
    max_gap_ms: int = 700,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for idx in sorted(set(issue_indices)):
        start, end = _expand_issue_window(idx, cue_count, radius, cues, max_gap_ms)
        current = (start, end)
        if windows and _can_merge_windows(windows[-1], current, cues, max_gap_ms):
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append(current)
    return windows


def _window_for_boundary(cues: list[Cue], boundary_ms: int) -> tuple[int, int] | None:
    if len(cues) < 3:
        return None
    current_idx: int | None = None
    for idx, cue in enumerate(cues):
        start_ts, _end_ts = _cue_start_end(cue)
        if _ts_to_ms(start_ts) < boundary_ms:
            current_idx = idx
        else:
            break
    if current_idx is None:
        current_idx = 0
    if current_idx >= len(cues) - 1:
        current_idx = len(cues) - 2
    start_idx = max(0, current_idx - 1)
    end_idx = min(len(cues) - 1, current_idx + 1)
    if end_idx - start_idx + 1 < 3:
        if start_idx == 0:
            end_idx = min(len(cues) - 1, 2)
        elif end_idx == len(cues) - 1:
            start_idx = max(0, len(cues) - 3)
    if end_idx - start_idx + 1 < 2:
        return None
    return start_idx, end_idx


def build_asr_boundary_windows(cues: list[Cue], boundary_ms: list[int]) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for boundary in sorted(set(int(value) for value in boundary_ms if int(value) > 0)):
        window = _window_for_boundary(cues, boundary)
        if window is None or window in seen:
            continue
        seen.add(window)
        windows.append((window[0], window[1], boundary))
    return windows


def repair_asr_boundary_windows(
    cues: list[Cue],
    boundary_ms: list[int],
    min_chars: int,
    max_chars: int,
    punctuation: str,
    fillers: tuple[str, ...],
    model: str,
    base_url: str,
    api_key: str | None,
    use_llm: bool,
    llm_timeout: int,
    fallback_on_llm_error: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[list[Cue], list[dict[str, Any]], list[dict[str, Any]], int]:
    windows = build_asr_boundary_windows(cues, boundary_ms)
    if progress_callback:
        progress_callback({
            "status": "planned",
            "message": "ASR boundary windows identified",
            "boundary_count": len(boundary_ms),
            "boundary_window_count": len(windows),
            "min_chars": min_chars,
            "max_chars": max_chars,
        })
    if not windows or not use_llm:
        return cues, [], [], len(windows)
    if not api_key:
        raise SubtitleOptimizerError("ASR boundary repair requires an API key.")

    rebuilt: list[Cue] = []
    changes: list[dict[str, Any]] = []
    llm_errors: list[dict[str, Any]] = []
    cursor = 0

    for position, (start_idx, end_idx, boundary) in enumerate(windows, start=1):
        while cursor < start_idx:
            rebuilt.append(cues[cursor])
            cursor += 1

        window_cues = cues[start_idx : end_idx + 1]
        before_texts = [_cue_text(cue) for cue in window_cues]
        segs = before_texts
        boundary_ts = _ms_to_ts(boundary)

        if progress_callback:
            progress_callback({
                "status": "progress",
                "message": "repairing ASR boundary window",
                "current": position,
                "total": len(windows),
                "boundary_ms": boundary,
                "boundary": boundary_ts,
                "window_start_cue": window_cues[0].index,
                "window_end_cue": window_cues[-1].index,
                "percent": round(position * 100 / max(1, len(windows))),
            })

        try:
            raw = chat_completion(
                build_asr_boundary_messages(
                    [(int(cue.index or i + 1), _cue_text(cue)) for i, cue in enumerate(window_cues)],
                    min_chars,
                    max_chars,
                    boundary_ts,
                ),
                model=model,
                base_url=base_url,
                api_key=api_key or "",
                timeout=llm_timeout,
            )
            segs = normalize_segments_payload(raw, " ".join(before_texts))
            if not segments_preserve_content(before_texts, segs, punctuation):
                raise SubtitleOptimizerError("LLM ASR boundary output changed, added, or removed subtitle text")
        except Exception as exc:
            if not fallback_on_llm_error:
                raise
            segs = before_texts
            error = {
                "boundary_ms": boundary,
                "boundary": boundary_ts,
                "window_start_cue": window_cues[0].index,
                "window_end_cue": window_cues[-1].index,
                "error": str(exc),
                "fallback": "keep_window",
            }
            llm_errors.append(error)
            if progress_callback:
                progress_callback({
                    "status": "fallback",
                    "message": "LLM ASR boundary repair failed; keeping original window",
                    "current": position,
                    "total": len(windows),
                    "boundary": boundary_ts,
                    "error": str(exc),
                })

        segs = [clean_subtitle_text(seg, punctuation, fillers) for seg in segs if clean_subtitle_text(seg, punctuation, fillers)]
        if not segs:
            segs = before_texts
        segs = enforce_max_chars(segs, max_chars)
        segs = repair_protected_term_boundaries(segs)
        segs = merge_short_segments(segs, min_chars, max_chars)

        start_ts, _ = _cue_start_end(window_cues[0])
        _, end_ts = _cue_start_end(window_cues[-1])
        timings = interpolate_timestamps(start_ts, end_ts, segs)
        for seg, (seg_start, seg_end) in zip(segs, timings):
            rebuilt.append(Cue(index="", timing=f"{seg_start} --> {seg_end}", lines=[seg]))

        if segs != before_texts:
            changes.append({
                "boundary_ms": boundary,
                "boundary": boundary_ts,
                "window_start_cue": window_cues[0].index,
                "window_end_cue": window_cues[-1].index,
                "before": before_texts,
                "after": segs,
                "segment_lengths": [visible_len(s) for s in segs],
            })
        cursor = end_idx + 1

    while cursor < len(cues):
        rebuilt.append(cues[cursor])
        cursor += 1
    return rebuilt, changes, llm_errors, len(windows)


def optimize_short_windows(
    cues: list[Cue],
    min_chars: int,
    max_chars: int,
    punctuation: str,
    fillers: tuple[str, ...],
    model: str,
    base_url: str,
    api_key: str | None,
    use_llm: bool,
    llm_timeout: int,
    max_consecutive_llm_failures: int,
    fallback_on_llm_error: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[list[Cue], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    short_indices = [
        idx
        for idx, cue in enumerate(cues)
        if 0 < visible_len(_cue_text(cue)) < min_chars
    ]
    windows = build_issue_windows(short_indices, len(cues), radius=2, cues=cues)
    if progress_callback:
        progress_callback({
            "status": "planned",
            "message": "short subtitle windows identified",
            "short_detected_count": len(short_indices),
            "short_window_count": len(windows),
            "min_chars": min_chars,
        })

    if not windows or not use_llm:
        return cues, [], [], len(short_indices), len(windows)
    if not api_key:
        raise SubtitleOptimizerError("Short subtitle optimization requires an API key.")

    rebuilt: list[Cue] = []
    short_changes: list[dict[str, Any]] = []
    llm_errors: list[dict[str, Any]] = []
    cursor = 0
    consecutive_llm_failures = 0
    llm_disabled = False

    for position, (start_idx, end_idx) in enumerate(windows, start=1):
        while cursor < start_idx:
            rebuilt.append(cues[cursor])
            cursor += 1

        window_cues = cues[start_idx : end_idx + 1]
        before_texts = [_cue_text(cue) for cue in window_cues]
        segs = before_texts

        if progress_callback:
            progress_callback({
                "status": "progress",
                "message": "optimizing short subtitle window",
                "current": position,
                "total": len(windows),
                "window_start_cue": window_cues[0].index,
                "window_end_cue": window_cues[-1].index,
                "percent": round(position * 100 / max(1, len(windows))),
                "mode": "local_keep" if llm_disabled else "llm",
            })

        if not llm_disabled:
            try:
                raw = chat_completion(
                    build_short_window_messages(
                        [(int(cue.index or i + 1), _cue_text(cue)) for i, cue in enumerate(window_cues)],
                        min_chars,
                        max_chars,
                    ),
                    model=model,
                    base_url=base_url,
                    api_key=api_key or "",
                    timeout=llm_timeout,
                )
                segs = normalize_segments_payload(raw, " ".join(before_texts))
                if not segments_preserve_content(before_texts, segs, punctuation):
                    raise SubtitleOptimizerError("LLM short window output changed, added, or removed subtitle text")
                consecutive_llm_failures = 0
            except Exception as exc:
                if not fallback_on_llm_error:
                    raise
                consecutive_llm_failures += 1
                segs = before_texts
                error = {
                    "window_start_cue": window_cues[0].index,
                    "window_end_cue": window_cues[-1].index,
                    "error": str(exc),
                    "fallback": "keep_window",
                }
                llm_errors.append(error)
                if progress_callback:
                    progress_callback({
                        "status": "fallback",
                        "message": "LLM short subtitle optimization failed; keeping original window",
                        "current": position,
                        "total": len(windows),
                        "error": str(exc),
                    })
                if consecutive_llm_failures >= max_consecutive_llm_failures:
                    llm_disabled = True

        segs = [clean_subtitle_text(seg, punctuation, fillers) for seg in segs if clean_subtitle_text(seg, punctuation, fillers)]
        if not segs:
            segs = before_texts
        segs = enforce_max_chars(segs, max_chars)
        segs = repair_protected_term_boundaries(segs)
        segs = merge_short_segments(segs, min_chars, max_chars)

        start_ts, _ = _cue_start_end(window_cues[0])
        _, end_ts = _cue_start_end(window_cues[-1])
        timings = interpolate_timestamps(start_ts, end_ts, segs)
        for seg, (seg_start, seg_end) in zip(segs, timings):
            rebuilt.append(Cue(index="", timing=f"{seg_start} --> {seg_end}", lines=[seg]))

        if segs != before_texts:
            short_changes.append({
                "window_start_cue": window_cues[0].index,
                "window_end_cue": window_cues[-1].index,
                "before": before_texts,
                "after": segs,
                "segment_lengths": [visible_len(s) for s in segs],
            })
        cursor = end_idx + 1

    while cursor < len(cues):
        rebuilt.append(cues[cursor])
        cursor += 1
    return rebuilt, short_changes, llm_errors, len(short_indices), len(windows)


def optimize_overlong_cues(
    cues: list[Cue],
    overlong_indices: list[int],
    max_chars: int,
    punctuation: str,
    fillers: tuple[str, ...],
    model: str,
    base_url: str,
    api_key: str | None,
    use_llm: bool,
    allow_neighbor_rewrite: bool,
    llm_timeout: int,
    max_consecutive_llm_failures: int,
    fallback_on_llm_error: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[list[Cue], list[dict[str, Any]], list[dict[str, Any]]]:
    if use_llm and overlong_indices and not api_key:
        raise SubtitleOptimizerError("Overlong subtitle splitting requires an API key.")

    split_results: dict[int, list[str]] = {}
    neighbor_results: dict[int, list[str]] = {}
    overlong_changes: list[dict[str, Any]] = []
    llm_errors: list[dict[str, Any]] = []

    if use_llm:
        consecutive_llm_failures = 0
        llm_disabled = False
        for position, idx in enumerate(overlong_indices, start=1):
            cue = cues[idx]
            before = _cue_text(cue)
            prev_text = _cue_text(cues[idx - 1]) if idx > 0 else ""
            next_text = _cue_text(cues[idx + 1]) if idx + 1 < len(cues) else ""

            if progress_callback:
                progress_callback({
                    "status": "progress",
                    "message": "optimizing overlong subtitle",
                    "current": position,
                    "total": len(overlong_indices),
                    "cue": cue.index,
                    "percent": round(position * 100 / max(1, len(overlong_indices))),
                    "mode": "local_fallback" if llm_disabled else "llm",
                })

            if llm_disabled:
                segs = greedy_split(before, max_chars)
            else:
                try:
                    raw = chat_completion(
                        build_messages(prev_text, before, next_text, max_chars, allow_neighbor_rewrite),
                        model=model,
                        base_url=base_url,
                        api_key=api_key or "",
                        timeout=llm_timeout,
                    )
                    consecutive_llm_failures = 0

                    if allow_neighbor_rewrite:
                        prev_segs, segs, next_segs = normalize_window_payload(raw, prev_text, before, next_text)
                        if not segments_preserve_content(
                            [prev_text, before, next_text],
                            [*prev_segs, *segs, *next_segs],
                            punctuation,
                        ):
                            raise SubtitleOptimizerError("LLM neighbor rewrite output changed, added, or removed subtitle text")
                        if idx > 0 and prev_segs and prev_segs != [prev_text]:
                            neighbor_results[idx - 1] = enforce_max_chars(
                                [clean_subtitle_text(s, punctuation, fillers) for s in prev_segs],
                                max_chars,
                            )
                        if idx + 1 < len(cues) and next_segs and next_segs != [next_text]:
                            neighbor_results[idx + 1] = enforce_max_chars(
                                [clean_subtitle_text(s, punctuation, fillers) for s in next_segs],
                                max_chars,
                            )
                    else:
                        segs = normalize_segments_payload(raw, before)
                        if not segments_preserve_content([before], segs, punctuation):
                            raise SubtitleOptimizerError("LLM split output changed, added, or removed subtitle text")
                except Exception as exc:
                    if not fallback_on_llm_error:
                        raise
                    consecutive_llm_failures += 1
                    segs = greedy_split(before, max_chars)
                    llm_errors.append(
                        {
                            "cue": cue.index,
                            "timing": cue.timing,
                            "error": str(exc),
                            "fallback": "local_split",
                        }
                    )
                    if progress_callback:
                        progress_callback({
                            "status": "fallback",
                            "message": "LLM subtitle split failed; using local fallback",
                            "current": position,
                            "total": len(overlong_indices),
                            "cue": cue.index,
                            "error": str(exc),
                        })
                    if consecutive_llm_failures >= max_consecutive_llm_failures:
                        llm_disabled = True
                        if progress_callback:
                            progress_callback({
                                "status": "fallback",
                                "message": "LLM subtitle split disabled after consecutive failures",
                                "consecutive_failures": consecutive_llm_failures,
                                "remaining": len(overlong_indices) - position,
                            })

            segs = enforce_max_chars([clean_subtitle_text(s, punctuation, fillers) for s in segs], max_chars)
            segs = repair_protected_term_boundaries(segs)
            split_results[idx] = segs
            overlong_changes.append({
                "cue": cue.index,
                "timing": cue.timing,
                "before": before,
                "after": segs,
                "split_count": len(segs),
                "segment_lengths": [visible_len(s) for s in segs],
            })

    expanded: list[Cue] = []
    for idx, cue in enumerate(cues):
        if idx in neighbor_results:
            segs = neighbor_results[idx]
            start, end = _cue_start_end(cue)
            timings = interpolate_timestamps(start, end, segs)
            for s, (s_start, s_end) in zip(segs, timings):
                expanded.append(Cue(index="", timing=f"{s_start} --> {s_end}", lines=[s]))
            continue

        if idx in split_results:
            segs = split_results[idx]
            start, end = _cue_start_end(cue)
            timings = interpolate_timestamps(start, end, segs)
            for s, (s_start, s_end) in zip(segs, timings):
                expanded.append(Cue(index="", timing=f"{s_start} --> {s_end}", lines=[s]))
            continue

        if idx in overlong_indices and not use_llm:
            before = _cue_text(cue)
            segments = repair_protected_term_boundaries(greedy_split(before, max_chars))
            start, end = _cue_start_end(cue)
            timings = interpolate_timestamps(start, end, segments)
            for seg, (seg_start, seg_end) in zip(segments, timings):
                expanded.append(Cue(index="", timing=f"{seg_start} --> {seg_end}", lines=[seg]))
            overlong_changes.append({
                "cue": cue.index,
                "timing": cue.timing,
                "before": before,
                "after": segments,
                "split_count": len(segments),
                "segment_lengths": [visible_len(s) for s in segments],
            })
            continue

        expanded.append(Cue(index="", timing=cue.timing, lines=[_cue_text(cue)]))

    return expanded, overlong_changes, llm_errors


# ── main optimizer ─────────────────────────────────────────────────────────────

def repair_asr_boundaries(
    srt: Path,
    output: Path | None,
    report_path: Path | None,
    boundary_ms: list[int],
    max_chars: int,
    min_chars: int,
    punctuation: str,
    model: str,
    base_url: str,
    api_key: str | None,
    use_llm: bool = True,
    fillers: tuple[str, ...] = DEFAULT_REMOVE_FILLER_WORDS,
    llm_timeout: int = 45,
    fallback_on_llm_error: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cues = parse_srt(srt)
    repaired, changes, llm_errors, window_count = repair_asr_boundary_windows(
        cues,
        boundary_ms=boundary_ms,
        min_chars=min_chars,
        max_chars=max_chars,
        punctuation=punctuation,
        fillers=fillers,
        model=model,
        base_url=base_url,
        api_key=api_key,
        use_llm=use_llm,
        llm_timeout=llm_timeout,
        fallback_on_llm_error=fallback_on_llm_error,
        progress_callback=progress_callback,
    )

    for i, cue in enumerate(repaired, start=1):
        cue.index = str(i)

    if output is None:
        output = srt.with_name(f"{srt.stem}.asr-boundary.srt")
    if report_path is None:
        report_path = DEFAULT_WORK_DIR / f"{srt.stem}.asr-boundary-report.json"

    write_srt(repaired, output)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_srt": str(srt),
        "output_srt": str(output),
        "boundary_ms": boundary_ms,
        "boundary_count": len(boundary_ms),
        "boundary_window_count": window_count,
        "boundary_changed_window_count": len(changes),
        "max_chars": max_chars,
        "min_chars": min_chars,
        "removed_punctuation": punctuation,
        "removed_fillers": list(fillers),
        "use_llm": use_llm,
        "llm_timeout": llm_timeout,
        "llm_fallback_error_count": len(llm_errors),
        "cue_count_before": len(cues),
        "cue_count_after": len(repaired),
        "boundary_changes": changes,
        "llm_errors": llm_errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def optimize_srt(
    srt: Path,
    output: Path | None,
    report_path: Path | None,
    max_chars: int,
    min_chars: int,
    punctuation: str,
    model: str,
    base_url: str,
    api_key: str | None,
    use_llm: bool = True,
    fillers: tuple[str, ...] = DEFAULT_REMOVE_FILLER_WORDS,
    allow_neighbor_rewrite: bool = False,
    optimize_short: bool = True,
    llm_timeout: int = 45,
    max_consecutive_llm_failures: int = 3,
    fallback_on_llm_error: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cues = parse_srt(srt)

    # Step 1: clean punctuation
    cleaned_cues: list[Cue] = []
    punctuation_changes: list[dict[str, Any]] = []
    for cue in cues:
        before = " ".join(cue.lines)
        after = clean_subtitle_text(before, punctuation, fillers)
        cleaned_cues.append(Cue(index=cue.index, timing=cue.timing, lines=after.split("\n") if after else [""]))
        if before != after:
            punctuation_changes.append({"cue": cue.index, "timing": cue.timing, "before": before, "after": after})

    short_detected_count = sum(
        1 for cue in cleaned_cues if 0 < visible_len(_cue_text(cue)) < min_chars
    )
    short_window_count = 0
    short_changes: list[dict[str, Any]] = []
    short_llm_errors: list[dict[str, Any]] = []
    if optimize_short:
        short_cues, short_changes, short_llm_errors, short_detected_count, short_window_count = optimize_short_windows(
            cleaned_cues,
            min_chars=min_chars,
            max_chars=max_chars,
            punctuation=punctuation,
            fillers=fillers,
            model=model,
            base_url=base_url,
            api_key=api_key,
            use_llm=use_llm,
            llm_timeout=llm_timeout,
            max_consecutive_llm_failures=max_consecutive_llm_failures,
            fallback_on_llm_error=fallback_on_llm_error,
            progress_callback=progress_callback,
        )
    else:
        short_cues = cleaned_cues
        if progress_callback:
            progress_callback({
                "status": "skipped",
                "message": "short subtitle windows deferred to full LLM terminology review",
                "short_detected_count": short_detected_count,
                "min_chars": min_chars,
            })

    # Step 2: identify overlong cues after cleanup, before full terminology review.
    overlong_indices = [
        idx for idx, cue in enumerate(short_cues) if visible_len(_cue_text(cue)) > max_chars
    ]
    if progress_callback:
        progress_callback({
            "status": "planned",
            "message": "overlong subtitles identified",
            "cue_count": len(short_cues),
            "punctuation_changed_cue_count": len(punctuation_changes),
            "short_detected_count": short_detected_count,
            "short_window_count": short_window_count,
            "overlong_detected_count": len(overlong_indices),
            "min_chars": min_chars,
            "use_llm": use_llm,
        })

    expanded, overlong_changes, overlong_llm_errors = optimize_overlong_cues(
        short_cues,
        overlong_indices=overlong_indices,
        max_chars=max_chars,
        punctuation=punctuation,
        fillers=fillers,
        model=model,
        base_url=base_url,
        api_key=api_key,
        use_llm=use_llm,
        allow_neighbor_rewrite=allow_neighbor_rewrite,
        llm_timeout=llm_timeout,
        max_consecutive_llm_failures=max_consecutive_llm_failures,
        fallback_on_llm_error=fallback_on_llm_error,
        progress_callback=progress_callback,
    )

    llm_errors: list[dict[str, Any]] = []
    llm_errors.extend(short_llm_errors)
    llm_errors.extend(overlong_llm_errors)

    # Step 4: renumber all cues and write output
    for i, cue in enumerate(expanded, start=1):
        cue.index = str(i)

    if output is None:
        output = srt.with_name(f"{srt.stem}.optimized.srt")
    if report_path is None:
        report_path = DEFAULT_WORK_DIR / f"{srt.stem}.subtitle-optimization-report.json"

    write_srt(expanded, output)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_srt": str(srt),
        "output_srt": str(output),
        "max_chars": max_chars,
        "min_chars": min_chars,
        "removed_punctuation": punctuation,
        "removed_fillers": list(fillers),
        "use_llm": use_llm,
        "llm_timeout": llm_timeout,
        "allow_neighbor_rewrite": allow_neighbor_rewrite,
        "optimize_short": optimize_short,
        "llm_fallback_error_count": len(llm_errors),
        "cue_count_before": len(cues),
        "cue_count_after": len(expanded),
        "punctuation_changed_cue_count": len(punctuation_changes),
        "overlong_detected_count": len(overlong_indices),
        "overlong_changed_cue_count": len(overlong_changes),
        "short_detected_count": short_detected_count,
        "short_window_count": short_window_count,
        "short_changed_window_count": len(short_changes),
        "punctuation_changes": punctuation_changes,
        "overlong_changes": overlong_changes,
        "short_changes": short_changes,
        "llm_errors": llm_errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def greedy_split(text: str, max_chars: int) -> list[str]:
    return split_preserving_english_spaces(text, max_chars)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split overlong SRT cues into independent short cues.")
    parser.add_argument("--srt", required=True)
    parser.add_argument("--output")
    parser.add_argument("--report")
    parser.add_argument("--max-chars", type=int, default=20)
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument("--remove-punctuation", default=DEFAULT_REMOVE_PUNCTUATION)
    parser.add_argument(
        "--remove-fillers",
        default=DEFAULT_REMOVE_FILLERS_TEXT,
        help="Comma/space separated filler words to remove. Empty value disables filler cleanup.",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-timeout", type=int, default=30)
    parser.add_argument(
        "--allow-neighbor-rewrite",
        action="store_true",
        help="Allow LLM to redistribute text across previous/current/next cues.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = optimize_srt(
        srt=Path(args.srt).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve() if args.output else None,
        report_path=Path(args.report).expanduser().resolve() if args.report else None,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        punctuation=args.remove_punctuation,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        use_llm=not args.no_llm,
        fillers=parse_filler_words(args.remove_fillers),
        allow_neighbor_rewrite=args.allow_neighbor_rewrite,
        llm_timeout=args.llm_timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
