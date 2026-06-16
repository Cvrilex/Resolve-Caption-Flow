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
from typing import Any
from urllib.parse import urlparse

from term_corrector import Cue, parse_srt, write_srt


ROOT = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = ROOT / "work"
DEFAULT_REMOVE_PUNCTUATION = "，,"


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


# ── LLM helpers ────────────────────────────────────────────────────────────────

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
        raise SubtitleOptimizerError(f"LLM request failed: {exc.code} {detail[:1000]}") from exc
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise SubtitleOptimizerError(f"Unexpected LLM response shape: {data}") from exc


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

    segments: list[str] = []
    current = ""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+./-]*|[\u4e00-\u9fff]|[^\s]", text)
    for token in tokens:
        joiner = " " if current and re.match(r"^[A-Za-z0-9]", token) and re.search(r"[A-Za-z0-9+./-]$", current) else ""
        candidate = f"{current}{joiner}{token}" if current else token
        if current and visible_len(candidate) > max_chars:
            segments.append(current)
            current = token
        else:
            current = candidate
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


# ── main optimizer ─────────────────────────────────────────────────────────────

def optimize_srt(
    srt: Path,
    output: Path | None,
    report_path: Path | None,
    max_chars: int,
    punctuation: str,
    model: str,
    base_url: str,
    api_key: str | None,
    use_llm: bool = True,
    allow_neighbor_rewrite: bool = False,
) -> dict[str, Any]:
    cues = parse_srt(srt)

    # Step 1: clean punctuation
    cleaned_cues: list[Cue] = []
    punctuation_changes: list[dict[str, Any]] = []
    for cue in cues:
        before = " ".join(cue.lines)
        after = clean_punctuation(before, punctuation)
        cleaned_cues.append(Cue(index=cue.index, timing=cue.timing, lines=after.split("\n") if after else [""]))
        if before != after:
            punctuation_changes.append({"cue": cue.index, "timing": cue.timing, "before": before, "after": after})

    # Step 2: identify overlong cues
    overlong_indices = [
        idx for idx, cue in enumerate(cleaned_cues) if visible_len("".join(cue.lines)) > max_chars
    ]

    if use_llm and overlong_indices and not api_key:
        raise SubtitleOptimizerError("Overlong subtitle splitting requires an API key.")

    # Step 3: split overlong cues — LLM once per cue, store result, then rebuild
    split_results: dict[int, list[str]] = {}  # cue_idx -> segments
    neighbor_results: dict[int, list[str]] = {}  # cue_idx -> new segments (for neighbor rewrites)
    overlong_changes: list[dict[str, Any]] = []

    if use_llm:
        final_cues = list(cleaned_cues)
        for idx in overlong_indices:
            cue = final_cues[idx]
            before = " ".join(cue.lines)
            prev_text = " ".join(final_cues[idx - 1].lines) if idx > 0 else ""
            next_text = " ".join(final_cues[idx + 1].lines) if idx + 1 < len(final_cues) else ""

            raw = chat_completion(
                build_messages(prev_text, before, next_text, max_chars, allow_neighbor_rewrite),
                model=model, base_url=base_url, api_key=api_key or "",
            )

            if allow_neighbor_rewrite:
                prev_segs, segs, next_segs = normalize_window_payload(raw, prev_text, before, next_text)
                if idx > 0 and prev_segs and prev_segs != [prev_text]:
                    neighbor_results[idx - 1] = enforce_max_chars(
                        [clean_punctuation(s, punctuation) for s in prev_segs],
                        max_chars,
                    )
                if idx + 1 < len(final_cues) and next_segs and next_segs != [next_text]:
                    neighbor_results[idx + 1] = enforce_max_chars(
                        [clean_punctuation(s, punctuation) for s in next_segs],
                        max_chars,
                    )
            else:
                segs = normalize_segments_payload(raw, before)

            segs = enforce_max_chars([clean_punctuation(s, punctuation) for s in segs], max_chars)
            split_results[idx] = segs

            overlong_changes.append({
                "cue": cue.index, "timing": cue.timing,
                "before": before,
                "after": segs,
                "split_count": len(segs),
                "segment_lengths": [visible_len(s) for s in segs],
            })

    # Step 4: rebuild cue list
    expanded: list[Cue] = []
    for idx, cue in enumerate(cleaned_cues):
        # Check if this cue was rewritten by a neighbor split
        if idx in neighbor_results:
            segs = neighbor_results[idx]
            start, end = cue.timing.split(" --> ")
            timings = interpolate_timestamps(start, end, segs)
            for s, (s_start, s_end) in zip(segs, timings):
                expanded.append(Cue(index="", timing=f"{s_start} --> {s_end}", lines=[s]))
            continue

        if idx in split_results:
            segs = split_results[idx]
            start, end = cue.timing.split(" --> ")
            timings = interpolate_timestamps(start, end, segs)
            for s, (s_start, s_end) in zip(segs, timings):
                expanded.append(Cue(index="", timing=f"{s_start} --> {s_end}", lines=[s]))
            continue

        if idx in overlong_indices and not use_llm:
            full_text = "".join(cue.lines)
            segments = greedy_split(full_text, max_chars)
            start, end = cue.timing.split(" --> ")
            timings = interpolate_timestamps(start, end, segments)
            before = " ".join(cue.lines)
            for seg, (seg_start, seg_end) in zip(segments, timings):
                expanded.append(Cue(index="", timing=f"{seg_start} --> {seg_end}", lines=[seg]))
            overlong_changes.append({
                "cue": cue.index, "timing": cue.timing,
                "before": before,
                "after": segments,
                "split_count": len(segments),
                "segment_lengths": [visible_len(s) for s in segments],
            })
            continue

        # Normal cue, keep as-is (single line)
        expanded.append(Cue(index="", timing=cue.timing, lines=[" ".join(cue.lines)]))

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
        "removed_punctuation": punctuation,
        "allow_neighbor_rewrite": allow_neighbor_rewrite,
        "cue_count_before": len(cues),
        "cue_count_after": len(expanded),
        "punctuation_changed_cue_count": len(punctuation_changes),
        "overlong_detected_count": len(overlong_indices),
        "overlong_changed_cue_count": len(overlong_changes),
        "punctuation_changes": punctuation_changes,
        "overlong_changes": overlong_changes,
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
    parser.add_argument("--remove-punctuation", default=DEFAULT_REMOVE_PUNCTUATION)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--no-llm", action="store_true")
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
        punctuation=args.remove_punctuation,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        use_llm=not args.no_llm,
        allow_neighbor_rewrite=args.allow_neighbor_rewrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
