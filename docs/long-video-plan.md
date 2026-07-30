# Long Video Processing Plan

## Current Production Assumption

- Keep final Resolve import/render as one full video plus one merged SRT.
- Split audio/ASR work into segments so failures and online quota pressure stay local.
- Split LLM correction by subtitle cues, not by raw video duration.

## ASR Segmentation

- Default target segment: 10 minutes.
- Default hard limit: 12 minutes.
- Prefer silence cuts within 45 seconds of the target cut point.
- Store every segment with `start_ms`, `end_ms`, `offset_ms`, and duration.
- Each segment can be sent to online ASR or local ASR independently.
- After ASR, segment-level SRT cues are offset back to full-video time and merged.
- `caption_core.integrations.ffmpeg` provides duration probing, silence detection, and audio segment extraction.
- `caption_core.pipeline.asr` emits segment-level progress events for the Web panel: `running`, `segment_running`, `segment_progress`, `segment_done`, `segment_failed`, `failed`, and `done`.

## Concurrency Policy

- Default `max_workers` should stay at `1` for online ASR until provider behavior is known.
- Allow a user-configurable value such as `2` for online ASR experiments, with clear failure reporting per segment.
- Local ASR can use higher concurrency only after checking CPU/GPU memory pressure.
- Failed segments should be retryable independently; a full-video rerun should not be required.

## LLM Correction

- Process SRT in cue batches rather than sending the whole course at once.
- Use neighboring cues as read-only context.
- Require patch-style output: cue id, original text, corrected text, reason.
- Validate patches before applying: no timing changes, no unknown cue ids, no empty text.
- Run term impact preview before approval.

## Local ASR Track

- Keep online ASR adapters as the stable baseline.
- Add a local adapter behind the same interface.
- First local candidates:
  - FunASR / SenseVoice for Chinese professional videos, including medical courses as one example.
  - faster-whisper as a mature fallback with strong timestamp behavior.
- Compare on 5-10 minute samples before switching production defaults.
