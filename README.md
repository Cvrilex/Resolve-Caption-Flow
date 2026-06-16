# DRautocut

Local automation for medical course captioning:

```text
video -> online ASR -> SRT -> PDF/LLM terminology review -> subtitle cleanup -> DaVinci Resolve render
```

## Current State

The working MVP lives in `mvp_pipeline/`. It has already validated the full loop:

- upload video and optional PDF through a local web panel
- run online ASR
- generate terminology candidates from course context
- review and approve terminology replacements
- split overlong subtitles and remove configured punctuation
- import video/SRT into DaVinci Resolve through the styled `sub.drp` template
- render with the `ffpg-fast-23` Resolve preset

## Development Direction

Formal product development starts from the repository root under `src/drautocut/`.
The MVP is kept intact as a reference implementation while reusable code is
migrated into package modules.

Planned module boundaries:

- `drautocut.domain`: shared data models and pure subtitle/term logic
- `drautocut.pipeline`: orchestration and resumable job state
- `drautocut.integrations`: ASR, LLM, PDF, and DaVinci Resolve adapters
- `drautocut.web`: the local FastAPI panel and static frontend
- `drautocut.storage`: job folders, logs, artifacts, and history

## Run The MVP

```bash
python3 mvp_pipeline/web_server.py --host 127.0.0.1 --port 8742
```

Then open:

```text
http://127.0.0.1:8742/
```

Runtime inputs, outputs, logs, SRT files, and rendered videos are intentionally
ignored by git.

