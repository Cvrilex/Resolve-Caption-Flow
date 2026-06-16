# MVP To Product Migration Plan

## Why Keep `mvp_pipeline/`

`mvp_pipeline/` is the proven closed-loop prototype. It should remain runnable
until the product package reaches feature parity. Moving or renaming it now would
risk breaking a working workflow.

## Target Structure

```text
src/drautocut/
  domain/         pure models and subtitle/term transformations
  integrations/   ASR, LLM, PDF, Resolve adapters
  pipeline/       job orchestration and resumability
  storage/        artifact paths, logs, job history
  web/            FastAPI panel
```

## Migration Order

1. Move pure subtitle and terminology logic first.
   - Source: `mvp_pipeline/term_corrector.py`
   - Source: `mvp_pipeline/subtitle_optimizer.py`
   - Target: `drautocut.domain`

2. Introduce job state and artifact storage.
   - Target: `drautocut.pipeline`
   - Target: `drautocut.storage`
   - Goal: make every step restartable without rerunning ASR.

3. Wrap external services behind adapters.
   - ASR adapter for `tool/online_asr.py`
   - LLM adapter for OpenAI-compatible APIs and LM Studio
   - Resolve adapter for template import and render preset handling

4. Rebuild the web panel on the product package.
   - Keep the current MVP panel as a reference.
   - Add job history, step retry, and terminology impact preview.

5. Retire `mvp_pipeline/` only after the product path runs the same end-to-end
   workflow successfully.

## Next Feature

Build terminology impact preview:

- for every candidate replacement, list affected cues
- show original and preview text
- allow per-term approval before correction
- support continuing without terminology correction

