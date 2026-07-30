# Directory Migration Plan

## Current Decision

The old `mvp_pipeline/` prototype directory has been retired as the main
development entry. Proven code now lives in shallow, responsibility-oriented
top-level folders.

## Target Structure

```text
app/            WebUI and local FastAPI service
pipeline/       caption production, terminology, cleanup, and LLM review
integrations/   online ASR and external service adapters
caption_core/       reusable domain models and package-style modules
resources/      Resolve template, subtitle styles, sample resources
data/           inputs, work artifacts, outputs, logs, knowledge base
docs/           business workflow and technical records
tests/          automated checks
scripts/        helper scripts
vendor/         third-party reference projects
```

## Migration Status

1. Web server moved to `app/web_server.py`.
2. Static WebUI moved to `app/web/`.
3. Core pipeline moved to `pipeline/caption_pipeline.py`.
4. Terminology and subtitle tools moved to `pipeline/`.
5. Online ASR adapters moved to `integrations/online_asr.py`.
6. Resolve template and style resources moved to `resources/`.
7. Runtime input, work, output, and log folders moved to `data/`.
8. Third-party reference projects moved to `vendor/`.
9. Reusable package modules moved to root-level `caption_core/`.

Third-party folders under `vendor/` keep their original internal structure.

## Next Feature

Continue product work on terminology impact preview:

- for every candidate replacement, list affected cues
- show original and preview text
- allow per-term approval before correction
- support continuing without terminology correction
