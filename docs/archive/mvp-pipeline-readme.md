# MVP Pipeline Workspace

Goal: validate the closed loop:

```text
input video -> online ASR -> SRT -> DaVinci Resolve/MCP import -> render output
```

Put files here:

- `input/` - test videos
- `tool/` - ASR interface scripts or wrappers
- `work/` - generated intermediate files, including ASR SRT
- `output/` - rendered videos
- `logs/` - pipeline logs

Existing ASR script found at:

```text
davinci-resolve-mcp/tool/online_asr.py
```

Run the MVP:

```bash
python3 mvp_pipeline/mvp_pipeline.py --video mvp_pipeline/input/3min.mp4 --engine bcut
```

Current defaults:

- Subtitle preset: `sub01`
- Render preset: `ffpg-fast-23`
- Render format/codec without a preset: MP4/H.264
- Render type request: `x264 8-bit 4:2:0(FFmpeg)`
- Other render settings are left at Resolve defaults except output path/name,
  full-timeline export, video/audio/subtitle export, and source resolution/fps.

Run only the Resolve/render half with an existing SRT:

```bash
python3 mvp_pipeline/mvp_pipeline.py \
  --video mvp_pipeline/input/3min.mp4 \
  --srt mvp_pipeline/work/3min-20260615-224954.bcut.srt
```

Run with terminology correction before Resolve import:

```bash
python3 mvp_pipeline/mvp_pipeline.py \
  --video mvp_pipeline/input/3min.mp4 \
  --srt mvp_pipeline/work/3min-20260615-224954.bcut.srt \
  --terms mvp_pipeline/terms.sample.json \
  --allow-subtitle-preset-fallback \
  --prepare-only
```

The corrected SRT is written to `work/`, and a cue-by-cue replacement report is
written to `logs/`.

Generate a terminology map from course context, then correct SRT before Resolve
import:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="deepseek-v4-pro"
# Optional for OpenAI-compatible services:
# export OPENAI_BASE_URL="https://api.deepseek.com"

python3 mvp_pipeline/mvp_pipeline.py \
  --video mvp_pipeline/input/3min.mp4 \
  --srt mvp_pipeline/work/3min-20260615-224954.bcut.srt \
  --context mvp_pipeline/input/朱晨芳-2026甲状腺相关眼病学习班幻灯朱晨芳.pdf \
  --llm-model deepseek-v4-pro \
  --llm-base-url https://api.deepseek.com \
  --allow-subtitle-preset-fallback \
  --prepare-only
```

For this MVP, `--context` accepts `.txt`, `.md`, and text-extractable `.pdf`.
Image-only PDFs still need OCR/MinerU in a later step.

Split overlong subtitles before Resolve import:

```bash
python3 mvp_pipeline/mvp_pipeline.py \
  --video mvp_pipeline/input/3min.mp4 \
  --srt mvp_pipeline/work/3min-20260616-001248.corrected.srt \
  --optimize-subtitles \
  --subtitle-max-chars 20 \
  --llm-model deepseek-v4-pro \
  --llm-base-url https://api.deepseek.com \
  --template-project mvp_pipeline/sub.drp \
  --use-template-timeline \
  --prepare-only
```

This step removes configured punctuation first, then finds cues over the length
threshold and splits each overlong cue into multiple independent SRT cues with
new time ranges inside the original cue duration. It does not just insert line
breaks inside one subtitle item. `--no-subtitle-llm` is available as a fallback,
but it is mechanical; use the LLM path for semantic splitting.

Preview the LLM prompt without making an API request:

```bash
python3 mvp_pipeline/term_mapper.py \
  --context mvp_pipeline/input/朱晨芳-2026甲状腺相关眼病学习班幻灯朱晨芳.pdf \
  --srt mvp_pipeline/work/3min-20260615-224954.bcut.srt \
  --print-prompt
```

Run while allowing the current scripting gaps to fall back:

```bash
python3 mvp_pipeline/mvp_pipeline.py \
  --video mvp_pipeline/input/3min.mp4 \
  --srt mvp_pipeline/work/3min-20260615-224954.bcut.srt \
  --allow-subtitle-preset-fallback \
  --allow-render-type-fallback
```

List render presets visible to Resolve scripting:

```bash
python3 mvp_pipeline/mvp_pipeline.py --list-resolve-presets
```

Run with a Resolve render preset:

```bash
python3 mvp_pipeline/mvp_pipeline.py \
  --video mvp_pipeline/input/3min.mp4 \
  --srt mvp_pipeline/work/3min-20260615-224954.bcut.srt \
  --allow-subtitle-preset-fallback
```

For the requested render type `x264 8-bit 4:2:0(FFmpeg)`, create a Resolve
render preset manually after selecting that Type in the Deliver page, then pass
that exact preset name with `--render-preset`. When a render preset is supplied,
the script preserves the preset's format/codec/quality settings and only
overrides output directory, output name, full-timeline range, and subtitle burn-in.
Use `--no-render-preset` to deliberately skip the default preset and let the
script choose MP4/H.264 directly.

Prepare the Resolve timeline, apply subtitle Track styling in the Inspector,
then render the current timeline:

```bash
python3 mvp_pipeline/mvp_pipeline.py \
  --video mvp_pipeline/input/3min.mp4 \
  --srt mvp_pipeline/work/3min-20260615-224954.bcut.srt \
  --allow-subtitle-preset-fallback \
  --prepare-only

# After the subtitle Track style is applied in Resolve:
python3 mvp_pipeline/mvp_pipeline.py --render-current
```

This is the current route for the `sub01` subtitle style, because Resolve
scripting does not expose the subtitle Inspector style fields. The styling is
applied on the subtitle `Track` tab, so it affects the whole subtitle track
rather than one subtitle item.
The style values captured from the Resolve Inspector are stored in
`subtitle_style_sub01.json`.

Preferred route for `sub01`: import a styled Resolve template project and reuse
its existing timeline, so the saved subtitle track style is preserved:

```bash
python3 mvp_pipeline/mvp_pipeline.py \
  --video mvp_pipeline/input/3min.mp4 \
  --srt mvp_pipeline/work/3min-20260616-021436.optimized.srt \
  --template-project mvp_pipeline/sub.drp \
  --use-template-timeline \
  --prepare-only
```

This path deletes placeholder items from the imported template timeline, keeps
the template tracks, appends the video at frame 0, then imports the SRT onto the
styled subtitle track. It does not need `--allow-subtitle-preset-fallback`,
because the style comes from the template timeline instead of Resolve's preset
API.

Verified on `3min.mp4`:

- ASR: `bcut` generated a 56-cue SRT.
- Terminology correction: `terms.sample.json` produced a corrected SRT and
  auditable replacement report before Resolve import.
- LLM terminology map: `term_mapper.py` generated a replacement JSON from the
  course PDF plus ASR SRT using DeepSeek's OpenAI-compatible chat API.
- Resolve: connected to DaVinci Resolve Studio 20.3.0.10.
- SRT import: `MediaPool.ImportMedia(.srt)` + `AppendToTimeline` worked.
- Render: MP4/H.264 output with burned-in subtitles completed.
- Render preset path: loading `ffpg-fast-23` with `--render-preset` completed
  and produced a valid 14,929,483-byte MP4/H.264/AAC output matching the manual
  preset render size.
- Subtitle style route: applying the Track Inspector style manually/through UI
  automation, then running `--render-current`, produced a Resolve-rendered output
  with red MiSans Semibold subtitles, white stroke, drop shadow, and the requested
  position.
- Subtitle template route: importing `sub.drp` with `--use-template-timeline`
  preserved the template subtitle track style and placed all 56 SRT cues onto
  that single styled subtitle track without rendering.

Current known issue:

- `sub01` is not exposed through the tested Resolve scripting paths:
  `Project.LoadBurnInPreset`, subtitle item `LoadBurnInPreset`, and subtitle item
  `SetProperty(...)` all returned false. Use the `sub.drp` template route with
  `--use-template-timeline` to preserve the styled subtitle track automatically.
- Reusing the same Resolve project can leave timeline/render state behind. The
  script now creates a unique `DRautocut_MVP_<run_id>` project by default and
  validates timeline duration before rendering.
- Resolve's public scripting docs do not list a stable render setting key named
  `Type`. Resolve rejected the tested keys for
  `x264 8-bit 4:2:0(FFmpeg)`. The script now supports loading a Resolve render
  preset with `--render-preset`, which is the preferred path for this setting.
