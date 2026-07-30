from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable

from caption_core.domain.srt import Cue, parse_srt_text


class OnlineAsrError(RuntimeError):
    pass


class OnlineAsrTranscriber:
    def __init__(self, engine: str, *, tool_dir: Path, jianying_sign_service_url: str = ""):
        self.engine = engine
        self.name = f"online:{engine}"
        self.tool_dir = tool_dir
        self.jianying_sign_service_url = jianying_sign_service_url

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress: Callable[[int, str], None] | None = None,
    ) -> list[Cue]:
        module = self._load_module()
        if self.engine == "bcut":
            result = module.BcutASR(str(audio_path)).run(callback=progress)
        elif self.engine == "jianying":
            result = module.JianYingASR(str(audio_path)).run(callback=progress)
        else:
            raise OnlineAsrError(f"Unknown online ASR engine: {self.engine}")

        if not result.has_data():
            raise OnlineAsrError("Online ASR returned no subtitle segments")
        return parse_srt_text(result.to_srt())

    def preflight(self) -> None:
        if self.engine != "jianying":
            return
        module = self._load_module()
        try:
            module.JianYingASR(b"")._get_sign("/lv/v1/upload_sign")
        except Exception as exc:
            raise OnlineAsrError(f"Jianying sign service unavailable: {exc}") from exc

    def _load_module(self) -> ModuleType:
        module = load_online_asr_module(self.tool_dir)
        if self.engine == "jianying" and self.jianying_sign_service_url:
            module.JianYingASR.SIGN_SERVICE_URL = self.jianying_sign_service_url
        return module


def load_online_asr_module(tool_dir: Path) -> ModuleType:
    module_path = tool_dir / "online_asr.py"
    if not module_path.exists():
        raise OnlineAsrError(f"online_asr.py not found in {tool_dir}")

    spec = importlib.util.spec_from_file_location("caption_core_external_online_asr", module_path)
    if spec is None or spec.loader is None:
        raise OnlineAsrError(f"Cannot load online_asr.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
