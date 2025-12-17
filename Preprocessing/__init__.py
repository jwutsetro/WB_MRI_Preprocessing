"""WB MRI preprocessing package.

Keep imports lightweight so submodules can be used without eagerly importing optional
runtime dependencies (e.g., YAML) at package import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Preprocessing.config import PipelineConfig as PipelineConfig
    from Preprocessing.pipeline import PipelineRunner as PipelineRunner

__all__ = ["PipelineConfig", "PipelineRunner"]


def __getattr__(name: str) -> Any:
    if name == "PipelineConfig":
        from Preprocessing.config import PipelineConfig as _PipelineConfig

        return _PipelineConfig
    if name == "PipelineRunner":
        from Preprocessing.pipeline import PipelineRunner as _PipelineRunner

        return _PipelineRunner
    raise AttributeError(name)
