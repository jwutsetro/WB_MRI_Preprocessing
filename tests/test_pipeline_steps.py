from __future__ import annotations

from pathlib import Path

import pytest

from Preprocessing.config import NyulConfig, PipelineConfig, StepConfig
from Preprocessing.alignment import pipeline_apply_step_selection, pipeline_select_steps


def _cfg(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        sequence_rules=[],
        steps=StepConfig(),
        nyul=NyulConfig(enable=False),
    )


def test_pipeline_select_steps_only() -> None:
    selected = pipeline_select_steps(only=["adc", "reconstruct"])
    assert selected == {"adc", "reconstruct"}


def test_pipeline_select_steps_range() -> None:
    selected = pipeline_select_steps(from_step="adc", to_step="reconstruct")
    assert "adc" in selected
    assert "reconstruct" in selected
    assert "dicom_sort" not in selected
    assert "nyul" not in selected


def test_pipeline_apply_step_selection(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    selected = {"adc", "reconstruct"}
    pipeline_apply_step_selection(cfg, selected)
    assert cfg.steps.adc is True
    assert cfg.steps.reconstruct is True
    assert cfg.steps.dicom_sort is False


def test_pipeline_select_steps_invalid_range() -> None:
    with pytest.raises(ValueError):
        pipeline_select_steps(from_step="reconstruct", to_step="adc")
