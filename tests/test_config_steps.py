from __future__ import annotations

from Preprocessing.config import StepConfig


def test_steps_string_means_only_those_steps() -> None:
    cfg = StepConfig.from_dict("dicom_sort,adc")
    assert cfg.dicom_sort is True
    assert cfg.adc is True
    assert cfg.registration is False
    assert cfg.reconstruct is False


def test_steps_list_supports_legacy_aliases() -> None:
    cfg = StepConfig.from_dict(["dicom_reconstruction", "adc_creation", "noise_bias_removal", "isis"])
    assert cfg.dicom_sort is True
    assert cfg.adc is True
    assert cfg.noise_bias is True
    assert cfg.isis is True
    assert cfg.registration is False
    assert cfg.reconstruct is False

