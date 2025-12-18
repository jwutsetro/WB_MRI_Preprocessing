from __future__ import annotations

import math
from pathlib import Path

import SimpleITK as sitk

from Preprocessing.noise_bias import _write_image_atomic, apply_log_bias_field


def test_apply_log_bias_field_clamps_extreme_negative_field() -> None:
    image = sitk.Image([8, 8, 8], sitk.sitkFloat32) + 1.0
    extreme_log_field = sitk.Image([8, 8, 8], sitk.sitkFloat32) - 100.0
    extreme_log_field.CopyInformation(image)

    corrected = apply_log_bias_field(image, extreme_log_field)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(corrected)
    maximum = float(stats.GetMaximum())
    assert math.isfinite(maximum)
    assert 1000.0 < maximum < 10000.0


def test_write_image_atomic_preserves_nifti_extension(tmp_path: Path) -> None:
    image = sitk.Image([4, 4, 4], sitk.sitkFloat32) + 1.0
    out = tmp_path / "1.nii.gz"
    _write_image_atomic(image, out)
    assert out.exists()
    assert not (tmp_path / "1.tmp.nii.gz").exists()


def test_write_image_atomic_preserves_plain_nifti_extension(tmp_path: Path) -> None:
    image = sitk.Image([4, 4, 4], sitk.sitkFloat32) + 1.0
    out = tmp_path / "1.nii"
    _write_image_atomic(image, out)
    assert out.exists()
    assert not (tmp_path / "1.tmp.nii").exists()
