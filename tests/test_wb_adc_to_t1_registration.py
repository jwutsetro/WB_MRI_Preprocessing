from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from Preprocessing.registration import register_wholebody_dwi_to_anatomical


def _gradient_corr(fixed: sitk.Image, moving: sitk.Image) -> float:
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(fixed, 0, 1, 128)

    fixed_g = sitk.GradientMagnitude(sitk.SmoothingRecursiveGaussian(fixed, 1.0))
    moving_g = sitk.GradientMagnitude(sitk.SmoothingRecursiveGaussian(moving, 1.0))

    a = sitk.GetArrayFromImage(fixed_g)
    b = sitk.GetArrayFromImage(moving_g)
    m = sitk.GetArrayFromImage(mask).astype(bool)
    if int(m.sum()) < 10:
        return float("-inf")

    a1 = a[m].astype(np.float32)
    b1 = b[m].astype(np.float32)
    a1 -= float(a1.mean())
    b1 -= float(b1.mean())
    denom = float(np.linalg.norm(a1) * np.linalg.norm(b1))
    if denom <= 0:
        return float("-inf")
    return float(np.dot(a1, b1) / denom)


def test_wb_adc_to_t1_registration_improves_alignment_and_resamples_all_dwi(tmp_path) -> None:
    fixed = sitk.GaussianSource(
        sitk.sitkFloat32,
        size=[96, 96, 64],
        mean=[48, 50, 28],
        sigma=[10, 12, 8],
        spacing=[1.2, 1.2, 2.0],
    )
    fixed = sitk.RescaleIntensity(fixed, 0.0, 1.0)
    fixed.SetOrigin((0.0, 0.0, 0.0))

    adc_arr = sitk.GetArrayFromImage(fixed)
    adc_arr = 1.0 - adc_arr
    adc = sitk.GetImageFromArray(adc_arr.astype(np.float32))
    adc.CopyInformation(fixed)
    adc.SetOrigin((8.0, 0.0, 0.0))  # introduce physical misalignment (8mm)
    adc = sitk.AdditiveGaussianNoise(adc, mean=0.0, standardDeviation=0.01)

    b1000 = sitk.Multiply(adc, 2.0)

    sitk.WriteImage(fixed, str(tmp_path / "T1.nii.gz"))
    sitk.WriteImage(adc, str(tmp_path / "ADC.nii.gz"))
    sitk.WriteImage(b1000, str(tmp_path / "1000.nii.gz"))
    sitk.WriteImage(b1000, str(tmp_path / "dwi.nii.gz"))

    adc_in_fixed_before = sitk.Resample(adc, fixed, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    corr_before = _gradient_corr(fixed, adc_in_fixed_before)

    register_wholebody_dwi_to_anatomical(tmp_path)

    adc_after = sitk.ReadImage(str(tmp_path / "ADC.nii.gz"))
    b1000_after = sitk.ReadImage(str(tmp_path / "1000.nii.gz"))
    dwi_after = sitk.ReadImage(str(tmp_path / "dwi.nii.gz"))

    assert adc_after.GetSize() == fixed.GetSize()
    assert adc_after.GetSpacing() == pytest.approx(fixed.GetSpacing(), abs=1e-6)
    assert adc_after.GetOrigin() == pytest.approx(fixed.GetOrigin(), abs=1e-6)
    assert adc_after.GetDirection() == pytest.approx(fixed.GetDirection(), abs=1e-6)

    assert b1000_after.GetSize() == fixed.GetSize()
    assert dwi_after.GetSize() == fixed.GetSize()

    corr_after = _gradient_corr(fixed, adc_after)
    assert corr_after > corr_before + 0.05
