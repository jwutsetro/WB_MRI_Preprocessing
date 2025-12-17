from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from Preprocessing.elastix_resample import resample_moving_to_fixed_crop, transform_from_elastix_parameter_map


def test_resample_to_fixed_crop_changes_size_when_spacing_changes() -> None:
    fixed = sitk.Image(100, 100, 100, sitk.sitkFloat32)
    fixed.SetSpacing((1.0, 1.0, 1.0))
    fixed.SetOrigin((0.0, 0.0, 0.0))

    moving = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    moving.SetSpacing((2.0, 2.0, 2.0))
    moving.SetOrigin((0.0, 0.0, 0.0))

    pm = sitk.ParameterMap()
    pm["Transform"] = ["TranslationTransform"]
    pm["TransformParameters"] = ["0", "0", "0"]

    transform = transform_from_elastix_parameter_map(pm)
    out = resample_moving_to_fixed_crop(moving, fixed, transform)

    assert out.GetSpacing() == fixed.GetSpacing()
    assert out.GetSize() == (19, 19, 19)


def test_euler_transform_parses_and_resamples() -> None:
    fixed = sitk.Image(100, 100, 100, sitk.sitkFloat32)
    fixed.SetSpacing((1.0, 1.0, 1.0))
    fixed.SetOrigin((0.0, 0.0, 0.0))

    moving = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    moving.SetSpacing((2.0, 2.0, 2.0))
    moving.SetOrigin((0.0, 0.0, 0.0))

    pm = sitk.ParameterMap()
    pm["Transform"] = ["EulerTransform"]
    pm["TransformParameters"] = ["0", "0", "0", "0", "0", "0"]
    pm["CenterOfRotationPoint"] = ["0", "0", "0"]

    transform = transform_from_elastix_parameter_map(pm)
    out = resample_moving_to_fixed_crop(moving, fixed, transform)

    assert out.GetSpacing() == fixed.GetSpacing()
    assert out.GetSize() == (19, 19, 19)

