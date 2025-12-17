from __future__ import annotations

import math
from typing import Sequence, Tuple

import SimpleITK as sitk


def transform_from_elastix_parameter_map(param_map: sitk.ParameterMap) -> sitk.Transform:
    """Convert a SimpleElastix parameter map to a SimpleITK transform.

    Notes:
    - The returned transform follows ITK/SimpleITK resampling convention: it maps
      points from the output/reference space to the input/moving space.
    - Only the transforms used by this repo's debug registration pipeline are
      supported (EulerTransform and TranslationTransform).
    """

    raw_name = param_map["Transform"][0] if "Transform" in param_map else ""
    name = _decode(raw_name).lower()
    params = [_to_float(v) for v in (param_map["TransformParameters"] if "TransformParameters" in param_map else [])]

    if name == "eulertransform":
        if len(params) != 6:
            raise ValueError(f"Expected 6 Euler parameters, got {len(params)}")
        if "CenterOfRotationPoint" not in param_map:
            raise ValueError("Missing CenterOfRotationPoint for EulerTransform.")
        center = [_to_float(v) for v in param_map["CenterOfRotationPoint"]]
        if len(center) != 3:
            raise ValueError(f"Expected 3 CenterOfRotationPoint values, got {len(center)}")
        t = sitk.Euler3DTransform()
        t.SetCenter(tuple(center))
        t.SetRotation(params[0], params[1], params[2])
        t.SetTranslation(tuple(params[3:6]))
        return t

    if name == "translationtransform":
        if len(params) != 3:
            raise ValueError(f"Expected 3 translation parameters, got {len(params)}")
        t = sitk.TranslationTransform(3)
        t.SetOffset(tuple(params))
        return t

    raise NotImplementedError(f"Unsupported elastix transform: {name!r}")


def build_cropped_reference_in_fixed_space(
    fixed_space: sitk.Image,
    moving_image: sitk.Image,
    transform_output_to_input: sitk.Transform,
    *,
    pad_voxels: int = 0,
) -> sitk.Image:
    """Build an output reference image aligned to `fixed_space`, cropped to the moved bounds of `moving_image`.

    The output image:
    - Uses `fixed_space` spacing + direction.
    - Is axis-aligned to `fixed_space` index grid.
    - Covers the axis-aligned bounding box of the moving image after applying the
      inverse transform (moving->fixed), optionally padded by `pad_voxels`.
    """
    inverse = transform_output_to_input.GetInverse()
    corner_indices = _corner_indices(moving_image.GetSize())
    fixed_continuous_indices = []
    for idx in corner_indices:
        point_m = moving_image.TransformIndexToPhysicalPoint(idx)
        point_f = inverse.TransformPoint(point_m)
        fixed_continuous_indices.append(fixed_space.TransformPhysicalPointToContinuousIndex(point_f))

    mins = [min(c[i] for c in fixed_continuous_indices) for i in range(3)]
    maxs = [max(c[i] for c in fixed_continuous_indices) for i in range(3)]
    start = [int(math.floor(v)) - pad_voxels for v in mins]
    end = [int(math.ceil(v)) + pad_voxels for v in maxs]
    size = [max(1, end[i] - start[i] + 1) for i in range(3)]

    origin = fixed_space.TransformIndexToPhysicalPoint(tuple(start))
    ref = sitk.Image(size[0], size[1], size[2], sitk.sitkFloat32)
    ref.SetSpacing(fixed_space.GetSpacing())
    ref.SetDirection(fixed_space.GetDirection())
    ref.SetOrigin(origin)
    return ref


def resample_moving_to_fixed_crop(
    moving_image: sitk.Image,
    fixed_space: sitk.Image,
    transform_output_to_input: sitk.Transform,
    *,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
    pad_voxels: int = 0,
) -> sitk.Image:
    """Resample `moving_image` into `fixed_space` (spacing/direction) using a cropped reference grid."""
    ref = build_cropped_reference_in_fixed_space(
        fixed_space=fixed_space,
        moving_image=moving_image,
        transform_output_to_input=transform_output_to_input,
        pad_voxels=pad_voxels,
    )
    return sitk.Resample(moving_image, ref, transform_output_to_input, interpolator, default_value, moving_image.GetPixelID())


def _corner_indices(size: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    sx, sy, sz = (int(size[0]), int(size[1]), int(size[2]))
    return (
        (0, 0, 0),
        (max(0, sx - 1), 0, 0),
        (0, max(0, sy - 1), 0),
        (0, 0, max(0, sz - 1)),
        (max(0, sx - 1), max(0, sy - 1), 0),
        (max(0, sx - 1), 0, max(0, sz - 1)),
        (0, max(0, sy - 1), max(0, sz - 1)),
        (max(0, sx - 1), max(0, sy - 1), max(0, sz - 1)),
    )


def _decode(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return str(value)


def _to_float(value: object) -> float:
    return float(_decode(value))
