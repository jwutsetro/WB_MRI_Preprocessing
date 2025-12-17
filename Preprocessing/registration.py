from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


PARAMETER_DIR = Path(__file__).resolve().parent / "parameter_files"


def register_patient(patient_dir: Path) -> None:
    """Run station-to-station registration using DWI as moving and apply transforms to ADC."""
    dwi_dir = _choose_dwi_dir(patient_dir)
    adc_dir = patient_dir / "ADC"
    if dwi_dir is None or not dwi_dir.exists() or not adc_dir.exists():
        return
    param_dir = patient_dir / "_registration_params"
    param_dir.mkdir(parents=True, exist_ok=True)
    stations = _load_stations(dwi_dir)
    if len(stations) <= 1:
        return
    center_idx = (len(stations) - 1) // 2
    adc_index = _build_station_index(adc_dir) if adc_dir.exists() else {}
    _register_chain(
        stations[center_idx:],
        adc_index=adc_index,
        param_dir=param_dir,
    )
    _register_chain(
        list(reversed(stations[: center_idx + 1])),
        adc_index=adc_index,
        param_dir=param_dir,
    )


def register_wholebody_dwi_to_anatomical(patient_dir: Path) -> None:
    """Rigidly register whole-body DWI/ADC to anatomical (T1) and resample onto the anatomical grid.

    This step is intentionally conservative: for typical datasets where DWI and T1 are already
    close to aligned, it avoids deformable registration and selects the best of (identity,
    initialization, optimized) transforms by gradient-magnitude correlation to reduce the chance
    of making the alignment worse.
    """
    fixed_path = patient_dir / "T1.nii.gz"
    if not fixed_path.exists():
        fixed_path = _choose_anatomical_wb(patient_dir)
        if fixed_path is None:
            return

    dwi_targets = _wb_dwi_targets(patient_dir)
    if not dwi_targets:
        # Fallback for older layouts where only ADC exists.
        adc_only = patient_dir / "ADC.nii.gz"
        if adc_only.exists():
            dwi_targets = [adc_only]
        else:
            return

    moving_reg_path = next((p for p in dwi_targets if _modality_from_path(p).lower() == "adc"), None)
    if moving_reg_path is None:
        moving_reg_path = _choose_dwi_wb(patient_dir) or dwi_targets[0]

    fixed = sitk.ReadImage(str(fixed_path))
    moving_reg = sitk.ReadImage(str(moving_reg_path))
    if fixed.GetDimension() != 3 or moving_reg.GetDimension() != 3:
        return

    transform = _estimate_rigid_transform_fixed_to_moving(fixed=fixed, moving=moving_reg)

    for target_path in dwi_targets:
        if not target_path.exists() or target_path.resolve() == fixed_path.resolve():
            continue
        moving = sitk.ReadImage(str(target_path))
        if moving.GetDimension() != 3:
            continue
        registered = _resample_to_reference(moving=moving, reference=fixed, transform=transform)
        sitk.WriteImage(registered, str(target_path), True)


def _estimate_rigid_transform_fixed_to_moving(
    *,
    fixed: sitk.Image,
    moving: sitk.Image,
    target_spacing_mm: float = 4.0,
    metric_sampling_percentage: float = 0.2,
    max_translation_mm: float = 120.0,
    max_rotation_deg: float = 20.0,
    min_eval_voxels: int = 2000,
) -> sitk.Transform:
    """Estimate a robust rigid transform (fixed->moving) for cross-modality WB ADC/DWI to anatomical.

    Returns a transform suitable for `sitk.Resample(moving, fixed, transform, ...)`.
    """
    identity = sitk.Euler3DTransform()
    fixed_ds = _prepare_for_registration(fixed, target_spacing_mm=target_spacing_mm)
    moving_ds = _prepare_for_registration(moving, target_spacing_mm=target_spacing_mm)
    fixed_mask = _foreground_mask(fixed_ds)

    initial = sitk.CenteredTransformInitializer(
        fixed_ds,
        moving_ds,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # RANDOM is more robust than REGULAR when the foreground mask is sparse/fragmented.
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(metric_sampling_percentage, seed=42)
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)
    try:
        final = registration.Execute(fixed_ds, moving_ds)
    except Exception as exc:
        # Metric sampling can fail when masks/initialization lead to zero valid samples.
        # Retry without masks as a last resort.
        try:
            registration = sitk.ImageRegistrationMethod()
            registration.SetInterpolator(sitk.sitkLinear)
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            registration.SetMetricSamplingPercentage(metric_sampling_percentage, seed=42)
            registration.SetOptimizerAsGradientDescentLineSearch(
                learningRate=1.0,
                numberOfIterations=200,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10,
            )
            registration.SetOptimizerScalesFromPhysicalShift()
            registration.SetShrinkFactorsPerLevel([4, 2, 1])
            registration.SetSmoothingSigmasPerLevel([2, 1, 0])
            registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration.SetInitialTransform(initial, inPlace=False)
            final = registration.Execute(fixed_ds, moving_ds)
        except Exception:
            print(f"[registration] WB rigid ADC→T1 registration failed ({exc}); using identity transform.")
            return identity

    candidates: List[Tuple[str, sitk.Transform]] = [
        ("identity", identity),
        ("initial", initial),
        ("final", final),
    ]
    best_name, best_transform = _select_best_transform_by_gradient_corr(
        fixed=fixed_ds,
        moving=moving_ds,
        fixed_mask=fixed_mask,
        candidates=candidates,
        min_eval_voxels=min_eval_voxels,
    )

    # Enforce a rigid (rotation+translation) transform type for downstream resampling.
    best_transform = _as_euler3d_transform(best_transform)

    if best_name != "identity":
        translation_mm, rotation_deg = _rigid_motion_magnitude(best_transform)
        if translation_mm > max_translation_mm or rotation_deg > max_rotation_deg:
            print(
                "[registration] Rejecting rigid transform due to excessive motion "
                f"(translation={translation_mm:.1f}mm, rotation={rotation_deg:.1f}deg); using identity."
            )
            return identity

    return best_transform


def _prepare_for_registration(image: sitk.Image, *, target_spacing_mm: float) -> sitk.Image:
    img = sitk.Cast(image, sitk.sitkFloat32)
    img = sitk.RescaleIntensity(img, 0.0, 1.0)
    return _downsample_to_spacing(img, target_spacing_mm=target_spacing_mm, interpolator=sitk.sitkLinear)


def _downsample_to_spacing(image: sitk.Image, *, target_spacing_mm: float, interpolator: int) -> sitk.Image:
    spacing = image.GetSpacing()
    new_spacing = tuple(max(float(s), float(target_spacing_mm)) for s in spacing)
    if all(abs(ns - s) < 1e-6 for ns, s in zip(new_spacing, spacing)):
        return image

    size = image.GetSize()
    new_size = [
        max(1, int(math.ceil(size[i] * spacing[i] / new_spacing[i])))
        for i in range(image.GetDimension())
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetDefaultPixelValue(0.0)
    resample.SetTransform(sitk.Transform(image.GetDimension(), sitk.sitkIdentity))
    resample.SetOutputPixelType(sitk.sitkFloat32)
    return resample.Execute(image)


def _foreground_mask(image: sitk.Image) -> sitk.Image:
    """Return a conservative foreground mask for registration sampling."""
    try:
        mask = sitk.OtsuThreshold(image, 0, 1, 128)
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        mask = sitk.BinaryFillhole(mask)
        mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
        if int(sitk.GetArrayViewFromImage(mask).sum()) == 0:
            raise ValueError("empty mask")
        return mask
    except Exception:
        return _ones_mask_like(image)


def _select_best_transform_by_gradient_corr(
    *,
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image,
    candidates: Sequence[Tuple[str, sitk.Transform]],
    min_eval_voxels: int,
) -> Tuple[str, sitk.Transform]:
    fixed_g = sitk.GradientMagnitude(sitk.SmoothingRecursiveGaussian(fixed, 1.0))
    fixed_g = sitk.Cast(fixed_g, sitk.sitkFloat32)

    fixed_arr = sitk.GetArrayViewFromImage(fixed_g)
    mask_arr = sitk.GetArrayViewFromImage(fixed_mask).astype(bool)
    if int(mask_arr.sum()) < min_eval_voxels:
        mask_arr = np.ones_like(mask_arr, dtype=bool)

    best_name = candidates[0][0]
    best_transform = candidates[0][1]
    best_score = float("-inf")
    for name, transform in candidates:
        moved = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        moved_g = sitk.GradientMagnitude(sitk.SmoothingRecursiveGaussian(moved, 1.0))
        moved_arr = sitk.GetArrayViewFromImage(moved_g)
        score = _masked_correlation(fixed_arr, moved_arr, mask_arr)
        if score > best_score:
            best_score = score
            best_name = name
            best_transform = transform
    return best_name, best_transform


def _masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a1 = np.asarray(a)[mask].astype(np.float32)
    b1 = np.asarray(b)[mask].astype(np.float32)
    if a1.size < 10 or b1.size < 10:
        return float("-inf")
    a1 = a1 - float(a1.mean())
    b1 = b1 - float(b1.mean())
    denom = float(np.linalg.norm(a1) * np.linalg.norm(b1))
    if denom <= 0:
        return float("-inf")
    return float(np.dot(a1, b1) / denom)


def _rigid_motion_magnitude(transform: sitk.Transform) -> Tuple[float, float]:
    """Return (translation_mm, rotation_deg) for common rigid transforms."""
    t = transform
    if isinstance(t, sitk.CompositeTransform) and t.GetNumberOfTransforms() > 0:
        t = t.GetBackTransform()
    params = list(t.GetParameters())
    if len(params) == 0:
        return 0.0, 0.0
    if len(params) == 3:
        translation = np.array(params, dtype=float)
        return float(np.linalg.norm(translation)), 0.0
    if len(params) == 6:
        rotation_rad = np.array(params[:3], dtype=float)
        translation = np.array(params[3:], dtype=float)
        rotation_deg = float(np.max(np.abs(rotation_rad)) * 180.0 / math.pi)
        return float(np.linalg.norm(translation)), rotation_deg
    return float("inf"), float("inf")


def _as_euler3d_transform(transform: sitk.Transform) -> sitk.Euler3DTransform:
    """Coerce a transform into a rigid Euler3D (rotation+translation) transform.

    This ensures the ADC→T1 whole-body registration stays strictly rigid (no scale/shear/deform).
    """
    t = transform
    if isinstance(t, sitk.CompositeTransform) and t.GetNumberOfTransforms() > 0:
        t = t.GetBackTransform()
    params = list(t.GetParameters())
    out = sitk.Euler3DTransform()
    if hasattr(t, "GetCenter"):
        try:
            out.SetCenter(t.GetCenter())
        except Exception:
            pass
    if len(params) == 6:
        out.SetParameters(params)
        return out
    if len(params) == 3:
        out.SetParameters([0.0, 0.0, 0.0, float(params[0]), float(params[1]), float(params[2])])
        return out
    # Identity fallback.
    out.SetParameters([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return out


def _resample_to_reference(*, moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    resample.SetTransform(transform)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(0.0)
    resample.SetOutputPixelType(sitk.sitkFloat32)
    return resample.Execute(moving)


def _load_adc_stations(adc_dir: Path) -> List[Dict]:
    """Deprecated wrapper; use _load_stations for arbitrary modalities."""
    return _load_stations(adc_dir)


def _load_stations(modality_dir: Path) -> List[Dict]:
    stations: List[Dict] = []
    for path in sorted(modality_dir.glob("*.nii*")):
        try:
            img = sitk.ReadImage(str(path))
        except Exception:
            continue
        origin_z = img.GetOrigin()[2]
        stations.append({"path": path, "image": img, "origin_z": origin_z, "station": _station_key(path)})
    stations.sort(key=lambda s: (s["origin_z"], s["path"].stem))
    return stations


def _build_station_index(modality_dir: Path) -> Dict[str, Path]:
    return {_station_key(path): path for path in modality_dir.glob("*.nii*")}


def _station_key(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _choose_dwi_dir(patient_dir: Path) -> Optional[Path]:
    for name in ("dwi", "DWI"):
        candidate = patient_dir / name
        if candidate.is_dir():
            return candidate
    return None


def _register_chain(
    stations: List[Dict],
    adc_index: Optional[Dict[str, Path]] = None,
    param_dir: Optional[Path] = None,
) -> None:
    if len(stations) < 2:
        return
    fixed_img = stations[0]["image"]
    cumulative_translation = np.zeros(3, dtype=float)
    previous_param_file: Optional[Path] = None
    for idx, station in enumerate(stations[1:], start=1):
        moving_img = station["image"]
        overlap = _compute_overlap_indices(fixed_img, moving_img)
        if overlap is None:
            fixed_img = moving_img
            cumulative_translation = np.zeros(3, dtype=float)
            previous_param_file = None
            continue
        fixed_roi, moving_roi = _extract_overlap_images(fixed_img, moving_img, overlap)
        mask = _overlap_sampling_mask(fixed_roi, moving_roi)
        param_files = ("Euler_S2S_MI.txt",)
        initial_transform = (
            _build_initial_transform(cumulative_translation, moving_img) if np.any(cumulative_translation) else None
        )
        initial_transform_file = str(previous_param_file) if previous_param_file is not None else None
        param_maps = _run_elastix(
            fixed=fixed_roi,
            moving=moving_roi,
            mask=mask,
            parameter_files=param_files,
            moving_origin=moving_img.GetOrigin(),
            output_reference=moving_img,
            initial_transform=initial_transform,
            initial_transform_file=initial_transform_file,
        )
        result_reg = _apply_transformix(moving_img, param_maps)
        sitk.WriteImage(result_reg, str(station["path"]), True)
        cumulative_translation += _translation_from_param_maps(param_maps)
        if param_dir is not None:
            param_dir.mkdir(parents=True, exist_ok=True)
            param_file = param_dir / f"{station.get('station', station['path'].stem)}_init.txt"
            try:
                sitk.WriteParameterFile(param_maps[0], str(param_file))
                previous_param_file = param_file
            except Exception:
                previous_param_file = None
        _apply_transform_to_targets(
            station_name=station.get("station", station["path"].stem),
            parameter_maps=param_maps,
            adc_index=adc_index,
            skip_paths={station["path"]},
        )
        fixed_img = result_reg


def _run_elastix(
    fixed: sitk.Image,
    moving: sitk.Image,
    mask: Optional[sitk.Image],
    parameter_files: Sequence[str],
    moving_origin: Optional[Tuple[float, float, float]] = None,
    output_reference: Optional[sitk.Image] = None,
    initial_transform: Optional[sitk.VectorOfParameterMap] = None,
    initial_transform_file: Optional[str] = None,
) -> sitk.VectorOfParameterMap:
    elastix = sitk.ElastixImageFilter()
    elastix.LogToConsoleOn()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)
    param_list = [_load_parameter_map(filename) for filename in parameter_files]
    params = sitk.VectorOfParameterMap()
    for param_map in param_list:
        params.append(param_map)
    elastix.SetParameterMap(params)
    if initial_transform is not None and len(initial_transform) > 0:
        if hasattr(elastix, "SetInitialTransformParameterMap"):
            elastix.SetInitialTransformParameterMap(initial_transform)
        else:
            print("[registration] Initial transform not supported by this SimpleITK build; running without initialization.")
    elif initial_transform_file is not None:
        for param_map in params:
            param_map["InitialTransformParametersFileName"] = [initial_transform_file]
    else:
        for param_map in params:
            param_map["InitialTransformParametersFileName"] = ["NoInitialTransform"]
    # Some parameter sets (RandomSparseMask) require a mask; supply a full-ones mask if none provided.
    def _sampler_name(param_map: sitk.ParameterMap) -> str:
        raw = param_map["ImageSampler"][0] if "ImageSampler" in param_map else ""
        return raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)

    needs_mask = any(_sampler_name(p) == "RandomSparseMask" for p in params)
    if mask is None and needs_mask:
        mask = _ones_mask_like(fixed)
    if mask is not None:
        elastix.SetFixedMask(mask)
    elastix.Execute()
    result = elastix.GetTransformParameterMap()
    reference = output_reference if output_reference is not None else moving
    for param_map in result:
        origin = moving_origin if moving_origin is not None else reference.GetOrigin()
        _set_output_geometry(param_map, reference, origin_override=origin)
    return result


def _compute_overlap_indices(
    fixed: sitk.Image, moving: sitk.Image
) -> Optional[
    Tuple[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]
]:
    spacing_f = fixed.GetSpacing()
    spacing_m = moving.GetSpacing()
    origin_f = fixed.GetOrigin()
    origin_m = moving.GetOrigin()
    size_f = fixed.GetSize()
    size_m = moving.GetSize()

    idx_f: list[tuple[int, int]] = []
    idx_m: list[tuple[int, int]] = []
    for axis in range(3):
        start = max(origin_f[axis], origin_m[axis])
        end = min(origin_f[axis] + size_f[axis] * spacing_f[axis], origin_m[axis] + size_m[axis] * spacing_m[axis])
        if end <= start:
            return None
        start_f = int(math.floor((start - origin_f[axis]) / spacing_f[axis]))
        end_f = int(math.ceil((end - origin_f[axis]) / spacing_f[axis]))
        start_m = int(math.floor((start - origin_m[axis]) / spacing_m[axis]))
        end_m = int(math.ceil((end - origin_m[axis]) / spacing_m[axis]))
        end_f = min(end_f, size_f[axis])
        end_m = min(end_m, size_m[axis])
        len_f = max(0, end_f - start_f)
        len_m = max(0, end_m - start_m)
        common = min(len_f, len_m)
        if common <= 0:
            return None
        idx_f.append((start_f, start_f + common))
        idx_m.append((start_m, start_m + common))

    return (tuple(idx_f), tuple(idx_m))  # type: ignore[return-value]


def _extract_overlap_images(
    fixed: sitk.Image,
    moving: sitk.Image,
    indices: Tuple[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
) -> Tuple[sitk.Image, sitk.Image]:
    idx_f, idx_m = indices
    (xf0, xf1), (yf0, yf1), (zf0, zf1) = idx_f
    (xm0, xm1), (ym0, ym1), (zm0, zm1) = idx_m
    size_f = (xf1 - xf0, yf1 - yf0, zf1 - zf0)
    size_m = (xm1 - xm0, ym1 - ym0, zm1 - zm0)

    roi_f = sitk.RegionOfInterest(fixed, size=size_f, index=(xf0, yf0, zf0))
    roi_m = sitk.RegionOfInterest(moving, size=size_m, index=(xm0, ym0, zm0))
    return roi_f, roi_m


def _overlap_sampling_mask(fixed_roi: sitk.Image, moving_roi: sitk.Image, threshold: float = 0.1, min_voxels: int = 500) -> sitk.Image:
    """Mask overlap region to voxels with signal in fixed ROI (e.g., non-head) above threshold and any signal in moving."""
    fixed_arr = sitk.GetArrayFromImage(fixed_roi).astype(np.float32)
    moving_arr = sitk.GetArrayFromImage(moving_roi).astype(np.float32)
    mask_arr = (fixed_arr > threshold) & (moving_arr > 0)
    if mask_arr.sum() < min_voxels:
        mask_arr = np.ones_like(mask_arr, dtype=np.uint8)
    else:
        mask_arr = mask_arr.astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(fixed_roi)
    return mask_img


def _set_output_geometry(
    param_map: sitk.ParameterMap, reference: sitk.Image, origin_override: Optional[Tuple[float, float, float]] = None
) -> None:
    size = reference.GetSize()
    spacing = reference.GetSpacing()
    origin = origin_override if origin_override is not None else reference.GetOrigin()
    direction = reference.GetDirection()
    param_map["Size"] = [str(v) for v in size]
    param_map["Spacing"] = [str(v) for v in spacing]
    param_map["Origin"] = [str(v) for v in origin]
    param_map["Direction"] = [str(v) for v in direction]


def _ones_mask_like(image: sitk.Image) -> sitk.Image:
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask = mask + 1
    mask.CopyInformation(image)
    return mask


def _apply_transformix(image: sitk.Image, parameter_maps: sitk.VectorOfParameterMap) -> sitk.Image:
    transformix = sitk.TransformixImageFilter()
    transformix.LogToConsoleOff()
    transformix.SetTransformParameterMap(parameter_maps)
    transformix.SetMovingImage(image)
    transformix.Execute()
    return transformix.GetResultImage()


def _apply_transform_to_targets(
    station_name: str,
    parameter_maps: sitk.VectorOfParameterMap,
    adc_index: Optional[Dict[str, Path]] = None,
    skip_paths: Optional[set[Path]] = None,
) -> None:
    skip_resolved = {p.resolve() for p in skip_paths} if skip_paths else set()

    def _should_skip(path: Path) -> bool:
        resolved = path.resolve()
        return resolved in skip_resolved

    if adc_index is not None:
        target = adc_index.get(station_name)
        if target is not None and target.exists() and not _should_skip(target):
            moving = sitk.ReadImage(str(target))
            result = _apply_transformix(moving, parameter_maps)
            sitk.WriteImage(result, str(target), True)


def _choose_anatomical_wb(patient_dir: Path) -> Optional[Path]:
    metadata = _load_metadata(patient_dir)
    candidates: List[Path] = []
    if metadata:
        for modality in metadata.get("anatomical_modalities", []):
            for candidate in _candidate_paths(patient_dir, modality):
                if candidate.exists():
                    candidates.append(candidate)
    candidates.sort(key=lambda p: _anatomical_priority(_modality_from_path(p)))
    if candidates:
        return candidates[0]
    fallbacks = [
        p
        for p in patient_dir.glob("*.nii.gz")
        if not _is_dwi_modality(_modality_from_path(p)) and _looks_wholebody_name(p)
    ]
    fallbacks.sort(key=lambda p: _anatomical_priority(_modality_from_path(p)))
    return fallbacks[0] if fallbacks else None


def _choose_dwi_wb(patient_dir: Path) -> Optional[Path]:
    targets = _wb_dwi_targets(patient_dir)
    adc = [t for t in targets if _modality_from_path(t).lower() == "adc"]
    if adc:
        return adc[0]
    numeric: List[tuple[float, Path]] = []
    for candidate in targets:
        modality = _modality_from_path(candidate)
        try:
            numeric.append((float(modality), candidate))
        except ValueError:
            continue
    if numeric:
        numeric.sort(key=lambda t: t[0], reverse=True)
        return numeric[0][1]
    return targets[0] if targets else None


def _wb_dwi_targets(patient_dir: Path) -> List[Path]:
    targets: List[Path] = []
    dwi_plain = patient_dir / "dwi.nii.gz"
    if dwi_plain.exists():
        targets.append(dwi_plain)
    metadata = _load_metadata(patient_dir)
    if metadata:
        modalities = metadata.get("modalities", {})
        for modality, info in modalities.items():
            files = []
            if isinstance(info, dict) and "files" in info:
                files = info["files"]
            elif isinstance(info, list):
                files = info
            for entry in files:
                if isinstance(entry, dict):
                    modality_name = entry.get("canonical_modality") or modality
                    if entry.get("b_value") is not None or _is_bvalue(modality_name):
                        candidate = patient_dir / entry.get("file", "")
                        if candidate.exists():
                            targets.append(candidate)
                elif isinstance(entry, str):
                    candidate = patient_dir / entry
                    if candidate.exists() and (_is_bvalue(modality) or modality.lower() == "dwi"):
                        targets.append(candidate)
    # Always scan the patient root for WB DWI volumes so we include `ADC.nii.gz` and numeric b-values
    # even when a combined `dwi.nii.gz` exists.
    discovered = [
        p for p in patient_dir.glob("*.nii.gz") if _is_dwi_modality(_modality_from_path(p)) and _looks_wholebody_name(p)
    ]
    targets.extend(discovered)
    targets = sorted(set(targets), key=lambda p: _modality_from_path(p))
    return targets


def _load_parameter_map(filename: str) -> sitk.ParameterMap:
    path = PARAMETER_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")
    return sitk.ReadParameterFile(str(path))


def _load_metadata(patient_dir: Path) -> Optional[Dict]:
    meta_path = patient_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _modality_from_path(path: Path) -> str:
    name = path.name
    stem = None
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            break
    if stem is None:
        stem = path.stem
    return stem[:-3] if stem.endswith("_WB") else stem


def _candidate_paths(patient_dir: Path, modality: str) -> List[Path]:
    return [
        patient_dir / f"{modality}_WB.nii.gz",
        patient_dir / f"{modality}.nii.gz",
    ]


def _looks_wholebody_name(path: Path) -> bool:
    stem = _modality_from_path(path)
    return stem.isalpha() or _is_bvalue(stem) or path.name.endswith("_WB.nii.gz")


def _anatomical_priority(modality: str) -> int:
    name = modality.lower()
    if name == "t1":
        return 0
    if "t2" in name:
        return 1
    if "dixon" in name:
        return 2
    return 3


def _is_bvalue(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _is_dwi_modality(name: str) -> bool:
    return name.lower() in {"adc", "dwi"} or _is_bvalue(name)


def _translation_from_param_maps(param_maps: sitk.VectorOfParameterMap) -> np.ndarray:
    for param_map in param_maps:
        if "Transform" not in param_map:
            continue
        transform_name_raw = param_map["Transform"][0]
        transform_name = transform_name_raw.decode() if isinstance(transform_name_raw, (bytes, bytearray)) else str(transform_name_raw)
        if transform_name.lower() != "translationtransform":
            continue
        params_raw = param_map["TransformParameters"] if "TransformParameters" in param_map else []
        return np.array([float(p) for p in params_raw], dtype=float)
    return np.zeros(3, dtype=float)


def _build_initial_transform(translation: np.ndarray, reference: sitk.Image) -> sitk.VectorOfParameterMap:
    param_map = sitk.ParameterMap()
    param_map["Transform"] = ["TranslationTransform"]
    param_map["NumberOfParameters"] = ["3"]
    param_map["TransformParameters"] = [str(v) for v in translation]
    param_map["InitialTransformParametersFileName"] = ["NoInitialTransform"]
    param_map["HowToCombineTransforms"] = ["Compose"]
    param_map["FixedImageDimension"] = ["3"]
    param_map["MovingImageDimension"] = ["3"]
    param_map["FixedInternalImagePixelType"] = ["float"]
    param_map["MovingInternalImagePixelType"] = ["float"]
    param_map["CenterOfRotationPoint"] = ["0", "0", "0"]
    _set_output_geometry(param_map, reference, origin_override=reference.GetOrigin())
    vec = sitk.VectorOfParameterMap()
    vec.append(param_map)
    return vec
