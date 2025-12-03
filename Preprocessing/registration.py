from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


PARAMETER_DIR = Path(__file__).resolve().parent / "parameter_files"


def register_patient(patient_dir: Path) -> None:
    """Run station-to-station registration driven by DWI and apply transforms to ADC and all DWI stations."""
    registration_dir = _choose_registration_dwi_dir(patient_dir)
    adc_dir = patient_dir / "ADC"
    if registration_dir is None and not adc_dir.exists():
        return
    if registration_dir is None:
        registration_dir = adc_dir
    stations = _load_stations(registration_dir)
    if len(stations) <= 1:
        return
    center_idx = (len(stations) - 1) // 2
    adc_index = _build_station_index(adc_dir) if adc_dir.exists() else {}
    dwi_dirs = _bvalue_dirs(patient_dir)
    dwi_indices = {d: _build_station_index(d) for d in dwi_dirs}
    _register_chain(
        stations[center_idx:],
        adc_index=adc_index,
        dwi_indices=dwi_indices,
        registration_dir=registration_dir,
    )
    _register_chain(
        list(reversed(stations[: center_idx + 1])),
        adc_index=adc_index,
        dwi_indices=dwi_indices,
        registration_dir=registration_dir,
    )


def register_wholebody_dwi_to_anatomical(patient_dir: Path) -> None:
    """Register whole-body ADC to anatomical (T1) and apply the transform to DWI."""
    fixed_path = patient_dir / "T1.nii.gz"
    moving_path = patient_dir / "ADC.nii.gz"
    dwi_path = patient_dir / "dwi.nii.gz"
    if not fixed_path.exists() or not moving_path.exists():
        fixed_path = _choose_anatomical_wb(patient_dir)
        moving_path = _choose_dwi_wb(patient_dir)
        dwi_path = moving_path if moving_path else dwi_path
        if fixed_path is None or moving_path is None:
            return
    fixed = sitk.ReadImage(str(fixed_path))
    moving = sitk.ReadImage(str(moving_path))
    param_maps = _run_elastix(
        fixed=fixed,
        moving=moving,
        mask=None,
        parameter_files=("S2A_Pair_Euler_WB.txt", "S2A_Pair_BSpline_WB.txt"),
    )
    adc_reg = _apply_transformix(moving, param_maps)
    sitk.WriteImage(adc_reg, str(moving_path), True)
    if dwi_path.exists():
        dwi_img = sitk.ReadImage(str(dwi_path))
        dwi_reg = _apply_transformix(dwi_img, param_maps)
        sitk.WriteImage(dwi_reg, str(dwi_path), True)


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


def _choose_registration_dwi_dir(patient_dir: Path) -> Optional[Path]:
    dwi_dirs = _bvalue_dirs(patient_dir)
    if dwi_dirs:
        return dwi_dirs[0]
    candidate = patient_dir / "dwi"
    if candidate.is_dir():
        return candidate
    candidate_upper = patient_dir / "DWI"
    if candidate_upper.is_dir():
        return candidate_upper
    return None


def _register_chain(
    stations: List[Dict],
    adc_index: Optional[Dict[str, Path]] = None,
    dwi_indices: Optional[Dict[Path, Dict[str, Path]]] = None,
    registration_dir: Optional[Path] = None,
) -> None:
    if len(stations) < 2:
        return
    fixed_img = stations[0]["image"]
    for idx, station in enumerate(stations[1:], start=1):
        moving_img = station["image"]
        overlap = _compute_overlap_indices(fixed_img, moving_img)
        if overlap is None:
            fixed_img = moving_img
            continue
        fixed_roi, moving_roi = _extract_overlap_images(fixed_img, moving_img, overlap)
        mask = _overlap_sampling_mask(fixed_roi, moving_roi)
        param_files = ("Euler_S2S_MI.txt",)
        param_maps = _run_elastix(
            fixed=fixed_roi,
            moving=moving_roi,
            mask=mask,
            parameter_files=param_files,
            moving_origin=moving_img.GetOrigin(),
            output_reference=moving_img,
        )
        result_reg = _apply_transformix(moving_img, param_maps)
        sitk.WriteImage(result_reg, str(station["path"]), True)
        _apply_transform_to_targets(
            station_name=station.get("station", station["path"].stem),
            parameter_maps=param_maps,
            adc_index=adc_index,
            dwi_indices=dwi_indices,
            skip_paths={station["path"]},
            registration_dir=registration_dir,
        )
        fixed_img = result_reg


def _run_elastix(
    fixed: sitk.Image,
    moving: sitk.Image,
    mask: Optional[sitk.Image],
    parameter_files: Sequence[str],
    moving_origin: Optional[Tuple[float, float, float]] = None,
    output_reference: Optional[sitk.Image] = None,
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
    dwi_indices: Optional[Dict[Path, Dict[str, Path]]] = None,
    skip_paths: Optional[set[Path]] = None,
    registration_dir: Optional[Path] = None,
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

    if dwi_indices is not None:
        for modality_dir, index in dwi_indices.items():
            target = index.get(station_name)
            if target is None or not target.exists():
                continue
            if registration_dir is not None and modality_dir == registration_dir and _should_skip(target):
                continue
            if _should_skip(target):
                continue
            moving = sitk.ReadImage(str(target))
            result = _apply_transformix(moving, parameter_maps)
            sitk.WriteImage(result, str(target), True)


def _apply_transform_to_dwi(patient_dir: Path, station_name: str, parameter_maps: sitk.VectorOfParameterMap) -> None:
    for modality_dir in _bvalue_dirs(patient_dir):
        target = modality_dir / f"{station_name}.nii.gz"
        if not target.exists():
            continue
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
    if not targets:
        targets = [
            p for p in patient_dir.glob("*.nii.gz") if _is_dwi_modality(_modality_from_path(p)) and _looks_wholebody_name(p)
        ]
    targets = sorted(set(targets), key=lambda p: _modality_from_path(p))
    return targets


def _bvalue_dirs(patient_dir: Path) -> List[Path]:
    dirs = [p for p in patient_dir.iterdir() if p.is_dir() and _is_bvalue(p.name)]
    dirs.sort(key=lambda p: float(p.name))
    return dirs


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
    name = path.stem
    return name[:-3] if name.endswith("_WB") else name


def _candidate_paths(patient_dir: Path, modality: str) -> List[Path]:
    return [
        patient_dir / f"{modality}_WB.nii.gz",
        patient_dir / f"{modality}.nii.gz",
    ]


def _looks_wholebody_name(path: Path) -> bool:
    stem = path.stem
    return stem.endswith("_WB") or stem.isalpha() or _is_bvalue(stem)


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
