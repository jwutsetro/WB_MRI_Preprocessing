from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import SimpleITK as sitk


PARAMETER_DIR = Path(__file__).resolve().parent / "parameter_files"


def register_patient(patient_dir: Path) -> None:
    """Run station-to-station registration (ADC-driven) and apply transforms to all DWI stations."""
    adc_dir = patient_dir / "ADC"
    if not adc_dir.exists():
        return
    stations = _load_adc_stations(adc_dir)
    if len(stations) <= 1:
        return
    center_idx = (len(stations) - 1) // 2
    _register_chain(stations[center_idx:], patient_dir)
    _register_chain(list(reversed(stations[: center_idx + 1])), patient_dir)


def register_wholebody_dwi_to_anatomical(patient_dir: Path) -> None:
    """Register whole-body DWI (ADC or highest b-value) to anatomical WB and apply the transform to all DWI WB volumes."""
    fixed_path = _choose_anatomical_wb(patient_dir)
    moving_path = _choose_dwi_wb(patient_dir)
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
    for target in _wb_dwi_targets(patient_dir):
        img = sitk.ReadImage(str(target))
        result = _apply_transformix(img, param_maps)
        sitk.WriteImage(result, str(target), True)


def _load_adc_stations(adc_dir: Path) -> List[Dict]:
    stations: List[Dict] = []
    for path in sorted(adc_dir.glob("*.nii*")):
        try:
            img = sitk.ReadImage(str(path))
        except Exception:
            continue
        origin_z = img.GetOrigin()[2]
        stations.append({"path": path, "image": img, "origin_z": origin_z})
    stations.sort(key=lambda s: (s["origin_z"], s["path"].stem))
    return stations


def _register_chain(stations: List[Dict], patient_dir: Path) -> None:
    if len(stations) < 2:
        return
    fixed_img = stations[0]["image"]
    for station in stations[1:]:
        moving_img = station["image"]
        mask = _overlap_mask(fixed_img, moving_img)
        if mask is None:
            fixed_img = moving_img
            continue
        param_maps = _run_elastix(
            fixed=fixed_img,
            moving=moving_img,
            mask=mask,
            parameter_files=("Euler_S2S_MSD.txt",),
        )
        result_adc = _apply_transformix(moving_img, param_maps)
        sitk.WriteImage(result_adc, str(station["path"]), True)
        _apply_transform_to_dwi(patient_dir, station["path"].stem, param_maps)
        fixed_img = result_adc


def _run_elastix(
    fixed: sitk.Image,
    moving: sitk.Image,
    mask: Optional[sitk.Image],
    parameter_files: Sequence[str],
) -> sitk.VectorOfParameterMap:
    elastix = sitk.ElastixImageFilter()
    elastix.LogToConsoleOff()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)
    if mask is not None:
        elastix.SetFixedMask(mask)
    params = sitk.VectorOfParameterMap()
    for filename in parameter_files:
        params.append(_load_parameter_map(filename))
    elastix.SetParameterMap(params)
    elastix.Execute()
    result = elastix.GetTransformParameterMap()
    for param_map in result:
        param_map["Origin"] = [str(val) for val in moving.GetOrigin()]
    return result


def _overlap_mask(fixed: sitk.Image, moving: sitk.Image) -> Optional[sitk.Image]:
    spacing_f = fixed.GetSpacing()
    spacing_m = moving.GetSpacing()
    origin_f = fixed.GetOrigin()
    origin_m = moving.GetOrigin()
    size_f = fixed.GetSize()
    size_m = moving.GetSize()

    start_f = origin_f[2]
    end_f = origin_f[2] + size_f[2] * spacing_f[2]
    start_m = origin_m[2]
    end_m = origin_m[2] + size_m[2] * spacing_m[2]
    overlap_start = max(start_f, start_m)
    overlap_end = min(end_f, end_m)
    if overlap_end <= overlap_start:
        return None
    start_idx = max(0, int(math.floor((overlap_start - start_f) / spacing_f[2])))
    end_idx = min(size_f[2], int(math.ceil((overlap_end - start_f) / spacing_f[2])))
    if end_idx <= start_idx:
        end_idx = min(size_f[2], start_idx + 1)
    if end_idx <= start_idx:
        return None
    mask = sitk.Image(size_f, sitk.sitkUInt8)
    mask.CopyInformation(fixed)
    depth = max(1, end_idx - start_idx)
    ones = sitk.Image([size_f[0], size_f[1], depth], sitk.sitkUInt8)
    ones = ones + 1
    return sitk.Paste(mask, ones, ones.GetSize(), destinationIndex=[0, 0, start_idx])


def _apply_transformix(image: sitk.Image, parameter_maps: sitk.VectorOfParameterMap) -> sitk.Image:
    transformix = sitk.TransformixImageFilter()
    transformix.LogToConsoleOff()
    transformix.SetTransformParameterMap(parameter_maps)
    transformix.SetMovingImage(image)
    transformix.Execute()
    return transformix.GetResultImage()


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
            candidate = patient_dir / f"{modality}_WB.nii.gz"
            if candidate.exists():
                candidates.append(candidate)
    candidates.sort(key=lambda p: _anatomical_priority(_modality_from_wb(p)))
    if candidates:
        return candidates[0]
    fallbacks = [p for p in patient_dir.glob("*_WB.nii.gz") if not _is_dwi_modality(_modality_from_wb(p))]
    fallbacks.sort(key=lambda p: _anatomical_priority(_modality_from_wb(p)))
    return fallbacks[0] if fallbacks else None


def _choose_dwi_wb(patient_dir: Path) -> Optional[Path]:
    targets = _wb_dwi_targets(patient_dir)
    adc = [t for t in targets if _modality_from_wb(t).lower() == "adc"]
    if adc:
        return adc[0]
    numeric: List[tuple[float, Path]] = []
    for candidate in targets:
        modality = _modality_from_wb(candidate)
        try:
            numeric.append((float(modality), candidate))
        except ValueError:
            continue
    if numeric:
        numeric.sort(key=lambda t: t[0], reverse=True)
        return numeric[0][1]
    return targets[0] if targets else None


def _wb_dwi_targets(patient_dir: Path) -> List[Path]:
    targets = [
        path
        for path in patient_dir.glob("*_WB.nii.gz")
        if _is_dwi_modality(_modality_from_wb(path))
    ]
    targets.sort(key=lambda p: _modality_from_wb(p))
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


def _modality_from_wb(path: Path) -> str:
    name = path.stem
    return name[:-3] if name.endswith("_WB") else name


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
    return name.lower() == "adc" or _is_bvalue(name)
