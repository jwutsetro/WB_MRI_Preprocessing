from __future__ import annotations

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
    """Register whole-body ADC to anatomical (T1) and apply the transform to DWI."""
    fixed_path = patient_dir / "T1.nii.gz"
    moving_path = patient_dir / "ADC.nii.gz"
    dwi_path = patient_dir / "dwi.nii.gz"
    if not fixed_path.exists() or not moving_path.exists():
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
    for idx, station in enumerate(stations[1:], start=1):
        moving_img = station["image"]
        mask = _overlap_mask(fixed_img, moving_img)
        if mask is None:
            fixed_img = moving_img
            continue
        is_top_pair = idx == len(stations) - 1
        param_files = (
            ("Euler_S2S_MSD_head.txt", "Euler_S2S_MI_head.txt")
            if is_top_pair
            else ("Euler_S2S_MSD.txt", "Euler_S2S_MI.txt")
        )
        param_maps = _run_elastix(
            fixed=fixed_img,
            moving=moving_img,
            mask=mask,
            parameter_files=param_files,
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

    def extent(origin: Tuple[float, float, float], spacing: Tuple[float, float, float], size: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return (
            origin[0] + size[0] * spacing[0],
            origin[1] + size[1] * spacing[1],
            origin[2] + size[2] * spacing[2],
        )

    ext_f = extent(origin_f, spacing_f, size_f)
    ext_m = extent(origin_m, spacing_m, size_m)

    overlap_min = (
        max(origin_f[0], origin_m[0]),
        max(origin_f[1], origin_m[1]),
        max(origin_f[2], origin_m[2]),
    )
    overlap_max = (
        min(ext_f[0], ext_m[0]),
        min(ext_f[1], ext_m[1]),
        min(ext_f[2], ext_m[2]),
    )
    if overlap_max[0] <= overlap_min[0] or overlap_max[1] <= overlap_min[1] or overlap_max[2] <= overlap_min[2]:
        return None

    start_idx = [
        max(0, int(math.floor((overlap_min[i] - origin_f[i]) / spacing_f[i])))
        for i in range(3)
    ]
    end_idx = [
        min(size_f[i], int(math.ceil((overlap_max[i] - origin_f[i]) / spacing_f[i])))
        for i in range(3)
    ]
    size_idx = [max(1, end_idx[i] - start_idx[i]) for i in range(3)]

    mask = sitk.Image(size_f, sitk.sitkUInt8)
    mask.CopyInformation(fixed)
    ones = sitk.Image(size_idx, sitk.sitkUInt8)
    ones = ones + 1
    return sitk.Paste(mask, ones, ones.GetSize(), destinationIndex=start_idx)


def _apply_transformix(image: sitk.Image, parameter_maps: sitk.VectorOfParameterMap) -> sitk.Image:
    transformix = sitk.TransformixImageFilter()
    transformix.LogToConsoleOff()
    transformix.SetTransformParameterMap(parameter_maps)
    transformix.SetMovingImage(image)
    transformix.Execute()
    return transformix.GetResultImage()


def _apply_transform_to_dwi(patient_dir: Path, station_name: str, parameter_maps: sitk.VectorOfParameterMap) -> None:
    for modality_dir in _dwi_dirs(patient_dir):
        target = modality_dir / f"{station_name}.nii.gz"
        if not target.exists():
            continue
        moving = sitk.ReadImage(str(target))
        result = _apply_transformix(moving, parameter_maps)
        sitk.WriteImage(result, str(target), True)


def _dwi_dirs(patient_dir: Path) -> List[Path]:
    dirs = []
    dwi_dir = patient_dir / "dwi"
    if dwi_dir.is_dir():
        dirs.append(dwi_dir)
    dirs.extend([p for p in patient_dir.iterdir() if p.is_dir() and _is_bvalue(p.name)])
    return dirs


def _load_parameter_map(filename: str) -> sitk.ParameterMap:
    path = PARAMETER_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")
    return sitk.ReadParameterFile(str(path))


def _is_bvalue(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False
