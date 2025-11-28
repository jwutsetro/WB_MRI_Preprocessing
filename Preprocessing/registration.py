from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk


PARAM_FILE = Path(__file__).parent / "param_files" / "S2S.txt"


def _overlap_indices(fixed: sitk.Image, moving: sitk.Image) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
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

    idx_start_f = int(round((overlap_start - start_f) / spacing_f[2]))
    idx_end_f = idx_start_f + int(round((overlap_end - overlap_start) / spacing_f[2]))
    idx_start_m = int(round((overlap_start - start_m) / spacing_m[2]))
    idx_end_m = idx_start_m + int(round((overlap_end - overlap_start) / spacing_m[2]))

    idx_end_f = min(idx_end_f, size_f[2])
    idx_end_m = min(idx_end_m, size_m[2])
    return (idx_start_f, idx_end_f), (idx_start_m, idx_end_m)


def _mask_from_indices(image: sitk.Image, idx: Tuple[int, int]) -> sitk.Image:
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    start, end = idx
    region = [0, 0, start]
    size = [image.GetSize()[0], image.GetSize()[1], max(1, end - start)]
    ones = sitk.Image(size, sitk.sitkUInt8)
    ones = ones + 1
    mask = sitk.Paste(mask, ones, ones.GetSize(), destinationIndex=region)
    return mask


def _register_pair(fixed: sitk.Image, moving: sitk.Image, mask: Optional[sitk.Image]) -> sitk.ParameterMap:
    elastix = sitk.ElastixImageFilter()
    elastix.LogToConsoleOff()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)
    if mask is not None:
        elastix.SetFixedMask(mask)
    elastix.SetParameterMap(sitk.ReadParameterFile(str(PARAM_FILE)))
    elastix.Execute()
    return elastix.GetTransformParameterMap()[0]


def _apply_transform(image: sitk.Image, parammap: sitk.ParameterMap) -> sitk.Image:
    transformix = sitk.TransformixImageFilter()
    transformix.LogToConsoleOff()
    transformix.SetTransformParameterMap(parammap)
    transformix.SetMovingImage(image)
    transformix.Execute()
    return transformix.GetResultImage()


def _sorted_station_files(modality_dir: Path) -> List[Path]:
    files = [p for p in modality_dir.glob("*.nii*") if p.is_file()]
    files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    return files


def register_patient(patient_dir: Path) -> None:
    adc_dir = patient_dir / "ADC"
    if not adc_dir.exists():
        return
    adc_files = _sorted_station_files(adc_dir)
    if len(adc_files) <= 1:
        return
    adc_images: Dict[str, sitk.Image] = {f.stem: sitk.ReadImage(str(f)) for f in adc_files}
    stations = list(adc_files)
    center_idx = len(stations) // 2
    reference_station = stations[center_idx].stem

    # propagate upward
    ref_image = adc_images[reference_station]
    for f in stations[center_idx + 1 :]:
        moving_image = adc_images[f.stem]
        overlap = _overlap_indices(ref_image, moving_image)
        mask = _mask_from_indices(ref_image, overlap[0]) if overlap else None
        parammap = _register_pair(ref_image, moving_image, mask)
        # apply to ADC
        moved_adc = _apply_transform(moving_image, parammap)
        sitk.WriteImage(moved_adc, str(f), True)
        # apply to all b-value modalities
        _apply_transform_to_bvalues(patient_dir, f.stem, parammap)
        # update reference
        ref_image = moved_adc

    # propagate downward
    ref_image = adc_images[reference_station]
    for f in reversed(stations[:center_idx]):
        moving_image = adc_images[f.stem]
        overlap = _overlap_indices(ref_image, moving_image)
        mask = _mask_from_indices(ref_image, overlap[0]) if overlap else None
        parammap = _register_pair(ref_image, moving_image, mask)
        moved_adc = _apply_transform(moving_image, parammap)
        sitk.WriteImage(moved_adc, str(f), True)
        _apply_transform_to_bvalues(patient_dir, f.stem, parammap)
        ref_image = moved_adc


def _apply_transform_to_bvalues(patient_dir: Path, station: str, parammap: sitk.ParameterMap) -> None:
    for modality_dir in patient_dir.iterdir():
        if not modality_dir.is_dir():
            continue
        if not modality_dir.name.isdigit():
            continue
        target = modality_dir / f"{station}.nii.gz"
        if not target.exists():
            continue
        img = sitk.ReadImage(str(target))
        moved = _apply_transform(img, parammap)
        sitk.WriteImage(moved, str(target), True)
