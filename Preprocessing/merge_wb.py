from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk


def merge_patient(patient_dir: Path) -> None:
    """Merge all station images for each modality into whole-body volumes with feathered overlaps."""
    modality_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
    for modality_dir in modality_dirs:
        station_files = sorted(modality_dir.glob("*.nii*"))
        if not station_files:
            continue
        if len(station_files) == 1:
            _write_single_station(patient_dir, modality_dir.name, station_files[0])
            continue
        stations = _load_stations(station_files)
        if not stations:
            continue
        reference = _reference_image(stations)
        merged = _feather_merge(stations, reference)
        _write_outputs(patient_dir, modality_dir.name, merged)


def _write_single_station(patient_dir: Path, modality: str, file: Path) -> None:
    target_wb = patient_dir / f"{modality}_WB.nii.gz"
    target_plain = patient_dir / f"{modality}.nii.gz"
    img = sitk.ReadImage(str(file))
    sitk.WriteImage(img, str(target_wb), True)
    sitk.WriteImage(img, str(target_plain), True)


def _load_stations(files: List[Path]) -> List[Dict]:
    stations: List[Dict] = []
    for path in files:
        try:
            img = sitk.ReadImage(str(path))
        except Exception:
            continue
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        size = img.GetSize()
        extent = (
            origin[0] + size[0] * spacing[0],
            origin[1] + size[1] * spacing[1],
            origin[2] + size[2] * spacing[2],
        )
        stations.append(
            {
                "path": path,
                "image": img,
                "origin": origin,
                "spacing": spacing,
                "size": size,
                "extent": extent,
            }
        )
    stations.sort(key=lambda s: (s["origin"][2], s["path"].stem))
    return stations


def _reference_image(stations: List[Dict]) -> sitk.Image:
    spacing = stations[0]["spacing"]
    direction = stations[0]["image"].GetDirection()
    min_origin = [
        min(s["origin"][i] for s in stations) for i in range(3)
    ]
    max_extent = [
        max(s["extent"][i] for s in stations) for i in range(3)
    ]
    size = [
        int(math.ceil((max_extent[i] - min_origin[i]) / spacing[i]))
        for i in range(3)
    ]
    reference = sitk.Image(size[0], size[1], size[2], sitk.sitkFloat32)
    reference.SetSpacing(spacing)
    reference.SetOrigin(tuple(min_origin))
    reference.SetDirection(direction)
    return reference


def _feather_merge(stations: List[Dict], reference: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)

    accum = sitk.Image(reference.GetSize(), sitk.sitkFloat32)
    accum.CopyInformation(reference)
    weight_sum = sitk.Image(reference.GetSize(), sitk.sitkFloat32)
    weight_sum.CopyInformation(reference)

    z_extents = [(s["origin"][2], s["extent"][2]) for s in stations]

    for idx, station in enumerate(stations):
        image = sitk.Cast(station["image"], sitk.sitkFloat32)
        resampled_image = resampler.Execute(image)

        prev_extent = z_extents[idx - 1] if idx > 0 else None
        next_extent = z_extents[idx + 1] if idx + 1 < len(stations) else None
        weight_img = _station_weight_image(station, prev_extent, next_extent)
        weight_resampled = resampler.Execute(weight_img)

        contrib = sitk.Multiply(resampled_image, weight_resampled)
        accum = sitk.Add(accum, contrib)
        weight_sum = sitk.Add(weight_sum, weight_resampled)

    weight_safe = sitk.Add(weight_sum, 1e-6)
    merged = sitk.Divide(accum, weight_safe)
    merged.CopyInformation(reference)
    return merged


def _station_weight_image(
    station: Dict,
    prev_extent: Optional[Tuple[float, float]],
    next_extent: Optional[Tuple[float, float]],
) -> sitk.Image:
    spacing = station["spacing"]
    size = station["size"]
    origin = station["origin"]

    z_coords = origin[2] + spacing[2] * np.arange(size[2], dtype=np.float32)
    weights = np.ones_like(z_coords, dtype=np.float32)

    if prev_extent is not None:
        overlap_start = max(origin[2], prev_extent[0])
        overlap_end = min(origin[2] + size[2] * spacing[2], prev_extent[1])
        if overlap_end > overlap_start:
            length = overlap_end - overlap_start
            ramp = (z_coords - overlap_start) / length
            ramp = np.clip(ramp, 0.0, 1.0)
            weights = np.where((z_coords >= overlap_start) & (z_coords <= overlap_end), ramp, weights)

    if next_extent is not None:
        overlap_start = max(origin[2], next_extent[0])
        overlap_end = min(origin[2] + size[2] * spacing[2], next_extent[1])
        if overlap_end > overlap_start:
            length = overlap_end - overlap_start
            ramp = (overlap_end - z_coords) / length
            ramp = np.clip(ramp, 0.0, 1.0)
            weights = np.where((z_coords >= overlap_start) & (z_coords <= overlap_end), np.minimum(weights, ramp), weights)

    arr = np.ones((size[2], size[1], size[0]), dtype=np.float32)
    arr *= weights[:, None, None]
    weight_img = sitk.GetImageFromArray(arr)
    weight_img.SetSpacing(spacing)
    weight_img.SetOrigin(origin)
    weight_img.SetDirection(station["image"].GetDirection())
    return weight_img


def _write_outputs(patient_dir: Path, modality: str, image: sitk.Image) -> None:
    target_wb = patient_dir / f"{modality}_WB.nii.gz"
    target_plain = patient_dir / f"{modality}.nii.gz"
    sitk.WriteImage(image, str(target_wb), True)
    sitk.WriteImage(image, str(target_plain), True)
