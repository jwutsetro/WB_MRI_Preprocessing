from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk


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


def _translation_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    mask: Optional[sitk.Image] = None,
    scales: Tuple[float, float, float] = (1.0, 1.0, 1000.0),
    shrink_factors: Tuple[int, int, int] = (4, 2, 1),
    smoothing_sigmas: Tuple[int, int, int] = (2, 1, 0),
) -> sitk.Transform:
    tx = sitk.TranslationTransform(3)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2)
    if mask is not None:
        R.SetMetricFixedMask(mask)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        relaxationFactor=0.5,
    )
    R.SetOptimizerScales(scales)
    R.SetInitialTransform(tx, inPlace=False)
    R.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    R.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R.Execute(fixed, moving)


def _apply_transform(image: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    return sitk.Resample(
        image,
        reference,
        transform,
        sitk.sitkLinear,
        0.0,
        image.GetPixelID(),
    )


def _sorted_station_files(modality_dir: Path) -> List[Path]:
    files = [p for p in modality_dir.glob("*.nii*") if p.is_file()]
    files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    return files


def register_patient(patient_dir: Path) -> None:
    """Inter-station registration using ADC overlaps; transforms applied to all b-values."""
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
        transform = _translation_registration(ref_image, moving_image, mask)
        moved_adc = _apply_transform(moving_image, ref_image, transform)
        sitk.WriteImage(moved_adc, str(f), True)
        _apply_transform_to_bvalues(patient_dir, f.stem, ref_image, transform)
        ref_image = moved_adc

    # propagate downward
    ref_image = adc_images[reference_station]
    for f in reversed(stations[:center_idx]):
        moving_image = adc_images[f.stem]
        overlap = _overlap_indices(ref_image, moving_image)
        mask = _mask_from_indices(ref_image, overlap[0]) if overlap else None
        transform = _translation_registration(ref_image, moving_image, mask)
        moved_adc = _apply_transform(moving_image, ref_image, transform)
        sitk.WriteImage(moved_adc, str(f), True)
        _apply_transform_to_bvalues(patient_dir, f.stem, ref_image, transform)
        ref_image = moved_adc


def _apply_transform_to_bvalues(patient_dir: Path, station: str, reference: sitk.Image, transform: sitk.Transform) -> None:
    for modality_dir in patient_dir.iterdir():
        if not modality_dir.is_dir():
            continue
        if not modality_dir.name.isdigit():
            continue
        target = modality_dir / f"{station}.nii.gz"
        if not target.exists():
            continue
        img = sitk.ReadImage(str(target))
        moved = _apply_transform(img, reference, transform)
        sitk.WriteImage(moved, str(target), True)


def register_wholebody_adc_to_t1(patient_dir: Path) -> None:
    t1_wb = patient_dir / "T1_WB.nii.gz"
    adc_wb = patient_dir / "ADC_WB.nii.gz"
    if not t1_wb.exists() or not adc_wb.exists():
        return
    fixed = sitk.ReadImage(str(t1_wb))
    moving = sitk.ReadImage(str(adc_wb))
    transform = _translation_registration(
        fixed,
        moving,
        mask=None,
        scales=(1.0, 1.0, 500.0),
        shrink_factors=(4, 2, 1),
        smoothing_sigmas=(2, 1, 0),
    )
    moved_adc = _apply_transform(moving, fixed, transform)
    sitk.WriteImage(moved_adc, str(adc_wb), True)
