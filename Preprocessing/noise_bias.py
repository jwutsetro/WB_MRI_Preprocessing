from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import SimpleITK as sitk

from Preprocessing.adc import compute_body_mask
from Preprocessing.config import NoiseBiasConfig

def _is_bvalue_dir(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _ensure_uint8_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if (
        mask.GetSize() != reference.GetSize()
        or mask.GetSpacing() != reference.GetSpacing()
        or mask.GetOrigin() != reference.GetOrigin()
        or mask.GetDirection() != reference.GetDirection()
    ):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask = resampler.Execute(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(reference)
    return mask


def estimate_log_bias_field_n4(
    image: sitk.Image,
    *,
    mask: sitk.Image,
    shrink_factor: int,
    max_iterations: Iterable[int],
    convergence_threshold: float,
) -> sitk.Image:
    """Estimate the log bias field using N4 bias correction.

    Returns a log-bias-field image defined on the input `image` geometry.
    """
    image_f = sitk.Cast(image, sitk.sitkFloat32)
    mask_u8 = _ensure_uint8_mask(mask, image_f)
    if shrink_factor > 1:
        factors = [int(shrink_factor)] * 3
        image_small = sitk.Shrink(image_f, factors)
        mask_small = sitk.Shrink(mask_u8, factors)
    else:
        image_small = image_f
        mask_small = mask_u8

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([int(v) for v in max_iterations])
    n4.SetConvergenceThreshold(float(convergence_threshold))
    corrected_small = n4.Execute(image_small, mask_small)

    # SimpleITK wheels vary: some expose GetLogBiasFieldAsImage, others don't.
    try:
        getter = getattr(n4, "GetLogBiasFieldAsImage")
    except AttributeError:
        getter = None

    if getter is not None:
        log_field = getter(image_f)
        log_field.CopyInformation(image_f)
        return log_field

    # Fallback: derive log-field from (input / corrected) on the resolution N4 ran on, then resample.
    corrected_safe = sitk.Add(sitk.Cast(corrected_small, sitk.sitkFloat32), 1e-6)
    image_small_f = sitk.Cast(image_small, sitk.sitkFloat32)
    ratio = sitk.Divide(image_small_f, corrected_safe)
    log_field_small = sitk.Log(ratio)
    log_field_small = sitk.Multiply(log_field_small, sitk.Cast(mask_small, sitk.sitkFloat32))
    log_field_small.CopyInformation(image_small)

    if shrink_factor > 1:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_f)
        resampler.SetInterpolator(sitk.sitkLinear)
        log_field = resampler.Execute(log_field_small)
    else:
        log_field = log_field_small
    log_field.CopyInformation(image_f)
    return log_field


def apply_log_bias_field(image: sitk.Image, log_bias_field: sitk.Image) -> sitk.Image:
    """Apply a log bias field to an image (bias-correct via division)."""
    image_f = sitk.Cast(image, sitk.sitkFloat32)
    field = sitk.Cast(log_bias_field, sitk.sitkFloat32)
    if (
        field.GetSize() != image_f.GetSize()
        or field.GetSpacing() != image_f.GetSpacing()
        or field.GetOrigin() != image_f.GetOrigin()
        or field.GetDirection() != image_f.GetDirection()
    ):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_f)
        resampler.SetInterpolator(sitk.sitkLinear)
        field = resampler.Execute(field)
    corrected = sitk.Divide(image_f, sitk.Exp(field))
    corrected.CopyInformation(image)
    return corrected


def apply_body_mask(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """Zero-out voxels outside the body mask."""
    image_f = sitk.Cast(image, sitk.sitkFloat32)
    mask_u8 = _ensure_uint8_mask(mask, image_f)
    masked = sitk.Multiply(image_f, sitk.Cast(mask_u8, sitk.sitkFloat32))
    masked.CopyInformation(image)
    return masked


def _iter_station_files(modality_dir: Path) -> list[Path]:
    return sorted([p for p in modality_dir.glob("*.nii*") if p.is_file()])


def _write_mask(mask_root: Path, modality: str, station_file: Path, mask: sitk.Image) -> None:
    out_dir = mask_root / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / station_file.name
    sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(out_path), True)


def _mask_for_image(image: sitk.Image, cfg: NoiseBiasConfig) -> sitk.Image:
    return compute_body_mask(
        image,
        smoothing_sigma=float(cfg.mask_smoothing_sigma),
        closing_radius=int(cfg.mask_closing_radius),
        dilation_radius=int(cfg.mask_dilation_radius),
    )


def _process_anatomical(patient_dir: Path, cfg: NoiseBiasConfig, mask_root: Optional[Path]) -> None:
    for modality_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
        name = modality_dir.name
        if name.lower() == "adc" or name.lower() == cfg.mask_dir_name.lower():
            continue
        if name.startswith("_") or _is_bvalue_dir(name) or name.lower() == "dwi":
            continue
        for station_file in _iter_station_files(modality_dir):
            image = sitk.ReadImage(str(station_file))
            mask = _mask_for_image(image, cfg)
            if mask_root is not None:
                _write_mask(mask_root, name, station_file, mask)
            log_field = estimate_log_bias_field_n4(
                image,
                mask=mask,
                shrink_factor=int(cfg.n4_shrink_factor_anatomical),
                max_iterations=cfg.n4_max_iterations,
                convergence_threshold=float(cfg.n4_convergence_threshold),
            )
            corrected = apply_log_bias_field(image, log_field)
            if cfg.apply_body_mask:
                corrected = apply_body_mask(corrected, mask)
            sitk.WriteImage(corrected, str(station_file), True)


def _choose_dwi_reference_dir(patient_dir: Path, cfg: NoiseBiasConfig) -> Optional[Path]:
    b_dirs = [p for p in patient_dir.iterdir() if p.is_dir() and _is_bvalue_dir(p.name)]
    if not b_dirs:
        return None
    if cfg.dwi_reference.lower() == "b0":
        for d in b_dirs:
            if abs(float(d.name)) < 1e-6:
                return d
    return min(b_dirs, key=lambda p: float(p.name))


def _process_dwi(patient_dir: Path, cfg: NoiseBiasConfig, mask_root: Optional[Path]) -> None:
    b_dirs = sorted([p for p in patient_dir.iterdir() if p.is_dir() and _is_bvalue_dir(p.name)], key=lambda p: float(p.name))
    if not b_dirs:
        return
    ref_dir = _choose_dwi_reference_dir(patient_dir, cfg)
    if ref_dir is None:
        return

    station_files = _iter_station_files(ref_dir)
    if not station_files:
        return

    # Estimate per-station bias fields from the reference (lowest-b) volume and apply to all b-values.
    for station_file in station_files:
        b0_img = sitk.ReadImage(str(station_file))
        mask = _mask_for_image(b0_img, cfg)
        if mask_root is not None:
            _write_mask(mask_root, "dwi", station_file, mask)
        log_field = estimate_log_bias_field_n4(
            b0_img,
            mask=mask,
            shrink_factor=int(cfg.n4_shrink_factor_dwi),
            max_iterations=cfg.n4_max_iterations,
            convergence_threshold=float(cfg.n4_convergence_threshold),
        )
        for b_dir in b_dirs:
            target = b_dir / station_file.name
            if not target.exists():
                continue
            image = sitk.ReadImage(str(target))
            corrected = apply_log_bias_field(image, log_field)
            if cfg.apply_body_mask:
                corrected = apply_body_mask(corrected, mask)
            sitk.WriteImage(corrected, str(target), True)


def process_patient(patient_dir: Path, cfg: Optional[NoiseBiasConfig] = None) -> None:
    """Run bias correction + optional body masking for a patient output directory."""
    cfg = cfg or NoiseBiasConfig()
    mask_root: Optional[Path] = None
    if cfg.save_masks:
        mask_root = patient_dir / cfg.mask_dir_name
        mask_root.mkdir(parents=True, exist_ok=True)

    if cfg.apply_to_anatomical:
        _process_anatomical(patient_dir, cfg, mask_root)
    if cfg.apply_to_dwi:
        _process_dwi(patient_dir, cfg, mask_root)


def process_root(root_dir: Path, cfg: Optional[NoiseBiasConfig] = None) -> None:
    """Run bias correction + optional masking for all patients in root_dir."""
    patients = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        print(f"[Noise/Bias] Processing {patient.name}")
        process_patient(patient, cfg=cfg)
