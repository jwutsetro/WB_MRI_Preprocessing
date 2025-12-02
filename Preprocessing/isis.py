from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk


@dataclass
class Overlap:
    idx_a: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]  # (x, y, z)
    idx_b: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def _compute_overlap(a: sitk.Image, b: sitk.Image) -> Optional[Overlap]:
    """Compute 3D overlap between two aligned images and return index ranges per axis (x, y, z)."""
    spacing_a = a.GetSpacing()
    spacing_b = b.GetSpacing()
    origin_a = a.GetOrigin()
    origin_b = b.GetOrigin()
    size_a = a.GetSize()
    size_b = b.GetSize()

    idx_a: List[Tuple[int, int]] = []
    idx_b: List[Tuple[int, int]] = []

    for axis in range(3):
        start = max(origin_a[axis], origin_b[axis])
        end = min(origin_a[axis] + size_a[axis] * spacing_a[axis], origin_b[axis] + size_b[axis] * spacing_b[axis])
        if end <= start:
            return None
        start_idx_a = int(np.floor((start - origin_a[axis]) / spacing_a[axis]))
        end_idx_a = int(np.ceil((end - origin_a[axis]) / spacing_a[axis]))
        start_idx_b = int(np.floor((start - origin_b[axis]) / spacing_b[axis]))
        end_idx_b = int(np.ceil((end - origin_b[axis]) / spacing_b[axis]))

        end_idx_a = min(end_idx_a, size_a[axis])
        end_idx_b = min(end_idx_b, size_b[axis])

        len_a = max(0, end_idx_a - start_idx_a)
        len_b = max(0, end_idx_b - start_idx_b)
        common = min(len_a, len_b)
        if common == 0:
            return None

        idx_a.append((start_idx_a, start_idx_a + common))
        idx_b.append((start_idx_b, start_idx_b + common))

    return Overlap(idx_a=tuple(idx_a), idx_b=tuple(idx_b))


def _crop_zero_padding_z(image: sitk.Image, threshold: float = 0.0) -> sitk.Image:
    """Remove leading/trailing all-zero slices along z while preserving spacing/direction."""
    arr = sitk.GetArrayFromImage(image)  # z, y, x
    non_zero = np.abs(arr) > threshold
    z_any = non_zero.any(axis=(1, 2))
    if not z_any.any():
        return image
    z_indices = np.where(z_any)[0]
    z_start, z_end = int(z_indices[0]), int(z_indices[-1] + 1)
    if z_start == 0 and z_end == arr.shape[0]:
        return image

    cropped_arr = arr[z_start:z_end, :, :]
    cropped = sitk.GetImageFromArray(cropped_arr)
    cropped = sitk.Cast(cropped, image.GetPixelID())
    cropped.SetSpacing(image.GetSpacing())
    cropped.SetDirection(image.GetDirection())
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    cropped.SetOrigin((origin[0], origin[1], origin[2] + z_start * spacing[2]))
    return cropped


def _scale_to_match(reference: sitk.Image, target: sitk.Image) -> sitk.Image:
    overlap = _compute_overlap(reference, target)
    if overlap is None:
        return target
    ref_np = sitk.GetArrayFromImage(reference)
    tar_np = sitk.GetArrayFromImage(target)

    # numpy arrays are ordered (z, y, x) while indices are stored (x, y, z)
    x_a, y_a, z_a = overlap.idx_a
    x_b, y_b, z_b = overlap.idx_b

    ref_arr = ref_np[z_a[0] : z_a[1], y_a[0] : y_a[1], x_a[0] : x_a[1]]
    tar_arr = tar_np[z_b[0] : z_b[1], y_b[0] : y_b[1], x_b[0] : x_b[1]]

    shared_mask = (ref_arr > 0) & (tar_arr > 0)
    ref_vals = ref_arr[shared_mask]
    tar_vals = tar_arr[shared_mask]
    if ref_vals.size == 0 or tar_vals.size == 0:
        return target
    mean_ref = float(ref_vals.mean())
    mean_tar = float(tar_vals.mean())
    if mean_tar < 1e-6:
        return target
    scale = mean_ref / mean_tar
    scaled = sitk.Cast(target, sitk.sitkFloat32) * scale
    scaled.CopyInformation(target)
    return scaled


def _standardize_modality(mod_dir: Path) -> None:
    files = sorted(mod_dir.glob("*.nii*"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if len(files) <= 1:
        return
    images = [_crop_zero_padding_z(sitk.ReadImage(str(f))) for f in files]
    center_idx = len(images) // 2

    # propagate upwards (higher indices)
    ref = images[center_idx]
    for i in range(center_idx + 1, len(images)):
        scaled = _scale_to_match(ref, images[i])
        sitk.WriteImage(scaled, str(files[i]), True)
        ref = scaled
    # propagate downwards
    ref = images[center_idx]
    for i in range(center_idx - 1, -1, -1):
        scaled = _scale_to_match(ref, images[i])
        sitk.WriteImage(scaled, str(files[i]), True)
        ref = scaled


def standardize_patient(patient_dir: Path, skip_modalities: Iterable[str] = ("ADC",)) -> None:
    """Apply inter-station intensity standardisation per modality for a patient."""
    skip_set = {m.lower() for m in skip_modalities}
    for mod_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
        if mod_dir.name.lower() in skip_set:
            continue
        _standardize_modality(mod_dir)


def standardize_root(root_dir: Path, skip_modalities: Iterable[str] = ("ADC",)) -> None:
    patients = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        print(f"[ISIS] Processing {patient.name}")
        standardize_patient(patient, skip_modalities=skip_modalities)
