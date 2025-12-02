from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk


@dataclass
class Overlap:
    start: float
    end: float
    idx_a: Tuple[int, int]
    idx_b: Tuple[int, int]


def _compute_overlap(a: sitk.Image, b: sitk.Image) -> Optional[Overlap]:
    """Compute physical overlap along z-axis assuming aligned orientation."""
    spacing_a = a.GetSpacing()
    spacing_b = b.GetSpacing()
    origin_a = a.GetOrigin()
    origin_b = b.GetOrigin()
    size_a = a.GetSize()
    size_b = b.GetSize()

    start_a = origin_a[2]
    end_a = origin_a[2] + size_a[2] * spacing_a[2]
    start_b = origin_b[2]
    end_b = origin_b[2] + size_b[2] * spacing_b[2]

    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    if overlap_end <= overlap_start:
        return None

    idx_start_a = int(round((overlap_start - start_a) / spacing_a[2]))
    idx_end_a = idx_start_a + int(round((overlap_end - overlap_start) / spacing_a[2]))
    idx_start_b = int(round((overlap_start - start_b) / spacing_b[2]))
    idx_end_b = idx_start_b + int(round((overlap_end - overlap_start) / spacing_b[2]))

    idx_end_a = min(idx_end_a, size_a[2])
    idx_end_b = min(idx_end_b, size_b[2])

    return Overlap(
        start=overlap_start,
        end=overlap_end,
        idx_a=(idx_start_a, idx_end_a),
        idx_b=(idx_start_b, idx_end_b),
    )


def _scale_to_match(reference: sitk.Image, target: sitk.Image) -> sitk.Image:
    overlap = _compute_overlap(reference, target)
    if overlap is None:
        return target
    ref_arr = sitk.GetArrayFromImage(reference)[overlap.idx_a[0] : overlap.idx_a[1], ...]
    tar_arr = sitk.GetArrayFromImage(target)[overlap.idx_b[0] : overlap.idx_b[1], ...]
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
    images = [sitk.ReadImage(str(f)) for f in files]
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
