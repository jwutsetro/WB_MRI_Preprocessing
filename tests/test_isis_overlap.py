"""
CLI helper to inspect mean intensities in overlap regions between consecutive stations.

Usage:
    python -m tests.test_isis_overlap /path/to/output/<patient>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


def _station_key(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _load_stations(modality_dir: Path) -> List[dict]:
    stations: List[dict] = []
    for path in sorted(modality_dir.glob("*.nii*")):
        try:
            img = sitk.ReadImage(str(path))
        except Exception:
            continue
        origin_z = img.GetOrigin()[2]
        stations.append({"path": path, "image": img, "origin_z": origin_z, "station": _station_key(path)})
    stations.sort(key=lambda s: (s["origin_z"], s["path"].stem))
    return stations


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


def _overlap_mean(image: sitk.Image, index: Sequence[Tuple[int, int]]) -> float:
    (xf0, xf1), (yf0, yf1), (zf0, zf1) = index
    size = (xf1 - xf0, yf1 - yf0, zf1 - zf0)
    roi = sitk.RegionOfInterest(image, size=size, index=(xf0, yf0, zf0))
    arr = sitk.GetArrayFromImage(roi)
    return float(arr.mean()) if arr.size else float("nan")


def _shared_nonzero_means(
    fixed: sitk.Image,
    moving: sitk.Image,
    idx_f: Sequence[Tuple[int, int]],
    idx_m: Sequence[Tuple[int, int]],
) -> Tuple[float, float, int]:
    """Mean intensities using only voxels where both overlap arrays are non-zero."""
    (xf0, xf1), (yf0, yf1), (zf0, zf1) = idx_f
    (xm0, xm1), (ym0, ym1), (zm0, zm1) = idx_m
    size_f = (xf1 - xf0, yf1 - yf0, zf1 - zf0)
    size_m = (xm1 - xm0, ym1 - ym0, zm1 - zm0)
    roi_f = sitk.RegionOfInterest(fixed, size=size_f, index=(xf0, yf0, zf0))
    roi_m = sitk.RegionOfInterest(moving, size=size_m, index=(xm0, ym0, zm0))
    arr_f = sitk.GetArrayFromImage(roi_f)
    arr_m = sitk.GetArrayFromImage(roi_m)
    mask = (arr_f > 0) & (arr_m > 0)
    if not mask.any():
        return float("nan"), float("nan"), 0
    shared = mask.sum()
    return float(arr_f[mask].mean()), float(arr_m[mask].mean()), int(shared)


def check_overlap_means(patient_dir: Path) -> None:
    """Print mean intensities in overlap regions for each modality under a patient folder."""
    modalities = [p for p in patient_dir.iterdir() if p.is_dir()]
    for modality_dir in sorted(modalities):
        stations = _load_stations(modality_dir)
        if len(stations) < 2:
            continue
        print(f"\n{modality_dir.name}:")
        for fixed, moving in zip(stations[:-1], stations[1:]):
            overlap = _compute_overlap_indices(fixed["image"], moving["image"])
            if overlap is None:
                print(f"  {fixed['station']}->{moving['station']}: no overlap")
                continue
            idx_f, idx_m = overlap
            mean_fixed = _overlap_mean(fixed["image"], idx_f)
            mean_moving = _overlap_mean(moving["image"], idx_m)
            diff = mean_moving - mean_fixed
            ratio = mean_moving / mean_fixed if mean_fixed not in (0, float("nan")) else float("nan")
            nz_fixed, nz_moving, nz_count = _shared_nonzero_means(fixed["image"], moving["image"], idx_f, idx_m)
            nz_ratio = nz_moving / nz_fixed if nz_fixed not in (0, float("nan")) else float("nan")
            print(
                f"  {fixed['station']}->{moving['station']}: "
                f"mean_fixed={mean_fixed:.4f}, mean_moving={mean_moving:.4f}, "
                f"diff={diff:.4f}, ratio={ratio:.4f}; "
                f"nz_mean_fixed={nz_fixed:.4f}, nz_mean_moving={nz_moving:.4f}, "
                f"nz_ratio={nz_ratio:.4f}, shared_nz_voxels={nz_count}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect overlap mean intensities between stations.")
    parser.add_argument("patient_dir", type=Path, help="Path to patient output folder (containing modality subfolders).")
    return parser.parse_args()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args()
    check_overlap_means(args.patient_dir)


if __name__ == "__main__":
    main()
