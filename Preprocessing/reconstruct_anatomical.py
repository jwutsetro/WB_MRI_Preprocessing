from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk


def reconstruct_anatomical(
    patient_dir: Path,
    modality: str = "T1",
    *,
    keep_station_dir: bool = False,
) -> Optional[Path]:
    """Reconstruct a whole-body anatomical volume by feather-merging station images.

    This is a debug-friendly variant of `Preprocessing.merge_wb.merge_patient` which
    merges only a single anatomical modality (default: `T1`) and can optionally keep
    the station directory for inspection.
    """
    modality_dir = patient_dir / modality
    if not modality_dir.is_dir():
        return None
    station_files = sorted(modality_dir.glob("*.nii*"))
    if not station_files:
        return None

    stations = _load_stations(station_files)
    if not stations:
        return None
    reference = _reference_image(stations)
    merged = _feather_merge(stations, reference)
    out_path = patient_dir / f"{modality}.nii.gz"
    sitk.WriteImage(merged, str(out_path), True)

    if not keep_station_dir:
        shutil.rmtree(modality_dir, ignore_errors=True)
    return out_path


def reconstruct_anatomical_for_root(
    root_dir: Path,
    modality: str = "T1",
    *,
    keep_station_dir: bool = False,
) -> None:
    """Run `reconstruct_anatomical` across all patient subfolders under `root_dir`."""
    patients = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        out = reconstruct_anatomical(patient, modality=modality, keep_station_dir=keep_station_dir)
        if out is None:
            print(f"[reconstruct_anatomical] Skip {patient.name}: no {modality}/ stations found")
        else:
            print(f"[reconstruct_anatomical] Wrote {out}")


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
    min_origin = [min(s["origin"][i] for s in stations) for i in range(3)]
    max_extent = [max(s["extent"][i] for s in stations) for i in range(3)]
    size = [int(math.ceil((max_extent[i] - min_origin[i]) / spacing[i])) for i in range(3)]
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

        accum = sitk.Add(accum, sitk.Multiply(resampled_image, weight_resampled))
        weight_sum = sitk.Add(weight_sum, weight_resampled)

    merged = sitk.Divide(accum, sitk.Add(weight_sum, 1e-6))
    merged.CopyInformation(reference)
    return merged


def _station_weight_image(
    station: Dict,
    prev_extent: Optional[tuple[float, float]],
    next_extent: Optional[tuple[float, float]],
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
            ramp_up = np.clip((z_coords - overlap_start) / length, 0.0, 1.0)
            weights = np.where((z_coords >= overlap_start) & (z_coords <= overlap_end), ramp_up, weights)

    if next_extent is not None:
        overlap_start = max(origin[2], next_extent[0])
        overlap_end = min(origin[2] + size[2] * spacing[2], next_extent[1])
        if overlap_end > overlap_start:
            length = overlap_end - overlap_start
            ramp_down = np.clip((overlap_end - z_coords) / length, 0.0, 1.0)
            weights = np.where(
                (z_coords >= overlap_start) & (z_coords <= overlap_end),
                np.minimum(weights, ramp_down),
                weights,
            )

    arr = np.ones((size[2], size[1], size[0]), dtype=np.float32)
    arr *= weights[:, None, None]
    weight_img = sitk.GetImageFromArray(arr)
    weight_img.SetSpacing(spacing)
    weight_img.SetOrigin(origin)
    weight_img.SetDirection(station["image"].GetDirection())
    return weight_img


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct anatomical whole-body volume from stations.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--patient-dir", type=Path, help="Single patient output folder (contains T1/ stations).")
    scope.add_argument("--root-dir", type=Path, help="Root directory containing patient folders.")
    parser.add_argument("--modality", default="T1", help="Anatomical modality folder to merge (default: T1).")
    parser.add_argument("--keep-stations", action="store_true", help="Keep the station folder after writing the WB volume.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry-point for reconstructing the anatomical WB volume."""
    args = _parse_args(argv)
    if args.patient_dir is not None:
        out = reconstruct_anatomical(args.patient_dir, modality=args.modality, keep_station_dir=args.keep_stations)
        if out is None:
            raise SystemExit(f"No {args.modality}/ stations found under {args.patient_dir}")
        print(f"[reconstruct_anatomical] Wrote {out}")
        return
    reconstruct_anatomical_for_root(args.root_dir, modality=args.modality, keep_station_dir=args.keep_stations)


if __name__ == "__main__":
    main()

