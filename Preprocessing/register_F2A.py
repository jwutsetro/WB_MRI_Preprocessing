from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk

from Preprocessing.elastix_resample import resample_moving_to_fixed_crop, transform_from_elastix_parameter_maps
from Preprocessing.registration import (
    _compute_overlap_indices,
    _extract_overlap_images,
    _run_elastix,
)


@dataclass(frozen=True)
class F2APaths:
    """Resolved filesystem inputs for the F2A (functional-to-anatomical) step."""

    patient_dir: Path
    anatomical_wb: Path
    adc_dir: Path
    series_dirs: List[Path]


def _body_sampling_mask(
    fixed_roi: sitk.Image,
    moving_roi: sitk.Image,
    *,
    fixed_threshold: float,
    moving_threshold: float,
    min_voxels: int,
) -> sitk.Image:
    fixed_arr = sitk.GetArrayFromImage(fixed_roi).astype(np.float32)
    moving_arr = sitk.GetArrayFromImage(moving_roi).astype(np.float32)
    mask_arr = (fixed_arr > fixed_threshold) & (moving_arr > moving_threshold)
    if int(mask_arr.sum()) < int(min_voxels):
        mask_arr = np.ones_like(mask_arr, dtype=np.uint8)
    else:
        mask_arr = mask_arr.astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(fixed_roi)
    return mask


def register_functional_to_anatomical(
    patient_dir: Path,
    *,
    b_value: str = "1000",
    anatomical_wb: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    apply_to_all_bvalues: bool = True,
    t1_body_threshold: float = 20.0,
    adc_body_threshold: float = 1.0,
) -> Path:
    """Register per-station DWI volumes to the whole-body anatomical image (translation-only).

    Uses `ADC` stations to estimate translation transforms to the anatomical WB image and
    applies the same transform to other DWI b-values (station-matched) when present.

    Output layout mirrors the input DWI layout (`<bvalue>/<station>.nii.gz`).
    """
    resolved = _resolve_f2a_paths(
        patient_dir=patient_dir,
        b_value=b_value,
        anatomical_wb=anatomical_wb,
        apply_to_all_bvalues=apply_to_all_bvalues,
    )
    out_root = output_dir if output_dir is not None else patient_dir / "_F2A"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_params").mkdir(parents=True, exist_ok=True)

    fixed = sitk.ReadImage(str(resolved.anatomical_wb))
    station_files = _sorted_station_files(resolved.adc_dir)
    if not station_files:
        raise FileNotFoundError(f"No station NIfTI files found in {resolved.adc_dir}")

    for station_path in station_files:
        station_name = _station_key(station_path)
        moving_adc = sitk.ReadImage(str(station_path))
        overlap = _compute_overlap_indices(fixed, moving_adc)
        if overlap is None:
            print(f"[register_F2A] Skip station {station_name}: no overlap with anatomical WB")
            continue

        fixed_roi, moving_roi = _extract_overlap_images(fixed, moving_adc, overlap)
        mask = _body_sampling_mask(
            fixed_roi,
            moving_roi,
            fixed_threshold=t1_body_threshold,
            moving_threshold=adc_body_threshold,
            min_voxels=2000,
        )

        param_maps = _run_elastix(
            fixed=fixed_roi,
            moving=moving_roi,
            mask=mask,
            parameter_files=("F2A_Translation_MI.txt",),
            # The ROI is used for metric computation; resampling is handled explicitly
            # to ensure correct fixed-space geometry.
            output_reference=moving_adc,
            moving_origin=None,
            initial_transform=None,
            initial_transform_file=None,
        )
        transform = transform_from_elastix_parameter_maps(param_maps)

        param_path = out_root / "_params" / f"{station_name}_F2A.txt"
        try:
            sitk.WriteParameterFile(param_maps[-1], str(param_path))
        except Exception:
            pass

        for series_dir in resolved.series_dirs:
            src = series_dir / station_path.name
            if not src.exists():
                continue
            out = out_root / series_dir.name / station_path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            moving_img = sitk.ReadImage(str(src))
            result = resample_moving_to_fixed_crop(moving_img, fixed, transform, interpolator=sitk.sitkLinear)
            sitk.WriteImage(result, str(out), True)

    return out_root


def register_functional_to_anatomical_for_root(
    root_dir: Path,
    *,
    b_value: str = "1000",
    output_subdir: str = "_F2A",
    apply_to_all_bvalues: bool = True,
    t1_body_threshold: float = 20.0,
    adc_body_threshold: float = 1.0,
) -> None:
    """Run `register_functional_to_anatomical` across all patient folders under `root_dir`."""
    patients = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        try:
            out = register_functional_to_anatomical(
                patient,
                b_value=b_value,
                output_dir=patient / output_subdir,
                apply_to_all_bvalues=apply_to_all_bvalues,
                t1_body_threshold=t1_body_threshold,
                adc_body_threshold=adc_body_threshold,
            )
            print(f"[register_F2A] {patient.name}: wrote {out}")
        except Exception as exc:
            print(f"[register_F2A] {patient.name}: failed ({exc})")


def _resolve_f2a_paths(
    *,
    patient_dir: Path,
    b_value: str,
    anatomical_wb: Optional[Path],
    apply_to_all_bvalues: bool,
) -> F2APaths:
    anatomical = anatomical_wb if anatomical_wb is not None else patient_dir / "T1.nii.gz"
    if not anatomical.exists():
        raise FileNotFoundError(
            f"Anatomical WB image not found at {anatomical}. "
            "Run `reconstruct_anatomical` first or pass `--anatomical-wb`."
        )

    adc_dir = patient_dir / "ADC"
    if not adc_dir.is_dir():
        raise FileNotFoundError(f"ADC directory not found under {patient_dir} (expected <patient>/ADC/ stations).")

    if apply_to_all_bvalues:
        series_dirs = _list_bvalue_dirs(patient_dir) + [adc_dir]
    else:
        selected = _resolve_bvalue_dir(patient_dir, b_value)
        if not selected.is_dir():
            raise FileNotFoundError(f"b{b_value} directory not found under {patient_dir}")
        series_dirs = [selected, adc_dir]

    series_dirs = _sorted_series_dirs(series_dirs)
    return F2APaths(
        patient_dir=patient_dir,
        anatomical_wb=anatomical,
        adc_dir=adc_dir,
        series_dirs=series_dirs,
    )


def _list_bvalue_dirs(patient_dir: Path) -> List[Path]:
    dirs = [p for p in patient_dir.iterdir() if p.is_dir() and _is_bvalue(p.name)]
    return sorted(dirs, key=lambda p: float(p.name))

def _sorted_series_dirs(dirs: List[Path]) -> List[Path]:
    def _key(p: Path) -> tuple[int, float, str]:
        if _is_bvalue(p.name):
            return (0, float(p.name), p.name)
        return (1, 0.0, p.name)

    return sorted({p.resolve(): p for p in dirs}.values(), key=_key)


def _resolve_bvalue_dir(patient_dir: Path, b_value: str) -> Path:
    try:
        target = float(b_value)
    except ValueError as exc:
        raise ValueError(f"b_value must be numeric-like, got: {b_value!r}") from exc

    exact = patient_dir / str(int(target)) if target.is_integer() else patient_dir / str(target)
    if exact.is_dir():
        return exact
    for d in patient_dir.iterdir():
        if d.is_dir() and _is_bvalue(d.name):
            if float(d.name) == target:
                return d
    numeric = ", ".join(sorted((p.name for p in _list_bvalue_dirs(patient_dir)), key=float))
    raise FileNotFoundError(f"Could not find b{b_value} under {patient_dir}. Found numeric dirs: [{numeric}]")


def _sorted_station_files(modality_dir: Path) -> List[Path]:
    files = sorted(modality_dir.glob("*.nii*"))
    return sorted(files, key=lambda p: (_station_index(p), p.name))


def _station_index(path: Path) -> int:
    try:
        return int(_station_key(path))
    except Exception:
        return 10**9


def _station_key(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _is_bvalue(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register DWI stations to anatomical WB (F2A), using ADC as the driver.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--patient-dir", type=Path, help="Single patient output folder.")
    scope.add_argument("--root-dir", type=Path, help="Root directory containing patient folders.")
    parser.add_argument("--b-value", default="1000", help="When not propagating, which DWI b-value directory to write (default: 1000).")
    parser.add_argument("--anatomical-wb", type=Path, default=None, help="Path to anatomical WB image (default: <patient>/T1.nii.gz).")
    parser.add_argument("--output-subdir", default="_F2A", help="Output subdirectory under patient folder (default: _F2A).")
    parser.add_argument("--only-b1000", action="store_true", help="Only write a single b-value (use --b-value) instead of all b-values.")
    parser.add_argument("--t1-threshold", type=float, default=20.0, help="Fixed (T1) body threshold for sampling mask (default: 20).")
    parser.add_argument("--adc-threshold", type=float, default=1.0, help="Moving (ADC) body threshold for sampling mask (default: 1).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry-point for functional-to-anatomical station registration."""
    args = _parse_args(argv)
    if args.patient_dir is not None:
        out = register_functional_to_anatomical(
            args.patient_dir,
            b_value=args.b_value,
            anatomical_wb=args.anatomical_wb,
            output_dir=args.patient_dir / args.output_subdir,
            apply_to_all_bvalues=not args.only_b1000,
            t1_body_threshold=args.t1_threshold,
            adc_body_threshold=args.adc_threshold,
        )
        print(f"[register_F2A] Wrote {out}")
        return
    register_functional_to_anatomical_for_root(
        args.root_dir,
        b_value=args.b_value,
        output_subdir=args.output_subdir,
        apply_to_all_bvalues=not args.only_b1000,
        t1_body_threshold=args.t1_threshold,
        adc_body_threshold=args.adc_threshold,
    )


if __name__ == "__main__":
    main()
