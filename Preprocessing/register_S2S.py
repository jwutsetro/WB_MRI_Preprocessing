from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk

from Preprocessing.elastix_resample import resample_moving_to_fixed_crop, transform_from_elastix_parameter_map
from Preprocessing.registration import (
    _build_initial_transform,
    _compute_overlap_indices,
    _extract_overlap_images,
    _overlap_sampling_mask,
    _run_elastix,
    _translation_from_param_maps,
)


def register_station_to_station(
    patient_dir: Path,
    *,
    b_value: str = "1000",
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    apply_to_all_bvalues: bool = True,
) -> Path:
    """Register DWI stations to each other using overlap-only translation (S2S).

    This step uses `b1000` to estimate transforms between consecutive stations using
    only their overlap region, then applies the same transforms to other DWI b-values
    (station-matched) when present.

    Typical usage is to run this after F2A and point `input_dir` at the F2A outputs
    (default: `<patient>/_F2A`).
    """
    in_root = input_dir if input_dir is not None else patient_dir / "_F2A"
    if not in_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_root}")

    out_root = output_dir if output_dir is not None else patient_dir / "_S2S"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_params").mkdir(parents=True, exist_ok=True)

    b1000_dir = _resolve_bvalue_dir(in_root, b_value)
    b_dirs = _list_bvalue_dirs(in_root) if apply_to_all_bvalues else [b1000_dir]
    b_dirs = sorted(set(b_dirs), key=lambda p: float(p.name))

    stations = _load_station_dicts(b1000_dir)
    if len(stations) <= 1:
        _copy_all_stations(b_dirs, out_root)
        return out_root

    reference_space = _load_reference_space(patient_dir, stations[0]["image"])

    center_idx = (len(stations) - 1) // 2
    _register_chain(
        stations[center_idx:],
        b_dirs=b_dirs,
        out_root=out_root,
        param_dir=out_root / "_params",
        fixed_space=reference_space,
    )
    _register_chain(
        list(reversed(stations[: center_idx + 1])),
        b_dirs=b_dirs,
        out_root=out_root,
        param_dir=out_root / "_params",
        fixed_space=reference_space,
    )
    return out_root


def register_station_to_station_for_root(
    root_dir: Path,
    *,
    b_value: str = "1000",
    input_subdir: str = "_F2A",
    output_subdir: str = "_S2S",
    apply_to_all_bvalues: bool = True,
) -> None:
    """Run `register_station_to_station` across all patient folders under `root_dir`."""
    patients = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        try:
            out = register_station_to_station(
                patient,
                b_value=b_value,
                input_dir=patient / input_subdir,
                output_dir=patient / output_subdir,
                apply_to_all_bvalues=apply_to_all_bvalues,
            )
            print(f"[register_S2S] {patient.name}: wrote {out}")
        except Exception as exc:
            print(f"[register_S2S] {patient.name}: failed ({exc})")


def _register_chain(
    stations: List[Dict],
    *,
    b_dirs: List[Path],
    out_root: Path,
    param_dir: Path,
    fixed_space: sitk.Image,
) -> None:
    if len(stations) < 2:
        return

    fixed_station = stations[0]
    fixed_img = fixed_station["image"]
    _write_station_outputs(fixed_station["path"].name, b_dirs=b_dirs, out_root=out_root)

    cumulative_translation = np.zeros(3, dtype=float)
    previous_param_file: Optional[Path] = None

    for station in stations[1:]:
        moving_img = station["image"]
        overlap = _compute_overlap_indices(fixed_img, moving_img)
        if overlap is None:
            fixed_img = moving_img
            cumulative_translation = np.zeros(3, dtype=float)
            previous_param_file = None
            _write_station_outputs(station["path"].name, b_dirs=b_dirs, out_root=out_root)
            continue

        fixed_roi, moving_roi = _extract_overlap_images(fixed_img, moving_img, overlap)
        mask = _overlap_sampling_mask(fixed_roi, moving_roi, threshold=0.1, min_voxels=500)
        initial_transform = _build_initial_transform(cumulative_translation, moving_img) if np.any(cumulative_translation) else None
        initial_transform_file = str(previous_param_file) if previous_param_file is not None else None
        param_maps = _run_elastix(
            fixed=fixed_roi,
            moving=moving_roi,
            mask=mask,
            parameter_files=("Euler_S2S_MI.txt",),
            moving_origin=moving_img.GetOrigin(),
            output_reference=moving_img,
            initial_transform=initial_transform,
            initial_transform_file=initial_transform_file,
        )
        transform = transform_from_elastix_parameter_map(param_maps[0])

        station_name = _station_key(station["path"])
        param_file = param_dir / f"{station_name}_S2S_init.txt"
        try:
            sitk.WriteParameterFile(param_maps[0], str(param_file))
            previous_param_file = param_file
        except Exception:
            previous_param_file = None

        cumulative_translation += _translation_from_param_maps(param_maps)

        # Apply to b1000 and propagate the same transform to other b-values.
        for b_dir in b_dirs:
            src = b_dir / station["path"].name
            if not src.exists():
                continue
            out = out_root / b_dir.name / station["path"].name
            out.parent.mkdir(parents=True, exist_ok=True)
            moving_b = sitk.ReadImage(str(src))
            result = resample_moving_to_fixed_crop(moving_b, fixed_space, transform, interpolator=sitk.sitkLinear)
            sitk.WriteImage(result, str(out), True)

        fixed_img = resample_moving_to_fixed_crop(moving_img, fixed_space, transform, interpolator=sitk.sitkLinear)


def _copy_all_stations(b_dirs: List[Path], out_root: Path) -> None:
    for b_dir in b_dirs:
        out_dir = out_root / b_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(b_dir.glob("*.nii*")):
            img = sitk.ReadImage(str(src))
            sitk.WriteImage(img, str(out_dir / src.name), True)


def _write_station_outputs(station_filename: str, *, b_dirs: List[Path], out_root: Path) -> None:
    for b_dir in b_dirs:
        src = b_dir / station_filename
        if not src.exists():
            continue
        dst = out_root / b_dir.name / station_filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = sitk.ReadImage(str(src))
        sitk.WriteImage(img, str(dst), True)


def _load_station_dicts(modality_dir: Path) -> List[Dict]:
    stations: List[Dict] = []
    for path in sorted(modality_dir.glob("*.nii*")):
        try:
            img = sitk.ReadImage(str(path))
        except Exception:
            continue
        stations.append(
            {
                "path": path,
                "image": img,
                "origin_z": float(img.GetOrigin()[2]),
                "station": _station_key(path),
            }
        )
    stations.sort(key=lambda s: (s["origin_z"], _station_index(s["path"]), s["path"].name))
    return stations


def _load_reference_space(patient_dir: Path, fallback: sitk.Image) -> sitk.Image:
    t1 = patient_dir / "T1.nii.gz"
    if t1.exists():
        try:
            return sitk.ReadImage(str(t1))
        except Exception:
            return fallback
    return fallback


def _list_bvalue_dirs(root: Path) -> List[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and _is_bvalue(p.name)]
    return sorted(dirs, key=lambda p: float(p.name))


def _resolve_bvalue_dir(root: Path, b_value: str) -> Path:
    try:
        target = float(b_value)
    except ValueError as exc:
        raise ValueError(f"b_value must be numeric-like, got: {b_value!r}") from exc

    exact = root / str(int(target)) if target.is_integer() else root / str(target)
    if exact.is_dir():
        return exact
    for d in root.iterdir():
        if d.is_dir() and _is_bvalue(d.name):
            if float(d.name) == target:
                return d
    numeric = ", ".join(sorted((p.name for p in _list_bvalue_dirs(root)), key=float))
    raise FileNotFoundError(f"Could not find b{b_value} under {root}. Found numeric dirs: [{numeric}]")


def _station_key(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _station_index(path: Path) -> int:
    try:
        return int(_station_key(path))
    except Exception:
        return 10**9


def _is_bvalue(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register DWI stations to each other (S2S), using b1000 as reference.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--patient-dir", type=Path, help="Single patient output folder.")
    scope.add_argument("--root-dir", type=Path, help="Root directory containing patient folders.")
    parser.add_argument("--b-value", default="1000", help="Reference DWI b-value directory name (default: 1000).")
    parser.add_argument("--input-subdir", default="_F2A", help="Input subdirectory under patient folder (default: _F2A).")
    parser.add_argument("--output-subdir", default="_S2S", help="Output subdirectory under patient folder (default: _S2S).")
    parser.add_argument("--only-b1000", action="store_true", help="Only write b1000 outputs (do not propagate to other b-values).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry-point for station-to-station registration."""
    args = _parse_args(argv)
    if args.patient_dir is not None:
        out = register_station_to_station(
            args.patient_dir,
            b_value=args.b_value,
            input_dir=args.patient_dir / args.input_subdir,
            output_dir=args.patient_dir / args.output_subdir,
            apply_to_all_bvalues=not args.only_b1000,
        )
        print(f"[register_S2S] Wrote {out}")
        return
    register_station_to_station_for_root(
        args.root_dir,
        b_value=args.b_value,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
        apply_to_all_bvalues=not args.only_b1000,
    )


if __name__ == "__main__":
    main()
