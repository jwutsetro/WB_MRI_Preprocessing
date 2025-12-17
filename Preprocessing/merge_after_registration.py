from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from Preprocessing.merge_wb import merge_station_folders
from Preprocessing.reconstruct_anatomical import reconstruct_anatomical


def merge_wb_after_registration(
    patient_dir: Path,
    *,
    anatomical_modality: str = "T1",
    functional_subdir: str = "_S2S",
    keep_station_folders: bool = False,
) -> List[Path]:
    """Merge WB outputs for a debug registration workflow.

    Steps:
      1) Reconstruct the anatomical WB image from `<patient>/<anatomical_modality>/...` stations.
      2) Merge registered functional stations under `<patient>/<functional_subdir>/<bvalue>/...` into WB volumes.

    Returns a list of paths written by step (2). The anatomical WB path is written as
    `<patient>/<anatomical_modality>.nii.gz`.
    """
    anatomical = reconstruct_anatomical(
        patient_dir,
        modality=anatomical_modality,
        keep_station_dir=keep_station_folders,
    )
    if anatomical is None:
        raise FileNotFoundError(f"No {anatomical_modality}/ stations found under {patient_dir}")

    functional_root = patient_dir / functional_subdir
    if not functional_root.is_dir():
        raise FileNotFoundError(f"Functional registration output not found: {functional_root}")

    written = merge_station_folders(
        functional_root,
        output_dir=functional_root,
        remove_station_folders=not keep_station_folders,
    )
    return written


def merge_wb_after_registration_for_root(
    root_dir: Path,
    *,
    anatomical_modality: str = "T1",
    functional_subdir: str = "_S2S",
    keep_station_folders: bool = False,
) -> None:
    """Run `merge_wb_after_registration` across all patient folders under `root_dir`."""
    patients = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        try:
            written = merge_wb_after_registration(
                patient,
                anatomical_modality=anatomical_modality,
                functional_subdir=functional_subdir,
                keep_station_folders=keep_station_folders,
            )
            print(f"[merge_after_registration] {patient.name}: wrote {len(written)} WB functional volumes")
        except Exception as exc:
            print(f"[merge_after_registration] {patient.name}: failed ({exc})")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge WB outputs after debug registration (anatomical then functional).")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--patient-dir", type=Path, help="Single patient output folder.")
    scope.add_argument("--root-dir", type=Path, help="Root directory containing patient output folders.")
    parser.add_argument("--anatomical-modality", default="T1", help="Anatomical station folder (default: T1).")
    parser.add_argument("--functional-subdir", default="_S2S", help="Functional registration subdir (default: _S2S).")
    parser.add_argument("--keep-stations", action="store_true", help="Keep station folders after merge.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry-point for merging WB outputs after debug registration."""
    args = _parse_args(argv)
    if args.patient_dir is not None:
        written = merge_wb_after_registration(
            args.patient_dir,
            anatomical_modality=args.anatomical_modality,
            functional_subdir=args.functional_subdir,
            keep_station_folders=args.keep_stations,
        )
        print(f"[merge_after_registration] Wrote {len(written)} WB functional volumes")
        return
    merge_wb_after_registration_for_root(
        args.root_dir,
        anatomical_modality=args.anatomical_modality,
        functional_subdir=args.functional_subdir,
        keep_station_folders=args.keep_stations,
    )


if __name__ == "__main__":
    main()

