from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from Preprocessing.config import load_config
from Preprocessing.dicom_sort import DicomSorter
from Preprocessing.pipeline import PipelineRunner
from Preprocessing.reconstruct_anatomical import reconstruct_anatomical, reconstruct_anatomical_for_root
from Preprocessing.register_F2A import register_functional_to_anatomical, register_functional_to_anatomical_for_root
from Preprocessing.register_S2S import register_station_to_station, register_station_to_station_for_root
from Preprocessing.merge_after_registration import merge_wb_after_registration, merge_wb_after_registration_for_root
from Preprocessing.merge_wb import merge_station_folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WB MRI preprocessing pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run full preprocessing pipeline")
    run_p.add_argument("--config", required=True, type=Path, help="Path to pipeline config YAML")
    run_p.add_argument("--interactive", action="store_true", help="Prompt for unknown sequences")
    run_p.add_argument("--array-index", type=int, default=None, help="Index for job array shard")
    run_p.add_argument("--array-size", type=int, default=None, help="Total size for job array shard")

    scan_p = sub.add_parser("scan-sequences", help="Scan DICOM folders and report unknown sequences")
    scan_p.add_argument("--config", required=True, type=Path, help="Path to pipeline config YAML")
    scan_p.add_argument("--patient-dir", required=True, type=Path, help="Patient directory containing DICOMs")
    scan_p.add_argument("--output", type=Path, default=None, help="Optional JSON output for matched files")

    convert_p = sub.add_parser("convert-dicom", help="Convert DICOMs only using sequence rules")
    convert_p.add_argument("input_dir", type=Path, help="Directory containing patient DICOM folders")
    convert_p.add_argument("output_dir", type=Path, help="Directory to write NIfTI outputs")
    convert_p.add_argument("--config", type=Path, default=Path("config/pipeline.example.yaml"), help="Pipeline config with sequence rules (default: config/pipeline.example.yaml)")
    convert_p.add_argument("--interactive", action="store_true", help="Prompt for unknown sequences")

    recon_p = sub.add_parser("reconstruct-anatomical", help="Reconstruct anatomical WB volume from stations")
    recon_scope = recon_p.add_mutually_exclusive_group(required=True)
    recon_scope.add_argument("--patient-dir", type=Path, help="Single patient output folder (contains T1/ stations).")
    recon_scope.add_argument("--root-dir", type=Path, help="Root directory containing patient output folders.")
    recon_p.add_argument("--modality", default="T1", help="Anatomical modality folder to merge (default: T1).")
    recon_p.add_argument("--keep-stations", action="store_true", help="Keep the station folder after writing the WB volume.")

    f2a_p = sub.add_parser("register-f2a", help="Register DWI stations to anatomical WB using b1000 (rigid)")
    f2a_scope = f2a_p.add_mutually_exclusive_group(required=True)
    f2a_scope.add_argument("--patient-dir", type=Path, help="Single patient output folder.")
    f2a_scope.add_argument("--root-dir", type=Path, help="Root directory containing patient output folders.")
    f2a_p.add_argument("--b-value", default="1000", help="Reference DWI b-value directory name (default: 1000).")
    f2a_p.add_argument("--anatomical-wb", type=Path, default=None, help="Path to anatomical WB image (default: <patient>/T1.nii.gz).")
    f2a_p.add_argument("--output-subdir", default="_F2A", help="Output subdirectory under patient folder (default: _F2A).")
    f2a_p.add_argument("--only-b1000", action="store_true", help="Only write b1000 outputs (do not propagate to other b-values).")

    s2s_p = sub.add_parser("register-s2s", help="Register DWI stations to each other using b1000 overlap-only strategy")
    s2s_scope = s2s_p.add_mutually_exclusive_group(required=True)
    s2s_scope.add_argument("--patient-dir", type=Path, help="Single patient output folder.")
    s2s_scope.add_argument("--root-dir", type=Path, help="Root directory containing patient output folders.")
    s2s_p.add_argument("--b-value", default="1000", help="Reference DWI b-value directory name (default: 1000).")
    s2s_p.add_argument("--input-subdir", default="_F2A", help="Input subdirectory under patient folder (default: _F2A).")
    s2s_p.add_argument("--output-subdir", default="_S2S", help="Output subdirectory under patient folder (default: _S2S).")
    s2s_p.add_argument("--only-b1000", action="store_true", help="Only write b1000 outputs (do not propagate to other b-values).")

    merge_p = sub.add_parser("merge-wb", help="Generic merge: merge station folders into WB volumes")
    merge_scope = merge_p.add_mutually_exclusive_group(required=True)
    merge_scope.add_argument("--patient-dir", type=Path, help="Folder containing modality subfolders with station NIfTIs.")
    merge_scope.add_argument("--root-dir", type=Path, help="Root directory containing patient folders.")
    merge_p.add_argument("--target-subdir", default=None, help="Optional subdir to merge within each patient (e.g. _S2S).")
    merge_p.add_argument("--keep-stations", action="store_true", help="Keep station folders after merge.")

    merge_dbg_p = sub.add_parser("merge-after-registration", help="Debug merge: anatomical WB then functional WB")
    merge_dbg_scope = merge_dbg_p.add_mutually_exclusive_group(required=True)
    merge_dbg_scope.add_argument("--patient-dir", type=Path, help="Single patient output folder.")
    merge_dbg_scope.add_argument("--root-dir", type=Path, help="Root directory containing patient output folders.")
    merge_dbg_p.add_argument("--anatomical-modality", default="T1", help="Anatomical station folder (default: T1).")
    merge_dbg_p.add_argument("--functional-subdir", default="_S2S", help="Functional registration subdir (default: _S2S).")
    merge_dbg_p.add_argument("--keep-stations", action="store_true", help="Keep station folders after merge.")

    return parser.parse_args()


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    runner = PipelineRunner(
        cfg=cfg,
        interactive=args.interactive,
        array_index=args.array_index,
        array_size=args.array_size,
    )
    runner.run()


def cmd_scan_sequences(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    sorter = DicomSorter(cfg, interactive=False)
    matches = sorter.sort_and_convert(args.patient_dir, cfg.working_dir / "scan_only")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump([str(p) for p in matches], handle, indent=2)

def cmd_convert_dicom(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg.input_dir = args.input_dir
    cfg.output_dir = args.output_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    sorter = DicomSorter(cfg, interactive=args.interactive)
    patients = sorted([p for p in cfg.input_dir.iterdir() if p.is_dir()])
    for patient in patients:
        patient_out = cfg.output_dir / patient.name
        sorter.sort_and_convert(patient, patient_out)

def cmd_reconstruct_anatomical(args: argparse.Namespace) -> None:
    if getattr(args, "patient_dir", None) is not None:
        out = reconstruct_anatomical(args.patient_dir, modality=args.modality, keep_station_dir=args.keep_stations)
        if out is None:
            raise SystemExit(f"No {args.modality}/ stations found under {args.patient_dir}")
        print(f"[reconstruct-anatomical] Wrote {out}")
        return
    reconstruct_anatomical_for_root(args.root_dir, modality=args.modality, keep_station_dir=args.keep_stations)


def cmd_register_f2a(args: argparse.Namespace) -> None:
    if getattr(args, "patient_dir", None) is not None:
        out = register_functional_to_anatomical(
            args.patient_dir,
            b_value=args.b_value,
            anatomical_wb=args.anatomical_wb,
            output_dir=args.patient_dir / args.output_subdir,
            apply_to_all_bvalues=not args.only_b1000,
        )
        print(f"[register-f2a] Wrote {out}")
        return
    register_functional_to_anatomical_for_root(
        args.root_dir,
        b_value=args.b_value,
        output_subdir=args.output_subdir,
        apply_to_all_bvalues=not args.only_b1000,
    )


def cmd_register_s2s(args: argparse.Namespace) -> None:
    if getattr(args, "patient_dir", None) is not None:
        out = register_station_to_station(
            args.patient_dir,
            b_value=args.b_value,
            input_dir=args.patient_dir / args.input_subdir,
            output_dir=args.patient_dir / args.output_subdir,
            apply_to_all_bvalues=not args.only_b1000,
        )
        print(f"[register-s2s] Wrote {out}")
        return
    register_station_to_station_for_root(
        args.root_dir,
        b_value=args.b_value,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
        apply_to_all_bvalues=not args.only_b1000,
    )

def cmd_merge_wb(args: argparse.Namespace) -> None:
    def _merge_one(root: Path) -> None:
        target = root / args.target_subdir if args.target_subdir else root
        written = merge_station_folders(
            target,
            output_dir=target,
            remove_station_folders=not args.keep_stations,
        )
        print(f"[merge-wb] {target}: wrote {len(written)} volumes")

    if getattr(args, "patient_dir", None) is not None:
        _merge_one(args.patient_dir)
        return

    patients = sorted([p for p in args.root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        _merge_one(patient)


def cmd_merge_after_registration(args: argparse.Namespace) -> None:
    if getattr(args, "patient_dir", None) is not None:
        written = merge_wb_after_registration(
            args.patient_dir,
            anatomical_modality=args.anatomical_modality,
            functional_subdir=args.functional_subdir,
            keep_station_folders=args.keep_stations,
        )
        print(f"[merge-after-registration] Wrote {len(written)} WB functional volumes")
        return
    merge_wb_after_registration_for_root(
        args.root_dir,
        anatomical_modality=args.anatomical_modality,
        functional_subdir=args.functional_subdir,
        keep_station_folders=args.keep_stations,
    )


def main() -> None:
    args = parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "scan-sequences":
        cmd_scan_sequences(args)
    elif args.command == "convert-dicom":
        cmd_convert_dicom(args)
    elif args.command == "reconstruct-anatomical":
        cmd_reconstruct_anatomical(args)
    elif args.command == "register-f2a":
        cmd_register_f2a(args)
    elif args.command == "register-s2s":
        cmd_register_s2s(args)
    elif args.command == "merge-wb":
        cmd_merge_wb(args)
    elif args.command == "merge-after-registration":
        cmd_merge_after_registration(args)


if __name__ == "__main__":
    main()
