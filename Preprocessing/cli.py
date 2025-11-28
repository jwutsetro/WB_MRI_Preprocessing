from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from Preprocessing.config import load_config
from Preprocessing.dicom_sort import DicomSorter
from Preprocessing.pipeline import PipelineRunner


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


def main() -> None:
    args = parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "scan-sequences":
        cmd_scan_sequences(args)
    elif args.command == "convert-dicom":
        cmd_convert_dicom(args)


if __name__ == "__main__":
    main()
