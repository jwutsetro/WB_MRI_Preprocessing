from __future__ import annotations

import argparse
from pathlib import Path

from Preprocessing.config import load_config
from Preprocessing.dicom_sort import DicomSorter
from Preprocessing.nyul import nyul_apply_models, nyul_ensure_models
from Preprocessing.alignment import (
    ALIGNMENT_STEP_ORDER,
    AlignmentRunner,
    pipeline_apply_cli_step_overrides,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WB MRI preprocessing")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the WB alignment pipeline")
    run.add_argument("--config", required=True, type=Path, help="Path to pipeline config YAML")
    run.add_argument("--interactive", action="store_true", help="Prompt for unknown sequences")
    run.add_argument("--array-index", type=int, default=None, help="Index for job array shard")
    run.add_argument("--array-size", type=int, default=None, help="Total size for job array shard")
    run.add_argument("--patient-dir", type=Path, default=None, help="Run on a single patient directory (DICOMs)")
    run.add_argument("--patient-id", type=str, default=None, help="Override output folder name for --patient-dir")
    run.add_argument(
        "--steps",
        type=str,
        default=None,
        help=f"Comma-separated steps to run (known: {','.join(ALIGNMENT_STEP_ORDER)})",
    )
    run.add_argument(
        "--from-step",
        type=str,
        default=None,
        help=f"Start step (known: {','.join(ALIGNMENT_STEP_ORDER)})",
    )
    run.add_argument(
        "--to-step",
        type=str,
        default=None,
        help=f"End step (known: {','.join(ALIGNMENT_STEP_ORDER)})",
    )

    scan = sub.add_parser("scan-sequences", help="Scan a patient folder and report unknown sequences")
    scan.add_argument("--config", required=True, type=Path, help="Path to pipeline config YAML")
    scan.add_argument("--patient-dir", required=True, type=Path, help="Patient directory containing DICOMs")

    nyul = sub.add_parser("nyul", help="Fit/apply Nyul histogram standardisation models (dataset-level)")
    nyul.add_argument("--config", required=True, type=Path, help="Path to pipeline config YAML")
    nyul.add_argument(
        "--mode",
        choices=("fit", "apply", "fit-apply"),
        default="fit-apply",
        help="Fit models, apply models, or both (default: fit-apply)",
    )

    return parser.parse_args()


def _cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    only_steps = None
    if args.steps:
        only_steps = [s for s in args.steps.split(",")]
    pipeline_apply_cli_step_overrides(cfg, only=only_steps, from_step=args.from_step, to_step=args.to_step)
    runner = AlignmentRunner(
        cfg=cfg,
        interactive=args.interactive,
        array_index=args.array_index,
        array_size=args.array_size,
        patient_dir=args.patient_dir,
        patient_id=args.patient_id,
    )
    runner.run()


def _cmd_scan_sequences(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    sorter = DicomSorter(cfg, interactive=False)
    unknown = sorter.scan_unknown_sequences(args.patient_dir)
    if not unknown:
        print("[OK] No unknown sequences detected.")
        return
    print(f"[WARN] Unknown sequences: {len(unknown)}")
    for item in unknown[:20]:
        print(f"- {item.get('series_description','')} (ProtocolName={item.get('protocol_name','')})")
    if len(unknown) > 20:
        print(f"... ({len(unknown) - 20} more)")


def _cmd_nyul(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if not cfg.nyul.enable:
        return
    if args.mode in ("fit", "fit-apply"):
        nyul_ensure_models(cfg.output_dir, cfg.nyul)
    if args.mode in ("apply", "fit-apply"):
        nyul_apply_models(cfg.output_dir, cfg.nyul)


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        _cmd_run(args)
    elif args.command == "scan-sequences":
        _cmd_scan_sequences(args)
    elif args.command == "nyul":
        _cmd_nyul(args)


if __name__ == "__main__":
    main()
