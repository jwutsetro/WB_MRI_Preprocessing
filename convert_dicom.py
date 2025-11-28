#!/usr/bin/env python3
"""
Convenience wrapper to run the DICOM-to-NIfTI conversion step.
Usage:
    python convert_dicom.py <input_dir> <output_dir> [--config path/to/config.yaml] [--interactive]
"""

import argparse
from pathlib import Path

from Preprocessing.cli import cmd_convert_dicom, parse_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DICOMs to NIfTI using pipeline sequence rules")
    parser.add_argument("input_dir", type=Path, help="Directory containing patient DICOM folders")
    parser.add_argument("output_dir", type=Path, help="Directory to write NIfTI outputs")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.example.yaml"), help="Pipeline config with sequence rules")
    parser.add_argument("--interactive", action="store_true", help="Prompt for unknown sequences")
    args = parser.parse_args()
    # Reuse the CLI handler
    namespace = argparse.Namespace(
        command="convert-dicom",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=args.config,
        interactive=args.interactive,
    )
    cmd_convert_dicom(namespace)


if __name__ == "__main__":
    main()

