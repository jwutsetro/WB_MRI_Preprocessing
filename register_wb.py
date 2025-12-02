#!/usr/bin/env python3
"""
Register whole-body diffusion outputs (ADC and dwi) to anatomical T1 for all patients.
Usage:
    python register_wb.py /path/to/output_root
"""

import argparse
from pathlib import Path

from Preprocessing.registration import register_wholebody_dwi_to_anatomical


def main() -> None:
    parser = argparse.ArgumentParser(description="Register WB ADC/dwi to T1 for all patients")
    parser.add_argument("root_dir", type=Path, help="Root directory containing patient folders")
    args = parser.parse_args()
    patients = sorted([p for p in args.root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        print(f"[RegisterWB] Processing {patient.name}")
        register_wholebody_dwi_to_anatomical(patient)


if __name__ == "__main__":
    main()
