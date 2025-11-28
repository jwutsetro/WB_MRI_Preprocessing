#!/usr/bin/env python3
"""
Compute ADC images for all patients in a root directory.
Usage:
    python compute_adc.py /path/to/processed_root
"""

import argparse
from pathlib import Path

from Preprocessing.adc import compute_adc_for_patient


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ADC images for all patients in a root directory")
    parser.add_argument("root_dir", type=Path, help="Root directory containing patient folders with b-value subfolders")
    args = parser.parse_args()
    patients = sorted([p for p in args.root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        print(f"[ADC] Processing {patient.name}")
        written = compute_adc_for_patient(patient)
        if written:
            print(f"[ADC] Wrote {written}")
        else:
            print(f"[ADC] Skipped {patient.name} (insufficient b-values)")


if __name__ == "__main__":
    main()

