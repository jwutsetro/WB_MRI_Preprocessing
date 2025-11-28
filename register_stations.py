#!/usr/bin/env python3
"""
Run inter-station registration for all patients (ADC-driven, applied to all b-values).
Usage:
    python register_stations.py /path/to/output_root
"""

import argparse
from pathlib import Path

from Preprocessing.registration import register_patient


def main() -> None:
    parser = argparse.ArgumentParser(description="Register stations (ADC-driven) for all patients")
    parser.add_argument("root_dir", type=Path, help="Root directory containing patient folders")
    args = parser.parse_args()
    patients = sorted([p for p in args.root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        print(f"[Register] Processing {patient.name}")
        register_patient(patient)


if __name__ == "__main__":
    main()
