#!/usr/bin/env python3
"""
Merge per-station volumes into whole-body outputs with feathered overlaps.
Usage:
    python merge_WB.py /path/to/output_root
"""

import argparse
from pathlib import Path

from Preprocessing.merge_wb import merge_patient


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge station volumes into whole-body outputs for all patients")
    parser.add_argument("root_dir", type=Path, help="Root directory containing patient folders")
    args = parser.parse_args()
    patients = sorted([p for p in args.root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        print(f"[MergeWB] Processing {patient.name}")
        merge_patient(patient)


if __name__ == "__main__":
    main()
