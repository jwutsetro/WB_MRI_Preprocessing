#!/usr/bin/env python3
"""
Merge whole-body outputs after the debug registration workflow (anatomical then functional).

Usage:
    python merge_after_registration.py /path/to/output_root
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Preprocessing.merge_after_registration import merge_wb_after_registration_for_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge WB outputs after debug registration for all patients")
    parser.add_argument("root_dir", type=Path, help="Root directory containing patient folders")
    parser.add_argument("--anatomical-modality", default="T1", help="Anatomical station folder (default: T1).")
    parser.add_argument("--functional-subdir", default="_S2S", help="Functional registration subdir (default: _S2S).")
    parser.add_argument("--keep-stations", action="store_true", help="Keep station folders after merge.")
    args = parser.parse_args()
    merge_wb_after_registration_for_root(
        args.root_dir,
        anatomical_modality=args.anatomical_modality,
        functional_subdir=args.functional_subdir,
        keep_station_folders=args.keep_stations,
    )


if __name__ == "__main__":
    main()

