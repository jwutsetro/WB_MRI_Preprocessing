#!/usr/bin/env python3
"""
Reconstruct the whole-body anatomical image (default: T1) from per-station volumes.

Usage:
    python reconstruct_anatomical.py /path/to/output_root
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Preprocessing.reconstruct_anatomical import reconstruct_anatomical_for_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct anatomical WB volume from stations for all patients")
    parser.add_argument("root_dir", type=Path, help="Root directory containing patient folders")
    parser.add_argument("--modality", default="T1", help="Anatomical modality folder to merge (default: T1).")
    parser.add_argument("--keep-stations", action="store_true", help="Keep the station folder after writing the WB volume.")
    args = parser.parse_args()
    reconstruct_anatomical_for_root(args.root_dir, modality=args.modality, keep_station_dir=args.keep_stations)


if __name__ == "__main__":
    main()

