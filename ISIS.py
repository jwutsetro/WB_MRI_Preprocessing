#!/usr/bin/env python3
"""
Run inter-station intensity standardisation (ISIS) for all patients in a root directory.
Usage:
    python ISIS.py /path/to/output_root
"""

import argparse
from pathlib import Path

from Preprocessing.isis import standardize_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ISIS across all patients under root_dir")
    parser.add_argument("root_dir", type=Path, help="Root directory containing patient folders")
    parser.add_argument("--skip", nargs="*", default=["ADC"], help="Modalities to skip (default: ADC)")
    args = parser.parse_args()
    standardize_root(args.root_dir, skip_modalities=args.skip)


if __name__ == "__main__":
    main()

