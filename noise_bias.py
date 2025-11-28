#!/usr/bin/env python3
"""
Run noise/bias correction for all patients in a root directory.
Usage:
    python noise_bias.py /path/to/output_root
"""

import argparse
from pathlib import Path

from Preprocessing.noise_bias import process_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply noise/bias correction to all patients under root_dir")
    parser.add_argument("root_dir", type=Path, help="Root directory containing patient folders")
    args = parser.parse_args()
    process_root(args.root_dir)


if __name__ == "__main__":
    main()

