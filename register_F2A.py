#!/usr/bin/env python3
"""
Register DWI stations to the whole-body anatomical image (functional-to-anatomical, F2A).

Uses b1000 as the reference sequence for estimating transforms and applies the same
transform to other DWI b-values when present.

Usage:
    python register_F2A.py /path/to/output_root
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Preprocessing.register_F2A import register_functional_to_anatomical_for_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Register DWI stations to anatomical WB (F2A) for all patients")
    parser.add_argument("root_dir", type=Path, help="Root directory containing patient folders")
    parser.add_argument("--b-value", default="1000", help="Reference DWI b-value directory name (default: 1000).")
    parser.add_argument("--output-subdir", default="_F2A", help="Output subdirectory under patient folder (default: _F2A).")
    parser.add_argument("--only-b1000", action="store_true", help="Only write b1000 outputs (do not propagate to other b-values).")
    args = parser.parse_args()
    register_functional_to_anatomical_for_root(
        args.root_dir,
        b_value=args.b_value,
        output_subdir=args.output_subdir,
        apply_to_all_bvalues=not args.only_b1000,
    )


if __name__ == "__main__":
    main()

