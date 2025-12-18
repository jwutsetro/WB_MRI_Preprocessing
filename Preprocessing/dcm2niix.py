from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional


@dataclass(frozen=True)
class Dcm2niixResult:
    nifti_path: Path
    json_path: Optional[Path] = None
    bval_path: Optional[Path] = None
    bvec_path: Optional[Path] = None


def _require_dcm2niix() -> str:
    exe = shutil.which("dcm2niix")
    if not exe:
        raise RuntimeError(
            "dcm2niix not found on PATH. Install it (e.g., from https://github.com/rordenlab/dcm2niix) "
            "or set `dicom_converter: sitk` in your pipeline config."
        )
    return exe


def run_dcm2niix(*, input_dir: Path, output_dir: Path, filename: str = "converted") -> Dcm2niixResult:
    """Run dcm2niix on a directory and return the primary outputs.

    Notes:
    - dcm2niix is a command-line tool; there is no official in-process Python API.
    - This wrapper is intentionally minimal: it enforces deterministic filenames and
      discovers sidecars next to the generated NIfTI.
    """
    exe = _require_dcm2niix()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "-z",
        "y",  # gzip
        "-b",
        "y",  # BIDS sidecars (json/bval/bvec where applicable)
        "-f",
        filename,
        "-o",
        str(output_dir),
        str(input_dir),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    candidates = sorted(output_dir.glob(f"{filename}.nii*"))
    if not candidates:
        candidates = sorted(output_dir.glob("*.nii*"))
    if not candidates:
        raise RuntimeError(f"dcm2niix produced no NIfTI output in {output_dir}")

    nifti_path = candidates[0]
    stem = nifti_path.name
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    json_path = nifti_path.with_name(f"{stem}.json")
    bval_path = nifti_path.with_name(f"{stem}.bval")
    bvec_path = nifti_path.with_name(f"{stem}.bvec")

    return Dcm2niixResult(
        nifti_path=nifti_path,
        json_path=json_path if json_path.exists() else None,
        bval_path=bval_path if bval_path.exists() else None,
        bvec_path=bvec_path if bvec_path.exists() else None,
    )


def load_bvals(path: Path) -> List[float]:
    """Load b-values from a FSL-style .bval file."""
    raw = path.read_text(encoding="utf-8").strip().split()
    return [float(x) for x in raw if x.strip()]


def load_bvecs(path: Path) -> List[List[float]]:
    """Load b-vectors from a FSL-style .bvec file (3 rows)."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [[float(x) for x in ln.split()] for ln in lines]
    if len(rows) != 3:
        raise ValueError("Expected 3-row .bvec file.")
    return rows
