from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional


def prune_dwi_directories(patient_dir: Path) -> Optional[float]:
    """Keep only the highest b-value directory and rename it to dwi; remove others."""
    b_dirs = [p for p in patient_dir.iterdir() if p.is_dir() and _is_float(p.name)]
    if not b_dirs:
        return None
    best_dir = max(b_dirs, key=lambda p: float(p.name))
    best_b = float(best_dir.name)
    for d in b_dirs:
        if d == best_dir:
            continue
        shutil.rmtree(d, ignore_errors=True)
    target = patient_dir / "dwi"
    if best_dir != target:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        best_dir.rename(target)
    _update_metadata_bvalues(patient_dir, best_b, [float(p.name) for p in b_dirs])
    return best_b


def update_selected_modalities(patient_dir: Path) -> None:
    """Record selected modalities (anatomical, dwi, adc) in metadata after merges."""
    meta_path = patient_dir / "metadata.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    selections = {
        "anatomical": "T1" if (patient_dir / "T1.nii.gz").exists() else None,
        "dwi": "dwi" if (patient_dir / "dwi.nii.gz").exists() else None,
        "adc": "ADC" if (patient_dir / "ADC.nii.gz").exists() else None,
    }
    meta["selected_modalities"] = selections
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _update_metadata_bvalues(patient_dir: Path, selected: float, available: List[float]) -> None:
    meta_path = patient_dir / "metadata.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    meta.setdefault("dwi", {})
    meta["dwi"]["selected_b_value"] = selected
    meta["dwi"]["available_b_values"] = sorted(available)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False
