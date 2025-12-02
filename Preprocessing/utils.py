from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


ANATOMICAL_PRIORITY: List[str] = ["T1", "3D_T1_TSE", "DixonIP", "Dixon_IP", "Dixon", "T2"]


def prune_anatomical_modalities(patient_dir: Path, priority: Optional[List[str]] = None) -> None:
    """Keep a single anatomical modality and rename it to T1; remove other anatomical folders."""
    meta = _load_metadata(patient_dir)
    priority_list = priority or ANATOMICAL_PRIORITY
    available_dirs = {p.name: p for p in patient_dir.iterdir() if p.is_dir()}
    selected_name: Optional[str] = None
    for name in priority_list:
        if name in available_dirs:
            selected_name = name
            break
    if not selected_name:
        return
    selected_dir = available_dirs[selected_name]
    target = patient_dir / "T1"
    if selected_dir != target:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        selected_dir.rename(target)
    skipped: List[str] = []
    for name, path in available_dirs.items():
        if name == selected_name or name == "ADC" or _is_bvalue(name):
            continue
        if name in priority_list or (meta and name in meta.get("anatomical_modalities", [])):
            skipped.append(name)
            shutil.rmtree(path, ignore_errors=True)
    if meta:
        meta["anatomical_modalities"] = ["T1"]
        _rewrite_modalities(meta, selected_name, target_name="T1")
        if skipped:
            meta["skipped_modalities"] = sorted(skipped)
        _save_metadata(patient_dir, meta)


def prune_dwi_directories(patient_dir: Path) -> None:
    """Keep only the highest b-value DWI directory, rename to dwi, remove others, and update metadata."""
    b_dirs = [p for p in patient_dir.iterdir() if p.is_dir() and _is_bvalue(p.name)]
    if not b_dirs:
        return
    best_dir = max(b_dirs, key=lambda p: float(p.name))
    best_b = float(best_dir.name)
    for d in b_dirs:
        if d != best_dir:
            shutil.rmtree(d, ignore_errors=True)
    target = patient_dir / "dwi"
    if best_dir != target:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        best_dir.rename(target)
    meta = _load_metadata(patient_dir)
    if meta:
        meta.setdefault("dwi", {})
        meta["dwi"]["selected_b_value"] = best_b
        meta["dwi"]["available_b_values"] = sorted(float(p.name) for p in b_dirs)
        _rewrite_dwi_modalities(meta, original_name=str(best_b))
        _save_metadata(patient_dir, meta)


def _rewrite_modalities(meta: Dict, source_name: str, target_name: str) -> None:
    modalities = meta.get("modalities", {})
    if source_name in modalities:
        entries = modalities.pop(source_name)
        for entry in entries:
            entry["file"] = str(Path(target_name) / Path(entry["file"]).name)
        modalities[target_name] = entries
    meta["modalities"] = modalities


def _rewrite_dwi_modalities(meta: Dict, original_name: str) -> None:
    modalities = meta.get("modalities", {})
    if original_name not in modalities:
        return
    entries = modalities.pop(original_name)
    for entry in entries:
        entry["file"] = str(Path("dwi") / Path(entry["file"]).name)
        entry["b_value"] = float(original_name)
    modalities["dwi"] = entries
    meta["modalities"] = modalities


def _load_metadata(patient_dir: Path) -> Optional[Dict]:
    path = patient_dir / "metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_metadata(patient_dir: Path, meta: Dict) -> None:
    path = patient_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _is_bvalue(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False
