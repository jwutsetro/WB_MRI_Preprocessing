from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import SimpleITK as sitk

from Preprocessing.config import EddyConfig
from Preprocessing.eddy import run_eddy_for_patient


def test_eddy_skips_cleanly_when_binary_missing(tmp_path: Path, monkeypatch) -> None:
    patient_out = tmp_path / "patient"
    raw_dir = patient_out / "_dwi_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Minimal 4D NIfTI + bval/bvec sidecars.
    base = np.zeros((6, 6, 6), dtype=np.float32)
    base[2:4, 2:4, 2:4] = 100.0
    v0 = sitk.GetImageFromArray(base)
    v1 = sitk.GetImageFromArray(base * 0.8)
    v1.CopyInformation(v0)
    img4 = sitk.JoinSeries([v0, v1])
    img4.SetSpacing((2.0, 2.0, 2.0, 1.0))

    sitk.WriteImage(img4, str(raw_dir / "1.nii.gz"), True)
    (raw_dir / "1.bval").write_text("0 1000\n", encoding="utf-8")
    (raw_dir / "1.bvec").write_text("0 0\n0 0\n0 0\n", encoding="utf-8")

    # Force "no eddy binary" regardless of host environment.
    monkeypatch.setattr("Preprocessing.eddy._which_any", lambda *args, **kwargs: None)

    cfg = EddyConfig(enable=True, raw_dwi_dir_name="_dwi_raw")
    run_eddy_for_patient(patient_out, cfg, target_orientation="LPS")

    assert (raw_dir / "1.nii.gz").exists()


def test_eddy_skips_when_bvecs_degenerate_and_writes_metadata(tmp_path: Path, monkeypatch) -> None:
    patient_out = tmp_path / "patient"
    raw_dir = patient_out / "_dwi_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    base = np.zeros((6, 6, 6), dtype=np.float32)
    base[2:4, 2:4, 2:4] = 100.0
    v0 = sitk.GetImageFromArray(base)
    v1 = sitk.GetImageFromArray(base * 0.8)
    v1.CopyInformation(v0)
    img4 = sitk.JoinSeries([v0, v1, v1, v1])
    img4.SetSpacing((2.0, 2.0, 2.0, 1.0))

    sitk.WriteImage(img4, str(raw_dir / "1.nii.gz"), True)
    (raw_dir / "1.bval").write_text("0 1000 1000 1000\n", encoding="utf-8")
    # degenerate: all zeros
    (raw_dir / "1.bvec").write_text("0 0 0 0\n0 0 0 0\n0 0 0 0\n", encoding="utf-8")

    # Pretend eddy is installed; ensure we don't call it for degenerate bvecs.
    monkeypatch.setattr("Preprocessing.eddy._which_any", lambda *args, **kwargs: "/usr/local/bin/eddy")

    def _fail_run(*args, **kwargs):
        raise AssertionError("eddy should not be invoked for degenerate bvecs")

    monkeypatch.setattr("Preprocessing.eddy.subprocess.run", _fail_run)

    cfg = EddyConfig(enable=True, raw_dwi_dir_name="_dwi_raw")
    run_eddy_for_patient(patient_out, cfg, target_orientation="LPS")

    meta_path = patient_out / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["eddy"]["stations"]["1"]["status"] == "skipped"
    assert meta["eddy"]["stations"]["1"]["reason"] in ("bvec_all_zero", "bvec_not_varied")
