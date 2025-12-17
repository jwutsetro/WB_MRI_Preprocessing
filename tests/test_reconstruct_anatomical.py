from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from Preprocessing.reconstruct_anatomical import reconstruct_anatomical


def _write_station(path: Path, *, origin_z: float) -> None:
    img = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, float(origin_z)))
    sitk.WriteImage(img, str(path), True)


def test_reconstruct_anatomical_writes_output_and_removes_dir(tmp_path: Path) -> None:
    patient = tmp_path / "P001"
    t1 = patient / "T1"
    t1.mkdir(parents=True)
    _write_station(t1 / "1.nii.gz", origin_z=0.0)
    _write_station(t1 / "2.nii.gz", origin_z=8.0)

    out = reconstruct_anatomical(patient, modality="T1", keep_station_dir=False)
    assert out == patient / "T1.nii.gz"
    assert out.exists()
    assert not t1.exists()

    merged = sitk.ReadImage(str(out))
    assert merged.GetSize()[2] == 18


def test_reconstruct_anatomical_keep_station_dir(tmp_path: Path) -> None:
    patient = tmp_path / "P001"
    t1 = patient / "T1"
    t1.mkdir(parents=True)
    _write_station(t1 / "1.nii.gz", origin_z=0.0)
    _write_station(t1 / "2.nii.gz", origin_z=8.0)

    out = reconstruct_anatomical(patient, modality="T1", keep_station_dir=True)
    assert out == patient / "T1.nii.gz"
    assert out.exists()
    assert t1.exists()

