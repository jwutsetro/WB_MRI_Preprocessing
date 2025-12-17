from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from Preprocessing.merge_wb import merge_station_folders


def _write_station(path: Path, *, origin_z: float, value: float) -> None:
    img = sitk.Image(5, 5, 5, sitk.sitkFloat32)
    img = img + float(value)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, float(origin_z)))
    sitk.WriteImage(img, str(path), True)


def test_merge_station_folders_writes_and_removes(tmp_path: Path) -> None:
    root = tmp_path / "_S2S"
    b1000 = root / "1000"
    b1000.mkdir(parents=True)
    _write_station(b1000 / "1.nii.gz", origin_z=0.0, value=1.0)
    _write_station(b1000 / "2.nii.gz", origin_z=4.0, value=2.0)

    written = merge_station_folders(root, output_dir=root, remove_station_folders=True)
    assert (root / "1000.nii.gz") in written
    assert (root / "1000.nii.gz").exists()
    assert not b1000.exists()


def test_merge_station_folders_keep_stations(tmp_path: Path) -> None:
    root = tmp_path / "_S2S"
    b1000 = root / "1000"
    b1000.mkdir(parents=True)
    _write_station(b1000 / "1.nii.gz", origin_z=0.0, value=1.0)
    _write_station(b1000 / "2.nii.gz", origin_z=4.0, value=2.0)

    written = merge_station_folders(root, output_dir=root, remove_station_folders=False)
    assert (root / "1000.nii.gz") in written
    assert b1000.exists()

