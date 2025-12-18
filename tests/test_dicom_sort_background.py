from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from Preprocessing.config import NyulConfig, PipelineConfig, SequenceRule, StepConfig
from Preprocessing.dicom_sort import DicomSorter


def test_dicom_sort_does_not_apply_background_threshold(tmp_path: Path) -> None:
    rule = SequenceRule(
        name="t1",
        description_contains=["t1"],
        output_modality="T1",
        canonical_modality="T1",
        is_anatomical=True,
        background_threshold=5.0,
        mask_threshold=50.0,
    )
    cfg = PipelineConfig(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        sequence_rules=[rule],
        steps=StepConfig(),
        nyul=NyulConfig(enable=False),
    )
    sorter = DicomSorter(cfg, interactive=False)

    arr = np.zeros((4, 4, 4), dtype=np.float32)
    arr[1, 1, 1] = 3.0  # below background_threshold
    arr[2, 2, 2] = 7.0  # above background_threshold
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 2.0))

    out_dir = tmp_path / "patient"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path, _ = sorter._write_series(  # type: ignore[attr-defined]
        rule=rule,
        b_value=None,
        instances=[],
        patient_output=out_dir,
        station_idx=1,
        image=img,
    )
    out_img = sitk.ReadImage(str(out_path))
    out_arr = sitk.GetArrayFromImage(out_img).astype(np.float32)

    assert float(out_arr[1, 1, 1]) == 3.0
    assert float(out_arr[2, 2, 2]) == 7.0

