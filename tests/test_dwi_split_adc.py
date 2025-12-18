from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from Preprocessing.adc import compute_adc_for_patient
from Preprocessing.config import NyulConfig, PipelineConfig, SequenceRule, StepConfig
from Preprocessing.dicom_sort import DicomSorter


def test_dwi_split_writes_b_dirs_and_adc_runs(tmp_path: Path) -> None:
    rule = SequenceRule(
        name="dwi",
        description_contains=["dwi"],
        output_modality="DWI",
        canonical_modality="DWI",
        b_value_tag="0018|9087",
        mask_threshold=10.0,
    )
    cfg = PipelineConfig(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        sequence_rules=[rule],
        steps=StepConfig(),
        nyul=NyulConfig(enable=False),
    )
    sorter = DicomSorter(cfg, interactive=False)

    # Build a 4D image with b=0 (higher signal) and b=1000 (lower signal).
    base = np.zeros((8, 8, 8), dtype=np.float32)
    base[2:6, 2:6, 2:6] = 100.0
    b0 = sitk.GetImageFromArray(base)
    b0.SetSpacing((2.0, 2.0, 2.0))

    b1000_arr = base.copy()
    b1000_arr[2:6, 2:6, 2:6] = 50.0
    b1000 = sitk.GetImageFromArray(b1000_arr)
    b1000.CopyInformation(b0)

    img4 = sitk.JoinSeries([b0, b1000, b1000])  # repeated b=1000 volume to exercise averaging
    img4.SetSpacing((2.0, 2.0, 2.0, 1.0))

    patient_out = tmp_path / "patient"
    patient_out.mkdir(parents=True, exist_ok=True)

    sorter._write_dwi_split(rule=rule, image=img4, bvals=[0.0, 1000.0, 1000.0], patient_output=patient_out, station_idx=1)  # type: ignore[attr-defined]

    assert (patient_out / "0" / "1.nii.gz").exists()
    assert (patient_out / "1000" / "1.nii.gz").exists()

    adc_path = compute_adc_for_patient(patient_out)
    assert adc_path is not None
    assert (patient_out / "ADC" / "1.nii.gz").exists()
    # prune_dwi_directories runs inside compute_adc_for_patient -> highest b becomes /dwi
    assert (patient_out / "dwi" / "1.nii.gz").exists()

