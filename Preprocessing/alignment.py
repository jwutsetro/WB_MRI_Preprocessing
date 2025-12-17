from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from Preprocessing.config import PipelineConfig
from Preprocessing.adc import compute_adc_for_patient
from Preprocessing.dicom_sort import DicomSorter
from Preprocessing.noise_bias import process_patient as run_noise_bias
from Preprocessing.isis import standardize_patient
from Preprocessing.registration import register_patient, register_wholebody_dwi_to_anatomical
from Preprocessing.merge_wb import merge_patient


ALIGNMENT_STEP_ORDER: List[str] = [
    "dicom_sort",
    "adc",
    "noise_bias",
    "registration",
    "isis",
    "reconstruct",
    "resample_to_t1",
]


def pipeline_select_steps(
    *,
    only: Sequence[str] | None = None,
    from_step: str | None = None,
    to_step: str | None = None,
) -> set[str]:
    """Return a validated set of step names to run.

    Exactly one of (only) or (from_step/to_step) may be provided.
    """
    order = list(ALIGNMENT_STEP_ORDER)
    known = set(order)

    if only and (from_step or to_step):
        raise ValueError("Use either `only` or `from_step/to_step`, not both.")

    if only:
        selected = {s.strip() for s in only if s.strip()}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"Unknown step(s): {unknown}. Known steps: {order}")
        return selected

    if from_step is None and to_step is None:
        return known

    start = from_step or order[0]
    end = to_step or order[-1]
    if start not in known:
        raise ValueError(f"Unknown from_step: {start}. Known steps: {order}")
    if end not in known:
        raise ValueError(f"Unknown to_step: {end}. Known steps: {order}")
    i0 = order.index(start)
    i1 = order.index(end)
    if i1 < i0:
        raise ValueError("to_step must be after from_step in pipeline order.")
    return set(order[i0 : i1 + 1])


def pipeline_apply_step_selection(cfg: PipelineConfig, selected_steps: set[str]) -> None:
    """Mutate cfg.steps to match the selected step set."""
    cfg.steps.dicom_sort = "dicom_sort" in selected_steps
    cfg.steps.adc = "adc" in selected_steps
    cfg.steps.noise_bias = "noise_bias" in selected_steps
    cfg.steps.registration = "registration" in selected_steps
    cfg.steps.isis = "isis" in selected_steps
    cfg.steps.reconstruct = "reconstruct" in selected_steps
    cfg.steps.resample_to_t1 = "resample_to_t1" in selected_steps


def chunk_by_array_index(items: Sequence[Path], array_index: int | None, array_size: int | None) -> List[Path]:
    if array_index is None or array_size is None:
        return list(items)
    if array_index < 0 or array_index >= array_size:
        raise ValueError("array_index must be within [0, array_size).")
    return [item for idx, item in enumerate(items) if idx % array_size == array_index]


def resample_to_reference(image: sitk.Image, reference: sitk.Image, interpolator=sitk.sitkLinear) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    resample.SetInterpolator(interpolator)
    return resample.Execute(image)


class AlignmentRunner:
    def __init__(
        self,
        cfg: PipelineConfig,
        interactive: bool = False,
        array_index: int | None = None,
        array_size: int | None = None,
        patient_dir: Path | None = None,
        patient_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.interactive = interactive
        self.array_index = array_index
        self.array_size = array_size
        self.patient_dir = patient_dir
        self.patient_id = patient_id

    def run(self) -> None:
        if self.patient_dir is not None:
            patient_output = self.cfg.output_dir / (self.patient_id or self.patient_dir.name)
            patient_output.mkdir(parents=True, exist_ok=True)
            self._run_patient(self.patient_dir, patient_output)
            return

        patients = sorted([p for p in self.cfg.input_dir.iterdir() if p.is_dir()])
        patients = chunk_by_array_index(patients, self.array_index, self.array_size)
        for patient in tqdm(patients, desc="patients"):
            patient_output = self.cfg.output_dir / patient.name
            patient_output.mkdir(parents=True, exist_ok=True)
            self._run_patient(patient, patient_output)

    def _run_patient(self, patient_input: Path, patient_output: Path) -> None:
        written: List[Path] = []
        if self.cfg.steps.dicom_sort:
            sorter = DicomSorter(self.cfg, interactive=self.interactive)
            written = sorter.sort_and_convert(patient_input, patient_output)
        if self.cfg.steps.adc:
            self._run_adc(patient_output)
        if self.cfg.steps.noise_bias:
            self._run_bias(patient_output)
        if self.cfg.steps.registration:
            self._run_registration(patient_output)
        if self.cfg.steps.isis:
            self._run_isis(patient_output)
        if self.cfg.steps.reconstruct:
            self._run_reconstruct(patient_output)
        if self.cfg.steps.resample_to_t1:
            self._run_resample_to_t1(patient_output)

    def _run_adc(self, patient_output: Path) -> None:
        compute_adc_for_patient(patient_output)

    def _run_bias(self, patient_output: Path) -> None:
        run_noise_bias(patient_output)

    def _run_isis(self, patient_output: Path) -> None:
        standardize_patient(patient_output)

    def _run_registration(self, patient_output: Path) -> None:
        register_patient(patient_output)

    def _run_reconstruct(self, patient_output: Path) -> None:
        merge_patient(patient_output)

    def _run_resample_to_t1(self, patient_output: Path) -> None:
        register_wholebody_dwi_to_anatomical(patient_output)
