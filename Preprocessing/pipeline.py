from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from Preprocessing.config import PipelineConfig
from Preprocessing.adc import compute_adc_for_patient
from Preprocessing.dicom_sort import DicomSorter
from Preprocessing.noise_bias import process_patient as run_noise_bias
from Preprocessing.nyul import fit_nyul_model, load_model, save_model
from Preprocessing.isis import standardize_patient
from Preprocessing.registration import register_patient, register_wholebody_adc_to_t1


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


class PipelineRunner:
    def __init__(
        self,
        cfg: PipelineConfig,
        interactive: bool = False,
        array_index: int | None = None,
        array_size: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.interactive = interactive
        self.array_index = array_index
        self.array_size = array_size

    def run(self) -> None:
        patients = sorted([p for p in self.cfg.input_dir.iterdir() if p.is_dir()])
        patients = chunk_by_array_index(patients, self.array_index, self.array_size)
        for patient in tqdm(patients, desc="patients"):
            patient_output = self.cfg.output_dir / patient.name
            patient_output.mkdir(parents=True, exist_ok=True)
            self._run_patient(patient, patient_output)
        if self.cfg.steps.nyul and self.cfg.nyul.enable:
            self._run_nyul(self.cfg.output_dir)

    def _run_patient(self, patient_input: Path, patient_output: Path) -> None:
        written: List[Path] = []
        if self.cfg.steps.dicom_sort:
            sorter = DicomSorter(self.cfg, interactive=self.interactive)
            written = sorter.sort_and_convert(patient_input, patient_output)
        if self.cfg.steps.adc:
            self._run_adc(patient_output)
        if self.cfg.steps.noise_bias:
            self._run_bias(patient_output)
        if self.cfg.steps.isis:
            self._run_isis(patient_output)
        if self.cfg.steps.registration:
            self._run_registration(patient_output)
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
        for modality_dir in patient_output.iterdir():
            if not modality_dir.is_dir():
                continue
            station_files = sorted(modality_dir.glob("*.nii*"))
            if len(station_files) <= 1:
                continue
            station_files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
            station_images = [sitk.ReadImage(str(p)) for p in station_files]
            spacing = station_images[0].GetSpacing()
            size_x, size_y = station_images[0].GetSize()[:2]
            total_z = sum(im.GetSize()[2] for im in station_images)
            output = sitk.Image(size_x, size_y, total_z, sitk.sitkFloat32)
            output.SetSpacing(spacing)
            output.SetOrigin(station_images[0].GetOrigin())
            output.SetDirection(station_images[0].GetDirection())
            current_z = 0
            paste = sitk.PasteImageFilter()
            for image in station_images:
                size = image.GetSize()
                paste.SetDestinationIndex([0, 0, current_z])
                paste.SetSourceIndex([0, 0, 0])
                paste.SetSourceSize(size)
                output = paste.Execute(output, image)
                current_z += size[2]
            sitk.WriteImage(output, str(patient_output / f"{modality_dir.name}_WB.nii.gz"), True)

    def _run_resample_to_t1(self, patient_output: Path) -> None:
        register_wholebody_adc_to_t1(patient_output)

    def _run_nyul(self, output_root: Path) -> None:
        model_dir = self.cfg.nyul.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        for modality in self.cfg.nyul.modalities:
            images: List[sitk.Image] = []
            modality_files = sorted(output_root.glob(f"*/*{modality}_WB.nii.gz"))
            for file in modality_files:
                images.append(sitk.ReadImage(str(file)))
            if not images:
                continue
            model_path = model_dir / f"{modality}_nyul.json"
            if not self.cfg.nyul.refresh and model_path.exists():
                model = load_model(model_path)
            else:
                model = fit_nyul_model(modality, images, self.cfg.nyul)
                save_model(model, model_path)
            for file in modality_files:
                img = sitk.ReadImage(str(file))
                out = model.apply(img)
                sitk.WriteImage(out, str(file), True)
