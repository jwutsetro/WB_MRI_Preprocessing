from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


def apply_bias_correction(image: sitk.Image) -> sitk.Image:
    """Apply anisotropic diffusion followed by N4 bias correction."""
    image = sitk.Cast(image, sitk.sitkFloat32)
    diffusion = sitk.GradientAnisotropicDiffusionImageFilter()
    diffusion.SetConductanceParameter(4)
    diffusion.SetNumberOfIterations(10)
    diffusion.SetTimeStep(0.01)
    smoothed = diffusion.Execute(image)
    arr = sitk.GetArrayFromImage(smoothed)
    mask = sitk.BinaryThreshold(smoothed, lowerThreshold=5, upperThreshold=float(np.max(arr)), insideValue=1, outsideValue=0)
    bias = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias.Execute(smoothed, sitk.Cast(mask, sitk.sitkUInt8))
    return sitk.Cast(corrected, sitk.sitkFloat32)


def process_patient(patient_dir: Path) -> None:
    """Run noise/bias correction on all NIfTI files under a patient directory."""
    image_paths = sorted(patient_dir.rglob("*.nii*"))
    for image_path in image_paths:
        image = sitk.ReadImage(str(image_path))
        corrected = apply_bias_correction(image)
        sitk.WriteImage(corrected, str(image_path), True)


def process_root(root_dir: Path) -> None:
    """Run noise/bias correction for all patients in root_dir."""
    patients = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    for patient in patients:
        print(f"[Noise/Bias] Processing {patient.name}")
        process_patient(patient)

