from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk


def _linear_fit_adc(b_values: List[float], log_signals: np.ndarray) -> np.ndarray:
    """Compute ADC via linear fit of log(S) vs b. Returns ADC array (mm^2/s scaled by 1e-3)."""
    b = np.asarray(b_values, dtype=np.float32)
    n = float(b.size)
    sum_b = b.sum()
    sum_b2 = float((b * b).sum())
    sum_y = log_signals.sum(axis=0)
    sum_by = (b[:, None, None, None] * log_signals).sum(axis=0)
    denom = n * sum_b2 - sum_b * sum_b
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    slope = (n * sum_by - sum_b * sum_y) / denom
    adc = -slope  # negative slope of ln(S) vs b
    adc = np.clip(adc, 0, None)
    return adc.astype(np.float32)


BACKGROUND_INTENSITY_THRESHOLD = 0.005  # in original signal units
ADC_SCALE = 1000.0  # scale factor to report in mm^2/s * 1e-3


def compute_adc_image(b_images: List[sitk.Image], b_values: List[float]) -> sitk.Image:
    """Compute ADC image from list of images and their corresponding b-values."""
    if len(b_images) != len(b_values):
        raise ValueError("b_images and b_values length mismatch")
    pairs = sorted(zip(b_values, b_images), key=lambda x: x[0])
    b_vals_sorted, imgs_sorted = zip(*pairs)
    # Select b-values: prefer all >0; if only one >0 and b0 exists, use both
    positives = [(b, im) for b, im in pairs if b > 0]
    if len(positives) >= 2:
        use_pairs = positives
    else:
        use_pairs = pairs
    use_b = [b for b, _ in use_pairs]
    arrays = []
    for _, im in use_pairs:
        arr = sitk.GetArrayFromImage(im).astype(np.float32)
        arr = np.maximum(arr, 1e-6)
        arrays.append(np.log(arr))
    log_stack = np.stack(arrays, axis=0)
    adc_array = _linear_fit_adc(use_b, log_stack)
    # suppress background where mean signal is low
    mean_signal = np.mean(np.exp(log_stack), axis=0)
    adc_array = np.where(mean_signal >= BACKGROUND_INTENSITY_THRESHOLD, adc_array, 0.0)
    adc_array = adc_array * ADC_SCALE
    adc_image = sitk.GetImageFromArray(adc_array)
    adc_image.CopyInformation(imgs_sorted[0])
    return adc_image


def compute_adc_for_patient(patient_dir: Path) -> Path | None:
    """Create ADC image for a patient directory; returns ADC file path or None."""
    b_dirs = [p for p in patient_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not b_dirs:
        return None
    station_map: Dict[str, List[Tuple[float, Path]]] = {}
    for b_dir in b_dirs:
        try:
            b_val = float(b_dir.name)
        except ValueError:
            continue
        for file in sorted(b_dir.glob("*.nii*")):
            station_map.setdefault(file.stem, []).append((b_val, file))
    adc_dir = patient_dir / "ADC"
    adc_dir.mkdir(parents=True, exist_ok=True)
    last_written: Path | None = None
    for station, lst in station_map.items():
        if len(lst) < 2:
            continue
        lst.sort(key=lambda x: x[0])
        b_vals = [b for b, _ in lst]
        b_images = [sitk.ReadImage(str(path)) for _, path in lst]
        adc_image = compute_adc_image(b_images, b_vals)
        out_path = adc_dir / f"{station}.nii.gz"
        sitk.WriteImage(adc_image, str(out_path), True)
        last_written = out_path
    return last_written
