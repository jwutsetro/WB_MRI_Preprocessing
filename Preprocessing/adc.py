from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import math

import numpy as np
import SimpleITK as sitk
from Preprocessing.utils import prune_dwi_directories


def _largest_component(mask: sitk.Image) -> sitk.Image:
    """Keep only the largest connected component of a binary mask."""
    labeled = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(labeled, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(relabeled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
    largest.CopyInformation(mask)
    return largest


def compute_body_mask(
    image: sitk.Image,
    smoothing_sigma: float = 1.0,
    closing_radius: int = 1,
    dilation_radius: int = 1,
    *,
    threshold: float | None = None,
    padding_threshold: float = 0.0,
) -> sitk.Image:
    """Compute a body mask via fixed threshold + connected components.

    Algorithm (legacy behavior):
    1) Remove empty-padding borders (voxels with |I| <= padding_threshold).
    2) Threshold remaining voxels to form a candidate body mask.
    3) Keep the largest connected component (body).
    4) Invert mask, keep the largest connected component (outside background).
    5) Invert again (fills low-intensity holes inside the body, e.g. lungs).

    `threshold` is treated as a minimum; an adaptive floor based on the image's
    upper tail is also applied to reduce background noise leakage.
    """
    img_float = sitk.Cast(image, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img_float).astype(np.float32)

    empty_mask = sitk.Image(img_float.GetSize(), sitk.sitkUInt8)
    empty_mask.CopyInformation(img_float)
    if arr.size == 0:
        return empty_mask

    non_empty = np.abs(arr) > float(padding_threshold)
    if not non_empty.any():
        return empty_mask

    z_any = non_empty.any(axis=(1, 2))
    y_any = non_empty.any(axis=(0, 2))
    x_any = non_empty.any(axis=(0, 1))
    z_idx = np.where(z_any)[0]
    y_idx = np.where(y_any)[0]
    x_idx = np.where(x_any)[0]
    z0, z1 = int(z_idx[0]), int(z_idx[-1] + 1)
    y0, y1 = int(y_idx[0]), int(y_idx[-1] + 1)
    x0, x1 = int(x_idx[0]), int(x_idx[-1] + 1)

    roi = arr[z0:z1, y0:y1, x0:x1]
    positive = roi[roi > 0]
    adaptive_floor = 0.0
    if positive.size:
        p99 = float(np.percentile(positive, 99.0))
        adaptive_floor = max(0.0, 0.02 * p99)
        if not math.isfinite(adaptive_floor):
            adaptive_floor = 0.0
    thr = max(float(threshold or 0.0), float(adaptive_floor))

    mask_arr = np.zeros_like(arr, dtype=np.uint8)
    roi_mask = (roi > thr).astype(np.uint8)
    mask_arr[z0:z1, y0:y1, x0:x1] = roi_mask

    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(image)
    mask = _largest_component(mask)

    inv = sitk.GetImageFromArray((1 - sitk.GetArrayFromImage(mask).astype(np.uint8)).astype(np.uint8))
    inv.CopyInformation(image)
    inv = _largest_component(inv)
    filled = sitk.GetImageFromArray((1 - sitk.GetArrayFromImage(inv).astype(np.uint8)).astype(np.uint8))
    filled.CopyInformation(image)

    if closing_radius > 0:
        filled = sitk.BinaryMorphologicalClosing(filled, [int(closing_radius)] * 3)
    if dilation_radius > 0:
        filled = sitk.BinaryDilate(filled, [int(dilation_radius)] * 3)
    filled.CopyInformation(image)
    return filled


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


BACKGROUND_INTENSITY_THRESHOLD = 0.01  # suppress low-signal voxels
ADC_SCALE = 1000.0  # scale factor to report in mm^2/s * 1e-3
ADC_NOISE_THRESHOLD = 5.0  # zero-out values above this (pure noise)


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
    reference_image = pairs[0][1]
    #mask_img = compute_body_mask(reference_image, smoothing_sigma=1.0, closing_radius=1, dilation_radius=1)
    #mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
    arrays = []
    for _, im in use_pairs:
        arr = sitk.GetArrayFromImage(im).astype(np.float32)
      #  arr = arr * mask_arr
        arr = np.maximum(arr, 1e-6)
        arrays.append(np.log(arr))
    log_stack = np.stack(arrays, axis=0)
    adc_array = _linear_fit_adc(use_b, log_stack)
    # suppress background where mean signal is low
    mean_signal = np.mean(np.exp(log_stack), axis=0)
    adc_array = np.where(mean_signal >= BACKGROUND_INTENSITY_THRESHOLD, adc_array, 0.0)
    adc_array = adc_array * ADC_SCALE
    adc_array = np.where(adc_array > ADC_NOISE_THRESHOLD, 0.0, adc_array)
    #adc_array = adc_array * mask_arr
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
    prune_dwi_directories(patient_dir)
    return last_written
