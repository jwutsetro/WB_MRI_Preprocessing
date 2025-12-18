from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from Preprocessing.adc import compute_body_mask


def test_body_mask_fills_low_intensity_holes() -> None:
    arr = np.zeros((12, 24, 24), dtype=np.float32)  # z, y, x
    arr[2:10, 6:18, 6:18] = 100.0
    arr[5:7, 11:13, 11:13] = 0.0  # internal low-signal pocket

    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((2.0, 2.0, 5.0))

    mask = compute_body_mask(image, threshold=50.0, closing_radius=0, dilation_radius=0, padding_threshold=0.0)
    mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)

    assert mask_arr[6, 12, 12] == 1  # hole is filled
    assert mask_arr[0, 0, 0] == 0


def test_body_mask_ignores_empty_padding_borders() -> None:
    arr = np.zeros((10, 20, 20), dtype=np.float32)
    arr[3:7, 7:13, 7:13] = 80.0

    image = sitk.GetImageFromArray(arr)
    mask = compute_body_mask(image, threshold=20.0, closing_radius=0, dilation_radius=0, padding_threshold=0.0)
    mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)

    assert mask_arr.sum() > 0
    assert mask_arr[0, :, :].sum() == 0
    assert mask_arr[:, 0, :].sum() == 0
    assert mask_arr[:, :, 0].sum() == 0

