import numpy as np
import SimpleITK as sitk
from pathlib import Path

from Preprocessing.config import NyulConfig
from Preprocessing.nyul import NyulModel, fit_nyul_model


def _synthetic_image(value: float) -> sitk.Image:
    array = np.ones((4, 4, 4), dtype=np.float32) * value
    image = sitk.GetImageFromArray(array)
    return image


def test_nyul_fit_and_apply():
    cfg = NyulConfig(bins=10, landmarks=4, upper_outlier=99.0, remove_bg_below=0.0, model_dir=Path("models"), modalities=["T1"])
    images = [_synthetic_image(10.0), _synthetic_image(20.0)]
    model = fit_nyul_model("T1", images, cfg)
    assert model.landmarks, "Landmarks should be computed"
    target_image = _synthetic_image(15.0)
    transformed = model.apply(target_image)
    arr = sitk.GetArrayFromImage(transformed)
    assert arr.shape == target_image.GetSize()[::-1]
    assert np.any(arr), "Output should not be empty"
