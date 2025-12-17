from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import SimpleITK as sitk

from Preprocessing.config import NyulConfig


@dataclass
class NyulModel:
    modality: str
    landmarks: List[float]
    reference_landmarks: List[float]
    upper_outlier: float

    def to_dict(self) -> Dict:
        return {
            "modality": self.modality,
            "landmarks": self.landmarks,
            "reference_landmarks": self.reference_landmarks,
            "upper_outlier": self.upper_outlier,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NyulModel":
        return cls(
            modality=data["modality"],
            landmarks=list(data["landmarks"]),
            reference_landmarks=list(data["reference_landmarks"]),
            upper_outlier=float(data["upper_outlier"]),
        )

    def apply(self, image: sitk.Image) -> sitk.Image:
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        flat = array.flatten()
        result = np.zeros_like(flat)
        lm = self.landmarks
        ref = self.reference_landmarks
        for idx in range(len(lm) - 1):
            mask = (flat >= lm[idx]) & (flat <= lm[idx + 1]) if idx < len(lm) - 2 else flat >= lm[idx]
            if not np.any(mask):
                continue
            a = (ref[idx + 1] - ref[idx]) / (lm[idx + 1] - lm[idx] + 1e-8)
            b = ref[idx] - a * lm[idx]
            result[mask] = a * flat[mask] + b
        result = result.reshape(array.shape)
        out = sitk.GetImageFromArray(result)
        out.CopyInformation(image)
        return out


def _percentile_landmarks(image: np.ndarray, num_landmarks: int, upper_outlier: float, bg_threshold: float) -> List[float]:
    working = image.copy()
    working[working < bg_threshold] = np.nan
    percentiles = np.linspace(0, upper_outlier, num_landmarks - 1).tolist() + [upper_outlier]
    values = [np.nanpercentile(working, p) for p in percentiles]
    return values


def fit_nyul_model(
    modality: str,
    images: Sequence[sitk.Image],
    cfg: NyulConfig,
) -> NyulModel:
    arrays = [sitk.GetArrayFromImage(im).astype(np.float32) for im in images]
    per_image_landmarks = [
        _percentile_landmarks(arr, cfg.landmarks, cfg.upper_outlier, cfg.remove_bg_below) for arr in arrays
    ]
    stacked = np.stack(per_image_landmarks, axis=0)
    reference_landmarks = np.nanmedian(stacked, axis=0).tolist()
    return NyulModel(
        modality=modality,
        landmarks=reference_landmarks,
        reference_landmarks=reference_landmarks,
        upper_outlier=cfg.upper_outlier,
    )


def save_model(model: NyulModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(model.to_dict(), handle, indent=2)


def load_model(path: Path) -> NyulModel:
    """Load a Nyul histogram standardisation model from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return NyulModel.from_dict(data)


def nyul_collect_modality_files(output_root: Path, modality: str) -> List[Path]:
    """Collect whole-body NIfTI files for a modality under an output root."""
    files = sorted(output_root.glob(f"*/{modality}.nii.gz"))
    if files:
        return files
    return sorted(output_root.glob(f"*/*{modality}_WB.nii.gz"))


def nyul_ensure_models(output_root: Path, cfg: NyulConfig) -> None:
    """Ensure Nyul models exist (fit if missing or refresh is enabled)."""
    model_dir = cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    for modality in cfg.modalities:
        modality_files = nyul_collect_modality_files(output_root, modality)
        if not modality_files:
            continue
        model_path = model_dir / f"{modality}_nyul.json"
        if not cfg.refresh and model_path.exists():
            continue
        images: List[sitk.Image] = [sitk.ReadImage(str(path)) for path in modality_files]
        model = fit_nyul_model(modality, images, cfg)
        save_model(model, model_path)


def nyul_apply_models(output_root: Path, cfg: NyulConfig) -> None:
    """Apply existing Nyul models to all matching whole-body outputs."""
    for modality in cfg.modalities:
        model_path = cfg.model_dir / f"{modality}_nyul.json"
        if not model_path.exists():
            continue
        model = load_model(model_path)
        modality_files = nyul_collect_modality_files(output_root, modality)
        for path in modality_files:
            img = sitk.ReadImage(str(path))
            out = model.apply(img)
            sitk.WriteImage(out, str(path), True)
