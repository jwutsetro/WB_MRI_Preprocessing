from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


STEP_ALIASES: Dict[str, str] = {
    "dicom_reconstruction": "dicom_sort",
    "dicom_reconstruct": "dicom_sort",
    "dicom_convert": "dicom_sort",
    "adc_creation": "adc",
    "create_adc": "adc",
    "noise_bias_removal": "noise_bias",
    "bias_correction": "noise_bias",
    "n4_bias": "noise_bias",
    "isis_standardisation": "isis",
    "isis_standardization": "isis",
    "wb_reconstruct": "reconstruct",
    "wb_merge": "reconstruct",
    "merge_wb": "reconstruct",
    "resample": "resample_to_t1",
    "dwi_to_t1": "resample_to_t1",
}


@dataclass
class SequenceRule:
    """Configuration for mapping DICOM series to output modalities."""

    name: str
    description_contains: List[str]
    output_modality: str
    canonical_modality: Optional[str] = None
    is_anatomical: bool = False
    type_tag: Optional[str] = None
    include_types: List[str] = field(default_factory=list)
    b_value_tag: Optional[str] = None
    expect_stations: bool = True
    target_orientation: str = "LPS"
    keep_all_series: bool = False
    background_threshold: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "SequenceRule":
        return cls(
            name=data["name"],
            description_contains=data.get("description_contains", []),
            output_modality=data["output_modality"],
            canonical_modality=data.get("canonical_modality", data["output_modality"]),
            is_anatomical=data.get("is_anatomical", False),
            type_tag=data.get("type_tag"),
            include_types=data.get("include_types", []),
            b_value_tag=data.get("b_value_tag"),
            expect_stations=data.get("expect_stations", True),
            target_orientation=data.get("target_orientation", "LPS"),
            keep_all_series=data.get("keep_all_series", False),
            background_threshold=data.get("background_threshold"),
        )


@dataclass
class StepConfig:
    """Toggles for each pipeline step."""

    dicom_sort: bool = True
    adc: bool = True
    noise_bias: bool = True
    isis: bool = True
    registration: bool = True
    reconstruct: bool = True
    resample_to_t1: bool = True

    @classmethod
    def from_dict(cls, data: Any) -> "StepConfig":
        """Parse step toggles from YAML.

        Accepts:
        - mapping: `{dicom_sort: true, registration: false, ...}`
        - list/tuple: `['dicom_sort', 'adc']` meaning "only these steps"
        - string: `'dicom_sort,adc'` meaning "only these steps"

        Legacy step names are supported via aliases.
        """
        default = cls()
        step_keys = set(default.__dict__.keys())

        def normalize_step_name(name: str) -> str:
            raw = name.strip()
            return STEP_ALIASES.get(raw, raw)

        if data is None:
            return default

        if isinstance(data, str):
            selected = {normalize_step_name(s) for s in data.split(",") if s.strip()}
            merged = {k: False for k in step_keys}
            for step in selected:
                if step in step_keys:
                    merged[step] = True
                else:
                    print(f"[config] Ignoring unknown step '{step}' in steps string.")
            return cls(**merged)

        if isinstance(data, (list, tuple)):
            selected = {normalize_step_name(str(s)) for s in data if str(s).strip()}
            merged = {k: False for k in step_keys}
            for step in selected:
                if step in step_keys:
                    merged[step] = True
                else:
                    print(f"[config] Ignoring unknown step '{step}' in steps list.")
            return cls(**merged)

        if isinstance(data, dict):
            merged = dict(default.__dict__)
            for raw_key, value in (data or {}).items():
                key = normalize_step_name(str(raw_key))
                if key not in step_keys:
                    print(f"[config] Ignoring unknown step toggle '{raw_key}'.")
                    continue
                merged[key] = bool(value)
            return cls(**merged)

        raise TypeError("steps must be a mapping, list, or comma-separated string.")


@dataclass
class NyulConfig:
    enable: bool = True
    bins: int = 120
    landmarks: int = 6
    upper_outlier: float = 99.5
    remove_bg_below: float = 5.0
    model_dir: Path = Path("models")
    modalities: List[str] = field(default_factory=lambda: ["T1", "ADC"])
    refresh: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> "NyulConfig":
        defaults = cls()
        merged = {**defaults.__dict__, **(data or {})}
        merged["model_dir"] = Path(merged["model_dir"])
        return cls(**merged)


@dataclass
class PipelineConfig:
    """Top-level configuration for the preprocessing pipeline."""

    input_dir: Path
    output_dir: Path
    working_dir: Path = Path("work")
    target_orientation: str = "LPS"
    unknown_sequence_log: Path = Path("logs/unknown_sequences.jsonl")
    dicom_rules_path: Path = Path("dicom_config.json")
    sequence_rules: List[SequenceRule] = field(default_factory=list)
    steps: StepConfig = field(default_factory=StepConfig)
    nyul: NyulConfig = field(default_factory=NyulConfig)

    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineConfig":
        if "input_dir" not in data or "output_dir" not in data:
            raise ValueError("Configuration must define input_dir and output_dir.")
        rules_path = Path(data.get("dicom_rules_path", "dicom_config.json"))
        rules = _load_sequence_rules(rules_path, data.get("sequences", []))
        steps = StepConfig.from_dict(data.get("steps", {}))
        nyul_cfg = NyulConfig.from_dict(data.get("nyul", {}))
        if "station_labels" in (data or {}):
            print("[config] 'station_labels' is deprecated and ignored (stations are numeric).")
        return cls(
            input_dir=Path(data["input_dir"]),
            output_dir=Path(data["output_dir"]),
            working_dir=Path(data.get("working_dir", "work")),
            target_orientation=data.get("target_orientation", "LPS"),
            unknown_sequence_log=Path(data.get("unknown_sequence_log", "logs/unknown_sequences.jsonl")),
            dicom_rules_path=rules_path,
            sequence_rules=rules,
            steps=steps,
            nyul=nyul_cfg,
        )


def load_config(config_path: Optional[Path]) -> PipelineConfig:
    """Load configuration from YAML or return defaults with placeholders."""
    if config_path is None:
        raise ValueError("A configuration path is required.")
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return PipelineConfig.from_dict(data)


def _load_sequence_rules(path: Path, fallback_list: List[Dict]) -> List[SequenceRule]:
    """Load sequence rules from JSON file; fallback to provided list for compatibility."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            raw_rules = json.load(handle) or []
    else:
        raw_rules = fallback_list or []
    return [SequenceRule.from_dict(rule) for rule in raw_rules]
