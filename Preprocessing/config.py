from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml


DEFAULT_STATION_LABELS: List[str] = [
    "head",
    "torso",
    "pelvis",
    "legs",
    "lower_legs",
    "upper_feet",
    "feet",
]


@dataclass
class SequenceRule:
    """Configuration for mapping DICOM series to output modalities."""

    name: str
    description_contains: List[str]
    output_modality: str
    type_tag: Optional[str] = None
    include_types: List[str] = field(default_factory=list)
    b_value_tag: Optional[str] = None
    expect_stations: bool = True
    target_orientation: str = "LPS"
    keep_all_series: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> "SequenceRule":
        return cls(
            name=data["name"],
            description_contains=data.get("description_contains", []),
            output_modality=data["output_modality"],
            type_tag=data.get("type_tag"),
            include_types=data.get("include_types", []),
            b_value_tag=data.get("b_value_tag"),
            expect_stations=data.get("expect_stations", True),
            target_orientation=data.get("target_orientation", "LPS"),
            keep_all_series=data.get("keep_all_series", False),
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
    nyul: bool = True

    @classmethod
    def from_dict(cls, data: Dict) -> "StepConfig":
        default = cls()
        merged = {**default.__dict__, **(data or {})}
        return cls(**merged)


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
    station_labels: List[str] = field(default_factory=lambda: DEFAULT_STATION_LABELS.copy())
    unknown_sequence_log: Path = Path("logs/unknown_sequences.jsonl")
    sequence_rules: List[SequenceRule] = field(default_factory=list)
    steps: StepConfig = field(default_factory=StepConfig)
    nyul: NyulConfig = field(default_factory=NyulConfig)

    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineConfig":
        if "input_dir" not in data or "output_dir" not in data:
            raise ValueError("Configuration must define input_dir and output_dir.")
        rules = [SequenceRule.from_dict(rule) for rule in data.get("sequences", [])]
        steps = StepConfig.from_dict(data.get("steps", {}))
        nyul_cfg = NyulConfig.from_dict(data.get("nyul", {}))
        station_labels = data.get("station_labels", DEFAULT_STATION_LABELS)
        return cls(
            input_dir=Path(data["input_dir"]),
            output_dir=Path(data["output_dir"]),
            working_dir=Path(data.get("working_dir", "work")),
            target_orientation=data.get("target_orientation", "LPS"),
            station_labels=station_labels,
            unknown_sequence_log=Path(data.get("unknown_sequence_log", "logs/unknown_sequences.jsonl")),
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

