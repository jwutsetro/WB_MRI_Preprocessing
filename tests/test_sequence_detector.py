from pathlib import Path

from Preprocessing.config import PipelineConfig, SequenceRule, StepConfig, NyulConfig
from Preprocessing.dicom_sort import SequenceDetector


def test_unknown_sequence_logs(tmp_path: Path):
    rule = SequenceRule(
        name="t1",
        description_contains=["t1"],
        output_modality="T1",
    )
    cfg = PipelineConfig(
        input_dir=tmp_path,
        output_dir=tmp_path,
        sequence_rules=[rule],
        steps=StepConfig(),
        nyul=NyulConfig(),
    )
    detector = SequenceDetector(cfg.sequence_rules)
    match = detector.match("Unknown sequence", {})
    assert match is None


def test_known_sequence_matches(tmp_path: Path):
    rule = SequenceRule(
        name="dwi",
        description_contains=["dwibs", "dwi"],
        output_modality="DWI",
        b_value_tag="0018|9087",
    )
    cfg = PipelineConfig(
        input_dir=tmp_path,
        output_dir=tmp_path,
        sequence_rules=[rule],
        steps=StepConfig(),
        nyul=NyulConfig(),
    )
    detector = SequenceDetector(cfg.sequence_rules)
    match = detector.match("DWIBS axial", {"0018|9087": "1000"})
    assert match is not None
    assert match.output_modality == "DWI"

