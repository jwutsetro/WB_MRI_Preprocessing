from __future__ import annotations

from pathlib import Path

from Preprocessing.alignment import AlignmentRunner
from Preprocessing.config import NyulConfig, PipelineConfig, StepConfig


def _cfg(tmp_path: Path) -> PipelineConfig:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        sequence_rules=[],
        steps=StepConfig(),
        nyul=NyulConfig(enable=False),
    )


def test_alignment_runner_continues_after_patient_failure(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    (cfg.input_dir / "p1").mkdir()
    (cfg.input_dir / "p2").mkdir()

    runner = AlignmentRunner(cfg=cfg, interactive=False)

    def fake_run_patient(patient_input: Path, patient_output: Path) -> None:
        if patient_input.name == "p1":
            raise RuntimeError("boom")
        (patient_output / "ok.txt").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_patient", fake_run_patient)
    runner.run()

    err = cfg.output_dir / "p1" / "pipeline_error.txt"
    assert err.exists()
    assert "RuntimeError: boom" in err.read_text(encoding="utf-8")

    assert (cfg.output_dir / "p2" / "ok.txt").exists()


def test_alignment_runner_single_patient_writes_error_and_returns(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    patient_dir = cfg.input_dir / "single"
    patient_dir.mkdir()

    runner = AlignmentRunner(cfg=cfg, interactive=False, patient_dir=patient_dir)

    def fake_run_patient(patient_input: Path, patient_output: Path) -> None:
        raise ValueError("bad patient")

    monkeypatch.setattr(runner, "_run_patient", fake_run_patient)
    runner.run()

    err = cfg.output_dir / "single" / "pipeline_error.txt"
    assert err.exists()
    assert "ValueError: bad patient" in err.read_text(encoding="utf-8")


def test_alignment_runner_skips_patient_when_nifti_already_exists(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    (cfg.input_dir / "p1").mkdir()
    (cfg.input_dir / "p2").mkdir()

    existing = cfg.output_dir / "p1" / "T1"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "1.nii.gz").write_text("dummy", encoding="utf-8")

    runner = AlignmentRunner(cfg=cfg, interactive=False)

    def fake_run_patient(patient_input: Path, patient_output: Path) -> None:
        (patient_output / "ran.txt").write_text(patient_input.name, encoding="utf-8")

    monkeypatch.setattr(runner, "_run_patient", fake_run_patient)
    runner.run()

    assert not (cfg.output_dir / "p1" / "ran.txt").exists()
    assert (cfg.output_dir / "p2" / "ran.txt").read_text(encoding="utf-8") == "p2"


def test_alignment_runner_single_patient_skip_when_nifti_exists(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    patient_dir = cfg.input_dir / "single"
    patient_dir.mkdir()

    existing = cfg.output_dir / "single" / "ADC"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "1.nii.gz").write_text("dummy", encoding="utf-8")

    runner = AlignmentRunner(cfg=cfg, interactive=False, patient_dir=patient_dir)

    def fake_run_patient(patient_input: Path, patient_output: Path) -> None:
        (patient_output / "ran.txt").write_text("should_not_run", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_patient", fake_run_patient)
    runner.run()

    assert not (cfg.output_dir / "single" / "ran.txt").exists()
