# WB MRI Preprocessing – Agent Notes

## Purpose
Pipeline to turn raw DICOM dumps into WB NIfTI volumes: sort/label series, convert, create ADC, denoise/bias correct, inter-station intensity standardise (ISIS), register functional to anatomical, reconstruct whole-body, resample, and apply Nyul histogram standardisation.

## Key Modules
- `Preprocessing/config.py`: YAML-driven configuration (paths, steps, Nyul settings) plus loader for JSON DICOM rules.
- `Preprocessing/dicom_sort.py`: Maps DICOM series to modalities using `SequenceRule`, logs unknown sequences, writes station NIfTI files.
- `Preprocessing/noise_bias.py`: Anisotropic diffusion + N4 bias correction utilities.
- `Preprocessing/isis.py`: Inter-station intensity standardisation (center-out linear scaling).
- `Preprocessing/registration.py`: ADC-driven inter-station registration using SimpleITK translation-only; transforms applied to all b-values; also registers whole-body ADC to T1.
- `Preprocessing/alignment.py`: Orchestrates alignment steps, supports SLURM array sharding.
- `Preprocessing/nyul.py`: Lightweight Nyul model fit/apply with JSON persistence.
- `Preprocessing/__main__.py`: Entry-point (`python -m Preprocessing run ...`).

## Conventions
- Orientations default to `LPS`; adjust in config if needed.
- Outputs live under `output_dir/<patient>/<modality>/<station>.nii.gz` (numeric stations); DWI b-values become modality folder names (e.g., `/<patient>/1000/1.nii.gz`). Whole-body merges land as `<modality>_WB.nii.gz`.
- Per-patient metadata in `output_dir/<patient>/metadata.json` (series mapping, station counts, b-values, anatomical modalities).
- Unknown series are appended to `logs/unknown_sequences.jsonl`; in interactive mode the CLI prompts for mapping.
- Sequence rules live in `dicom_config.json`; update that file to add/adjust mappings (e.g., Dixon_IP/OP/W/F, DWI).

## Development
- Add docstrings to all public functions/classes.
- Prefer pure functions with explicit inputs/outputs; avoid hardcoded paths.
- Keep tests synthetic and fast (no real DICOM data).
- When adding new sequences, extend `config/pipeline.example.yaml` and update docs.

## Running
- Always activate the repo virtualenv (`source venv/bin/activate`) so SimpleITK-SimpleElastix is available.
- Full pipeline: `python -m Preprocessing run --config config/pipeline.example.yaml`.
- Sequence scan only: `python -m Preprocessing scan-sequences --config ... --patient-dir ...`.
- Use `--array-index/--array-size` for SLURM job arrays.

## Testing
- `pytest` from repo root; tests rely on synthetic data only.
