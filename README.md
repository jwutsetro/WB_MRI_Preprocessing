# WB MRI Preprocessing

CLI-first preprocessing pipeline for whole-body MRI datasets. Steps: DICOM sorting → NIfTI conversion → ADC creation → noise/bias removal → inter-station intensity standardisation (ISIS) → functional registration → WB reconstruction → diffusion resampling to T1 → Nyul histogram standardisation. Runs locally or on SLURM job arrays.

## Layout
- `Preprocessing/`: pipeline code (`config.py`, `dicom_sort.py`, `pipeline.py`, `nyul.py`, `cli.py`) plus legacy modules for reference.
- `config/pipeline.example.yaml`: editable template for sequence rules, paths, and step toggles.
- `Preprocessing/modules/`: legacy utilities kept for reference.
- `tests/`: synthetic fast tests for core utilities.

## Quickstart
1. Create a config from the template:
   ```bash
   cp config/pipeline.example.yaml config/pipeline.yaml
   # edit paths, sequence rules, and step toggles
   ```
2. Install deps (prefer a virtualenv):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python -m Preprocessing.cli run --config config/pipeline.yaml
   # optional SLURM sharding
   # python -m Preprocessing.cli run --config config/pipeline.yaml --array-index 0 --array-size 8
   ```
4. Scan sequences without processing (logs unknown SeriesDescriptions to `logs/unknown_sequences.jsonl`):
   ```bash
   python -m Preprocessing.cli scan-sequences --config config/pipeline.yaml --patient-dir /path/to/patient
   ```

## Notes
- New/renamed sequences are logged; run with `--interactive` to map them on the fly.
- All outputs are oriented to `LPS` by default; change `target_orientation` in the config if required.
- Nyul models are stored under `models/` and recomputed when `nyul.refresh` is true or no model exists.
- Tests use synthetic data only: `pytest`.
