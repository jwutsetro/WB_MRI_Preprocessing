# WB MRI Preprocessing

CLI-first preprocessing pipeline for whole-body MRI datasets. Steps: DICOM sorting → NIfTI conversion → ADC creation → noise/bias removal → inter-station intensity standardisation (ISIS) → functional registration → WB reconstruction → diffusion resampling to T1 → Nyul histogram standardisation. Runs locally or on SLURM job arrays.

## Layout
- `Preprocessing/`: pipeline code (`config.py`, `dicom_sort.py`, `pipeline.py`, `nyul.py`, `cli.py`).
- `config/pipeline.example.yaml`: editable template for paths/step toggles; DICOM sequence rules live in `dicom_config.json`.
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
5. Convert DICOMs only:
   ```bash
   python convert_dicom.py /path/to/raw_dicoms /path/to/output \
     --config config/pipeline.yaml
   # or
   python -m Preprocessing.cli convert-dicom /path/to/raw_dicoms /path/to/output --config config/pipeline.yaml
   ```
6. Compute ADC only (after DWI conversion):
   ```bash
   python compute_adc.py /path/to/output_root
   ```
7. Run noise/bias correction only:
   ```bash
   python noise_bias.py /path/to/output_root
   ```
8. Run inter-station intensity standardisation (ISIS) only:
   ```bash
   python ISIS.py /path/to/output_root
   ```
9. Run inter-station registration (ADC-driven) only:
   ```bash
   python register_stations.py /path/to/output_root
   ```

## Notes
- Outputs: `output_dir/<patient>/<modality>/<station>.nii.gz` (stations are numeric). DWI b-values become modality folder names (e.g., `output_dir/<patient>/1000/1.nii.gz`). DICOM metadata is summarized per patient in `output_dir/<patient>/metadata.json`.
- Known sequence names are defined in `dicom_config.json` (e.g., T1, DWI, Dixon_IP/OP/W/F). Update this file to add new sequences.
- New/renamed sequences are logged; run with `--interactive` to map them on the fly.
- All outputs are oriented to `LPS` by default; change `target_orientation` in the config if required.
- ADC computation fits log(S) vs b across all b-values (preferring >0), masks low-signal background (<0.01), clamps to 5.0, and scales by 1000.
- ISIS scales stations linearly from the center outward so overlap regions share the same mean intensity (per modality; ADC is skipped).
- Inter-station registration uses ADC station overlaps to compute rigid translations (scales penalize longitudinal shifts) and applies transforms to all b-value stations.
- Nyul models are stored under `models/` and recomputed when `nyul.refresh` is true or no model exists.
- Tests use synthetic data only: `pytest`.
