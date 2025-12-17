# WB MRI Preprocessing

CLI-first preprocessing pipeline for whole-body MRI datasets. Steps: DICOM sorting → NIfTI conversion → ADC creation → noise/bias removal → inter-station intensity standardisation (ISIS) → functional registration → WB reconstruction → diffusion resampling to T1 → Nyul histogram standardisation. Runs locally or on SLURM job arrays.

## Layout
- `Preprocessing/`: pipeline code (`config.py`, `dicom_sort.py`, `alignment.py`, `nyul.py`).
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
   python -m Preprocessing run --config config/pipeline.yaml
   # optional SLURM sharding
   # python -m Preprocessing run --config config/pipeline.yaml --array-index 0 --array-size 8
   # run a single patient folder (useful for SLURM per-patient jobs)
   # python -m Preprocessing run --config config/pipeline.yaml --patient-dir /path/to/patient
   # run only a slice of the pipeline (development/debugging)
   # python -m Preprocessing run --config config/pipeline.yaml --from-step adc --to-step reconstruct
   # python -m Preprocessing run --config config/pipeline.yaml --steps dicom_sort,adc
   ```
4. Scan sequences without processing (logs unknown SeriesDescriptions to `logs/unknown_sequences.jsonl`):
   ```bash
   python -m Preprocessing scan-sequences --config config/pipeline.yaml --patient-dir /path/to/patient
   ```
5. Run a single step (examples):
   - Convert only: `python -m Preprocessing run --config config/pipeline.yaml --steps dicom_sort`
   - ADC only: `python -m Preprocessing run --config config/pipeline.yaml --steps adc`
   - Merge only: `python -m Preprocessing run --config config/pipeline.yaml --steps reconstruct`

### SimpleElastix (registration)
- Recommended: install the PyPI build that bundles Elastix/SimpleITK: `pip install SimpleITK-SimpleElastix`.
- If you need to build locally (e.g., for newer SimpleITK), use the SimpleElastix superbuild on macOS/Apple Silicon:
  ```bash
  brew install cmake ninja git
  git clone https://github.com/SuperElastix/SimpleElastix.git
  cd SimpleElastix/SuperBuild
  mkdir build && cd build
  cmake -GNinja -DCMAKE_OSX_ARCHITECTURES=arm64 -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DUSE_ELASTIX=ON -DUSE_OPENMP=OFF -DPYTHON_EXECUTABLE=$(which python3) ..
  ninja
  python3 -m pip install --upgrade --force-reinstall SimpleITK-build/Wrapping/Python/dist/SimpleITK-*.whl
  python - <<'PY'
  import SimpleITK as sitk; print("Elastix available:", hasattr(sitk, "Elastix"))
  PY
  ```
  (The default PyPI `SimpleITK` wheel lacks SimpleElastix.)

## Notes
- Outputs: `output_dir/<patient>/<modality>/<station>.nii.gz` (stations are numeric). DWI b-values become modality folder names (e.g., `output_dir/<patient>/1000/1.nii.gz`). Whole-body merges write a single `<modality>.nii.gz` with feathered overlaps and remove per-station folders. DICOM metadata is summarized per patient in `output_dir/<patient>/metadata.json`.
- Known sequence names are defined in `dicom_config.json` (e.g., T1, DWI, Dixon_IP/OP/W/F). Update this file to add new sequences.
- New/renamed sequences are logged; run with `--interactive` to map them on the fly.
- All outputs are oriented to `LPS` by default; change `target_orientation` in the config if required.
- ADC computation fits log(S) vs b across all b-values (preferring >0), masks low-signal background (<0.01), zeros pure noise (>5.0), and scales by 1000.
- ISIS scales stations linearly from the center outward so overlap regions share the same mean intensity (per modality; ADC is skipped).
- Inter-station registration uses ADC station overlaps with SimpleElastix parameter maps (`Preprocessing/parameter_files`) and applies transforms to all DWI stations from the center outward.
- Station merges use feathered linear blending across overlaps to avoid seams between stations; station folders are cleaned up post-merge.
- Whole-body DWI-to-anatomical registration uses the same Elastix parameter maps (Euler + B-spline) and updates all DWI WB volumes to the anatomical WB space.
- Nyul models are stored under `models/` and recomputed when `nyul.refresh` is true or no model exists; run it as a dedicated command after all patients are processed:
  `python -m Preprocessing nyul --config config/pipeline.yaml --mode fit-apply`.
- Tests use synthetic data only: `pytest`.
