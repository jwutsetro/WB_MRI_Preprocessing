# WB MRI Preprocessing

This repository now only contains the whole-body MRI preprocessing utilities (all model training and inference code has been removed).

## Layout
- `Preprocessing/main/`: entry-point scripts for whole-body reconstruction, bone registration, and bounding-box conversion utilities.
- `Preprocessing/modules/`: shared preprocessing routines (registration helpers, intensity standardisation, DICOM handling, etc.) plus example notebooks.
- `Preprocessing/utilities/`: small helpers such as padding utilities.
- `Preprocessing/parameter_files/`: parameter maps for SimpleITK registration steps.

## Notes
- Several scripts contain hard-coded dataset paths; update them before running.
- Dependencies include `SimpleITK`, `numpy`, `pandas`, `tqdm`, `matplotlib`, and `scipy`.
- Notebooks under `Preprocessing/modules/` are kept for reference but are not wired into the scripts directly.
