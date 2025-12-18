from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
from typing import Optional

import SimpleITK as sitk

from Preprocessing.adc import compute_body_mask
from Preprocessing.config import EddyConfig
from Preprocessing.dcm2niix import load_bvals, load_bvecs


def _which_any(*names: str) -> Optional[str]:
    for name in names:
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_patient_metadata(patient_output: Path) -> dict:
    path = patient_output / "metadata.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_patient_metadata(patient_output: Path, meta: dict) -> None:
    path = patient_output / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _update_patient_metadata(patient_output: Path, eddy_update: dict) -> None:
    meta = _load_patient_metadata(patient_output)
    meta.setdefault("eddy", {})
    if not isinstance(meta["eddy"], dict):
        meta["eddy"] = {}
    meta["eddy"].update(eddy_update)
    _save_patient_metadata(patient_output, meta)


def _pe_to_fsl_vector(pe: str) -> str:
    pe = pe.strip()
    mapping = {
        "i": "1 0 0",
        "i-": "-1 0 0",
        "j": "0 1 0",
        "j-": "0 -1 0",
        "k": "0 0 1",
        "k-": "0 0 -1",
    }
    if pe in mapping:
        return mapping[pe]
    # already looks like "0 1 0"
    parts = pe.split()
    if len(parts) == 3:
        float(parts[0]), float(parts[1]), float(parts[2])
        return pe
    raise ValueError("phase encoding must be BIDS-like (i/j/k or i-/j-/k-) or '0 1 0'.")


def _acqp_params(cfg: EddyConfig, *, json_path: Optional[Path] = None) -> tuple[str, float]:
    pe_vec = _pe_to_fsl_vector(cfg.phase_encoding)
    total_readout = float(cfg.total_readout_time)
    if json_path is not None and json_path.exists():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            pe_bids = meta.get("PhaseEncodingDirection")
            trot = meta.get("TotalReadoutTime")
            if isinstance(pe_bids, str) and pe_bids.strip():
                pe_vec = _pe_to_fsl_vector(pe_bids)
            if trot is not None:
                total_readout = float(trot)
        except Exception:
            pass
    return pe_vec, total_readout


def _acqp_line(cfg: EddyConfig, *, json_path: Optional[Path] = None) -> str:
    pe_vec, total_readout = _acqp_params(cfg, json_path=json_path)
    return f"{pe_vec} {total_readout}"


def _num_volumes(image: sitk.Image) -> int:
    if image.GetDimension() == 4:
        return int(image.GetSize()[3])
    return 1


def _station_id(path: Path) -> str:
    name = path.name
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return path.stem


def _orient(image: sitk.Image, target_orientation: str) -> sitk.Image:
    try:
        return sitk.DICOMOrient(image, target_orientation)
    except Exception:
        return image


def _split_4d_to_b_means(
    image_4d: sitk.Image, bvals: list[float], *, target_orientation: str
) -> dict[int, sitk.Image]:
    if image_4d.GetDimension() != 4:
        raise ValueError("Expected 4D DWI image.")
    size4 = list(image_4d.GetSize())
    t_size = int(size4[3])
    if len(bvals) != t_size:
        raise ValueError(f"bvals length ({len(bvals)}) does not match 4D size ({t_size}).")

    extract_size = size4[:]
    extract_size[3] = 0
    oriented_vols: list[sitk.Image] = []
    groups: dict[int, list[sitk.Image]] = {}
    for t in range(t_size):
        vol = sitk.Extract(image_4d, extract_size, [0, 0, 0, int(t)])
        vol = _orient(vol, target_orientation)
        oriented_vols.append(vol)
        b_int = int(round(float(bvals[t])))
        groups.setdefault(b_int, []).append(vol)

    # compute union crop box (remove empty padding consistently)
    union_nonempty = None
    for vol in oriented_vols:
        arr = sitk.GetArrayFromImage(vol)  # z,y,x
        nonempty = (abs(arr) > 0).astype("uint8")
        union_nonempty = nonempty if union_nonempty is None else (union_nonempty | nonempty)
    crop_index = (0, 0, 0)
    crop_size = tuple(int(s) for s in oriented_vols[0].GetSize())
    if union_nonempty is not None and union_nonempty.any():
        import numpy as np

        z_any = union_nonempty.any(axis=(1, 2))
        y_any = union_nonempty.any(axis=(0, 2))
        x_any = union_nonempty.any(axis=(0, 1))
        z_idx = np.where(z_any)[0]
        y_idx = np.where(y_any)[0]
        x_idx = np.where(x_any)[0]
        z0, z1 = int(z_idx[0]), int(z_idx[-1] + 1)
        y0, y1 = int(y_idx[0]), int(y_idx[-1] + 1)
        x0, x1 = int(x_idx[0]), int(x_idx[-1] + 1)
        crop_index = (x0, y0, z0)
        crop_size = (x1 - x0, y1 - y0, z1 - z0)

    out: dict[int, sitk.Image] = {}
    for b_int, vols in groups.items():
        if len(vols) == 1:
            mean_vol = vols[0]
        else:
            import numpy as np

            arrays = [sitk.GetArrayFromImage(v).astype(np.float32) for v in vols]
            mean_arr = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
            mean_vol = sitk.GetImageFromArray(mean_arr)
            mean_vol.CopyInformation(vols[0])
        if crop_size != tuple(int(s) for s in mean_vol.GetSize()):
            mean_vol = sitk.RegionOfInterest(mean_vol, size=crop_size, index=crop_index)
        out[int(b_int)] = mean_vol
    return out


def _rewrite_split_outputs(
    *,
    patient_output: Path,
    station: str,
    image_4d: sitk.Image,
    bvals: list[float],
    target_orientation: str,
) -> None:
    b_means = _split_4d_to_b_means(image_4d, bvals, target_orientation=target_orientation)
    for b_int, vol in b_means.items():
        b_dir = patient_output / str(int(b_int))
        b_dir.mkdir(parents=True, exist_ok=True)
        out_path = b_dir / f"{station}.nii.gz"
        sitk.WriteImage(vol, str(out_path), True)


def _bvecs_usable_for_eddy(bvals: list[float], bvecs_rows: list[list[float]]) -> tuple[bool, str]:
    """Return (ok, reason) if bvecs contain usable directional information.

    For DWIBS/trace-style DWI, bvecs are often all zeros or constant; eddy can be
    unstable or unhelpful. We skip in those cases for robustness.
    """
    if len(bvecs_rows) != 3:
        return False, "invalid_bvec_shape"
    n = len(bvals)
    if any(len(r) != n for r in bvecs_rows):
        return False, "bvec_length_mismatch"

    dw_idx = [i for i, b in enumerate(bvals) if float(b) > 50.0]
    if len(dw_idx) < 3:
        return False, "too_few_dw_volumes"

    dirs = []
    for i in dw_idx:
        x, y, z = float(bvecs_rows[0][i]), float(bvecs_rows[1][i]), float(bvecs_rows[2][i])
        norm = (x * x + y * y + z * z) ** 0.5
        if norm < 1e-3:
            dirs.append((0.0, 0.0, 0.0))
            continue
        # sign-invariant normalized direction
        dirs.append((abs(x / norm), abs(y / norm), abs(z / norm)))

    if all(d == (0.0, 0.0, 0.0) for d in dirs):
        return False, "bvec_all_zero"

    uniq = {(round(d[0], 2), round(d[1], 2), round(d[2], 2)) for d in dirs if d != (0.0, 0.0, 0.0)}
    if len(uniq) < 2:
        return False, "bvec_not_varied"
    return True, "ok"


def run_eddy_for_patient(patient_output: Path, cfg: EddyConfig, *, target_orientation: str) -> None:
    """Run FSL eddy per-station on raw DWI 4D outputs if available.

    Robustness policy:
    - If cfg.enable is False: do nothing.
    - If FSL eddy is not available: print a warning and do nothing.
    - If no raw DWI folder exists: do nothing.
    - If station missing bval/bvec: skip that station.

    This step expects `dicom_sort` (dcm2niix mode) to have produced:
      - `<patient>/{cfg.raw_dwi_dir_name}/{station}.nii.gz` (4D)
      - `<patient>/{cfg.raw_dwi_dir_name}/{station}.bval`
      - `<patient>/{cfg.raw_dwi_dir_name}/{station}.bvec`
    """
    if not cfg.enable:
        return

    eddy_exe = _which_any("eddy_openmp", "eddy")
    if not eddy_exe:
        print("[eddy] FSL eddy not found on PATH; skipping eddy correction.")
        _update_patient_metadata(
            patient_output,
            {
                "enabled": True,
                "status": "skipped_no_binary",
                "binary": None,
                "raw_dir": cfg.raw_dwi_dir_name,
                "out_dir": cfg.out_dir_name,
            },
        )
        return

    raw_dir = patient_output / cfg.raw_dwi_dir_name
    if not raw_dir.is_dir():
        _update_patient_metadata(
            patient_output,
            {
                "enabled": True,
                "status": "skipped_no_raw_dwi",
                "binary": Path(eddy_exe).name,
                "raw_dir": cfg.raw_dwi_dir_name,
                "out_dir": cfg.out_dir_name,
            },
        )
        return

    out_root = patient_output / cfg.out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    stations_meta: dict[str, dict] = {}

    for nifti_path in sorted(raw_dir.glob("*.nii*")):
        station = _station_id(nifti_path)
        bval = raw_dir / f"{station}.bval"
        bvec = raw_dir / f"{station}.bvec"
        sidecar_json = raw_dir / f"{station}.json"
        if not bval.exists() or not bvec.exists():
            print(f"[eddy] Missing bval/bvec for {station}; skipping.")
            stations_meta[station] = {"status": "skipped", "reason": "missing_bval_bvec"}
            continue

        try:
            image = sitk.ReadImage(str(nifti_path))
        except Exception:
            print(f"[eddy] Failed to read {nifti_path}; skipping.")
            stations_meta[station] = {"status": "skipped", "reason": "read_failed"}
            continue

        nvol = _num_volumes(image)
        if nvol < 2:
            stations_meta[station] = {"status": "skipped", "reason": "single_volume", "nvol": nvol}
            continue

        try:
            bvals_list = load_bvals(bval)
            bvecs_rows = load_bvecs(bvec)
        except Exception:
            stations_meta[station] = {"status": "skipped", "reason": "invalid_bval_bvec", "nvol": nvol}
            continue

        ok, reason = _bvecs_usable_for_eddy(bvals_list, bvecs_rows)
        if not ok:
            print(f"[eddy] Skipping {station}: {reason}")
            stations_meta[station] = {"status": "skipped", "reason": reason, "nvol": nvol}
            continue

        station_out = out_root / station
        station_out.mkdir(parents=True, exist_ok=True)

        acqp_path = station_out / "acqp.txt"
        index_path = station_out / "index.txt"

        pe_vec, trot = _acqp_params(cfg, json_path=sidecar_json)
        _write_text(acqp_path, _acqp_line(cfg, json_path=sidecar_json) + "\n")
        _write_text(index_path, " ".join(["1"] * nvol) + "\n")

        # Build a conservative body mask from the mean volume.
        # (eddy needs a mask; for whole-body DWI this is not a brain mask)
        try:
            mean_3d = sitk.MeanProjection(image, projectionDimension=3)
        except Exception:
            # fallback: use first volume
            size4 = list(image.GetSize())
            size4[3] = 0
            mean_3d = sitk.Extract(image, size4, [0, 0, 0, 0])
        mask = compute_body_mask(mean_3d, threshold=None, closing_radius=1, dilation_radius=1, padding_threshold=0.0)
        mask_path = station_out / "mask.nii.gz"
        sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(mask_path), True)

        out_prefix = station_out / "eddy_corrected"
        cmd = [
            eddy_exe,
            f"--imain={nifti_path}",
            f"--mask={mask_path}",
            f"--acqp={acqp_path}",
            f"--index={index_path}",
            f"--bvecs={bvec}",
            f"--bvals={bval}",
            f"--out={out_prefix}",
        ]
        if cfg.repol:
            cmd.append("--repol")

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            print(f"[eddy] Failed for {station}: {e}; skipping station.")
            stations_meta[station] = {"status": "skipped", "reason": "eddy_failed", "nvol": nvol}
            continue

        corrected = station_out / "eddy_corrected.nii.gz"
        rotated_bvec = station_out / "eddy_corrected.eddy_rotated_bvecs"
        if not corrected.exists():
            print(f"[eddy] No corrected output for {station}; skipping.")
            stations_meta[station] = {"status": "skipped", "reason": "no_output", "nvol": nvol}
            continue

        # Atomically replace the raw DWI with the corrected 4D; keep originals in eddy output folder.
        tmp_target = raw_dir / f"{station}.tmp.nii.gz"
        corrected.replace(tmp_target)
        tmp_target.replace(nifti_path)
        if rotated_bvec.exists():
            rotated_bvec.replace(bvec)

        # Update per-b-value station files so downstream ADC/noise_bias use corrected data.
        try:
            corrected_img = sitk.ReadImage(str(nifti_path))
            _rewrite_split_outputs(
                patient_output=patient_output,
                station=station,
                image_4d=corrected_img,
                bvals=bvals_list,
                target_orientation=target_orientation,
            )
            stations_meta[station] = {
                "status": "ran",
                "nvol": nvol,
                "pe_vec": pe_vec,
                "total_readout_time": trot,
                "repol": bool(cfg.repol),
            }
        except Exception as e:
            print(f"[eddy] Failed to update split outputs for {station}: {e}")
            stations_meta[station] = {"status": "ran_partial", "nvol": nvol, "reason": "split_update_failed"}

    _update_patient_metadata(
        patient_output,
        {
            "enabled": True,
            "status": "completed",
            "binary": Path(eddy_exe).name,
            "raw_dir": cfg.raw_dwi_dir_name,
            "out_dir": cfg.out_dir_name,
            "stations": stations_meta,
        },
    )
