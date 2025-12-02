from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import SimpleITK as sitk
import numpy as np


from Preprocessing.adc import compute_body_mask
from Preprocessing.config import PipelineConfig, SequenceRule
from Preprocessing.utils import ANATOMICAL_PRIORITY, prune_anatomical_modalities


def _crop_zero_padding_z(image: sitk.Image, threshold: float = 0.0) -> sitk.Image:
    """Trim leading/trailing slices along z that are all below threshold, preserving spatial info."""
    arr = sitk.GetArrayFromImage(image)  # z, y, x
    non_zero = np.abs(arr) > threshold
    z_any = non_zero.any(axis=(1, 2))
    if not z_any.any():
        return image
    z_idx = np.where(z_any)[0]
    z_start, z_end = int(z_idx[0]), int(z_idx[-1] + 1)
    if z_start == 0 and z_end == arr.shape[0]:
        return image
    cropped_arr = arr[z_start:z_end, :, :]
    cropped = sitk.GetImageFromArray(cropped_arr)
    cropped = sitk.Cast(cropped, image.GetPixelID())
    cropped.SetSpacing(image.GetSpacing())
    cropped.SetDirection(image.GetDirection())
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    cropped.SetOrigin((origin[0], origin[1], origin[2] + z_start * spacing[2]))
    return cropped


@dataclass
class DicomInstance:
    filepath: Path
    series_uid: str
    series_description: str
    protocol_name: str
    instance_number: int
    station_name: Optional[str]
    type_value: Optional[str]
    b_value: Optional[str]
    metadata: Dict[str, str] = field(default_factory=dict)


class SequenceDetector:
    """Match DICOM metadata to configured sequence rules."""

    def __init__(self, rules: Sequence[SequenceRule]) -> None:
        self.rules = list(rules)

    def match(self, description: str, metadata: Dict[str, str]) -> Optional[SequenceRule]:
        lowered = description.lower()
        for rule in self.rules:
            if not any(fragment.lower() in lowered for fragment in rule.description_contains):
                continue
            if rule.type_tag:
                value = metadata.get(rule.type_tag, "").strip()
                if rule.include_types and value not in rule.include_types:
                    continue
            return rule
        return None


class DicomSorter:
    """Scan DICOM folders, map to configured modalities, and write NIfTI outputs."""

    def __init__(self, cfg: PipelineConfig, interactive: bool = False) -> None:
        self.cfg = cfg
        self.detector = SequenceDetector(cfg.sequence_rules)
        self.interactive = interactive

    def _collect_instances(self, patient_dir: Path) -> List[DicomInstance]:
        instances: List[DicomInstance] = []
        for filepath in patient_dir.rglob("*"):
            if not filepath.is_file():
                continue
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(filepath))
            reader.LoadPrivateTagsOn()
            try:
                reader.ReadImageInformation()
            except RuntimeError:
                continue
            meta = {
                "0008|103e": reader.GetMetaData("0008|103e") if reader.HasMetaDataKey("0008|103e") else "",
                "0018|1030": reader.GetMetaData("0018|1030") if reader.HasMetaDataKey("0018|1030") else "",
            }
            series_uid = reader.GetMetaData("0020|000e") if reader.HasMetaDataKey("0020|000e") else ""
            instance_number = int(reader.GetMetaData("0020|0013")) if reader.HasMetaDataKey("0020|0013") else 0
            station_name = reader.GetMetaData("0008|1010") if reader.HasMetaDataKey("0008|1010") else None
            type_value = None
            b_value = None
            extra_meta: Dict[str, str] = {}
            for key in ["2005|1011", "0018|9087"]:
                if reader.HasMetaDataKey(key):
                    value = reader.GetMetaData(key)
                    extra_meta[key] = value
                    if key == "2005|1011":
                        type_value = value
                    if key == "0018|9087":
                        b_value = value
            instances.append(
                DicomInstance(
                    filepath=filepath,
                    series_uid=series_uid,
                    series_description=meta["0008|103e"],
                    protocol_name=meta["0018|1030"],
                    instance_number=instance_number,
                    station_name=station_name,
                    type_value=type_value,
                    b_value=b_value,
                    metadata=extra_meta,
                )
            )
        return instances

    def _log_unknown(self, patient: str, instance: DicomInstance, metadata: Dict[str, str]) -> None:
        log_path = self.cfg.unknown_sequence_log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "patient": patient,
            "file": str(instance.filepath),
            "series_description": instance.series_description,
            "protocol_name": instance.protocol_name,
            "series_uid": instance.series_uid,
            "type_value": instance.type_value,
            "b_value": instance.b_value,
            "metadata": metadata,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _prompt_rule(self, instance: DicomInstance) -> Optional[SequenceRule]:
        print(f"\n[interactive] Unmapped sequence: {instance.series_description} ({instance.filepath})")
        print("Existing modalities:")
        for idx, rule in enumerate(self.cfg.sequence_rules):
            print(f"[{idx}] {rule.output_modality} ({rule.name})")
        print("[s] skip this sequence")
        choice = input("Select a rule index to map or 's' to skip: ").strip()
        if choice.lower() == "s":
            return None
        try:
            idx = int(choice)
            return self.cfg.sequence_rules[idx]
        except (ValueError, IndexError):
            print("Invalid choice, skipping.")
            return None

    def _group_instances(self, instances: Sequence[DicomInstance]) -> Dict[Tuple[str, str, Optional[str]], List[DicomInstance]]:
        grouped: Dict[Tuple[str, str, Optional[str]], List[DicomInstance]] = {}
        for instance in instances:
            meta = {
                "type_value": instance.type_value or "",
                "b_value": instance.b_value or "",
            }
            # include raw DICOM tags captured during collection (e.g., 2005|1011 for mDIXON types)
            meta.update(instance.metadata)
            rule = self.detector.match(instance.series_description, meta)
            if rule is None and self.interactive:
                rule = self._prompt_rule(instance)
            if rule is None:
                self._log_unknown(patient=instance.filepath.parents[1].name, instance=instance, metadata=meta)
                continue
            key = (rule.name, instance.series_uid, instance.b_value)
            grouped.setdefault(key, []).append(instance)
        return grouped

    def _orient_image(self, image: sitk.Image, target_orientation: str) -> sitk.Image:
        try:
            return sitk.DICOMOrient(image, target_orientation)
        except Exception:
            return image

    def _write_series(
        self,
        rule: SequenceRule,
        b_value: Optional[str],
        instances: Sequence[DicomInstance],
        patient_output: Path,
        station_idx: int,
        image: Optional[sitk.Image] = None,
    ) -> Tuple[Path, sitk.Image]:
        if image is None:
            reader = sitk.ImageSeriesReader()
            sorted_files = [str(inst.filepath) for inst in sorted(instances, key=lambda x: x.instance_number)]
            reader.SetFileNames(sorted_files)
            image = reader.Execute()
            image = self._orient_image(image, self.cfg.target_orientation)
        # apply background threshold if configured for this rule
        if rule.background_threshold is not None:
            oriented_image = image
            arr = sitk.GetArrayFromImage(oriented_image).astype(np.float32)
            arr = np.where(arr < rule.background_threshold, 0.0, arr)
            image = sitk.GetImageFromArray(arr)
            image.CopyInformation(oriented_image)
        modality_name = rule.canonical_modality or rule.output_modality
        is_dwi = (rule.canonical_modality or rule.output_modality).lower() == "dwi"
        # Only DWI uses b-value as modality folder
        if is_dwi and b_value:
            try:
                modality_name = str(int(float(b_value)))
            except Exception:
                modality_name = b_value
        modality_dir = patient_output / modality_name
        modality_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{station_idx}.nii.gz"
        output_path = modality_dir / filename
        sitk.WriteImage(image, str(output_path), True)
        return output_path, image

    def sort_and_convert(self, patient_dir: Path, output_dir: Path) -> List[Path]:
        """Return list of written NIfTI files; also writes patient metadata JSON."""
        instances = self._collect_instances(patient_dir)
        grouped = self._group_instances(instances)
        written: List[Path] = []
        modality_entries: Dict[str, List[Dict]] = {}
        skipped_sequences: List[Dict] = []
        all_dicoms: set[Path] = set(p for p in patient_dir.rglob("*") if p.is_file())
        used_dicoms: set[Path] = set()

        # First pass: read and orient to gather origins for ordering
        temp_entries = []
        for (rule_name, series_uid, b_value), items in grouped.items():
            rule = next(r for r in self.cfg.sequence_rules if r.name == rule_name)
            reader = sitk.ImageSeriesReader()
            sorted_files = [str(inst.filepath) for inst in sorted(items, key=lambda x: x.instance_number)]
            reader.SetFileNames(sorted_files)
            image = self._orient_image(reader.Execute(), self.cfg.target_orientation)
            image = _crop_zero_padding_z(image)
            origin = image.GetOrigin()
            temp_entries.append(
                {
                    "rule": rule,
                    "series_uid": series_uid,
                    "b_value": b_value,
                    "instances": items,
                    "origin": origin,
                    "image": image,
                }
            )

        # Select anatomical modality and keep all DWI; skip others
        selected_anatomical: Optional[str] = None
        anatomical_entries = [e for e in temp_entries if e["rule"].is_anatomical]
        for name in ANATOMICAL_PRIORITY:
            if any((e["rule"].canonical_modality or e["rule"].output_modality) == name for e in anatomical_entries):
                selected_anatomical = name
                break
        allowed_entries = []
        for entry in temp_entries:
            canonical = entry["rule"].canonical_modality or entry["rule"].output_modality
            is_dwi = canonical.lower() == "dwi"
            if entry["rule"].is_anatomical:
                if selected_anatomical and canonical == selected_anatomical:
                    allowed_entries.append(entry)
                else:
                    skipped_sequences.append(
                        {
                            "rule": entry["rule"].name,
                            "canonical_modality": canonical,
                            "series_uid": entry["series_uid"],
                            "series_description": entry["instances"][0].series_description,
                            "protocol_name": entry["instances"][0].protocol_name,
                        }
                    )
            elif is_dwi:
                allowed_entries.append(entry)
            else:
                skipped_sequences.append(
                    {
                        "rule": entry["rule"].name,
                        "canonical_modality": canonical,
                        "series_uid": entry["series_uid"],
                        "series_description": entry["instances"][0].series_description,
                        "protocol_name": entry["instances"][0].protocol_name,
                    }
                )

        # Station ordering per canonical modality by origin z, consistent across b-values
        by_modality: Dict[str, List[Dict]] = {}
        for entry in allowed_entries:
            canonical = entry["rule"].canonical_modality or entry["rule"].output_modality
            by_modality.setdefault(canonical, []).append(entry)
        for canonical, entries in by_modality.items():
            # order stations by origin using series UID as key
            series_order: Dict[str, int] = {}
            series_origins: Dict[str, float] = {}
            for entry in entries:
                series_origins.setdefault(entry["series_uid"], entry["origin"][2])
            for idx, (series_uid, _) in enumerate(sorted(series_origins.items(), key=lambda kv: kv[1]), start=1):
                series_order[series_uid] = idx
            for entry in entries:
                rule: SequenceRule = entry["rule"]
                station_idx = series_order.get(entry["series_uid"], 0)
                target_modality = "T1" if entry["rule"].is_anatomical else canonical
                written_path, image = self._write_series(
                    rule=rule,
                    b_value=entry["b_value"],
                    instances=entry["instances"],
                    patient_output=output_dir,
                    station_idx=station_idx,
                    image=entry["image"],
                )
                written.append(written_path)
                used_dicoms.update(inst.filepath for inst in entry["instances"])
                modality_entries.setdefault(canonical, []).append(
                    {
                        "rule": rule.name,
                        "canonical_modality": target_modality if entry["rule"].is_anatomical else canonical,
                        "series_uid": entry["series_uid"],
                        "b_value": entry["b_value"],
                        "station_index": station_idx,
                        "file": str(written_path.relative_to(output_dir)),
                        "series_description": entry["instances"][0].series_description,
                        "protocol_name": entry["instances"][0].protocol_name,
                        "spacing": image.GetSpacing(),
                        "size": image.GetSize(),
                        "origin": image.GetOrigin(),
                        "direction": image.GetDirection(),
                    }
                )

        self._write_metadata(output_dir, patient_dir.name, modality_entries, skipped_sequences)
        self._apply_dwi_background_mask(output_dir)
        prune_anatomical_modalities(output_dir)
        self._report_unused(patient_dir, used_dicoms, all_dicoms)
        return written

    def _write_metadata(self, patient_output: Path, patient_id: str, modality_entries: Dict[str, List[Dict]], skipped: List[Dict]) -> None:
        meta_path = patient_output / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        anatomical_modalities = {
            entry["canonical_modality"]
            for entries in modality_entries.values()
            for entry in entries
            if any(r.name == entry["rule"] and r.is_anatomical for r in self.cfg.sequence_rules)
        }
        summary = {
            "patient_id": patient_id,
            "target_orientation": self.cfg.target_orientation,
            "anatomical_modalities": sorted(anatomical_modalities),
            "modalities": {
                modality: {
                    "stations": len(entries),
                    "b_values": sorted({e["b_value"] for e in entries if e["b_value"] is not None}),
                    "files": entries,
                }
                for modality, entries in modality_entries.items()
            },
            "skipped_sequences": skipped,
        }
        meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _report_unused(self, patient_dir: Path, used: set[Path], all_files: set[Path]) -> None:
        unused = sorted(all_files - used)
        if unused:
            print(f"[WARN] Unused DICOMs for patient {patient_dir.name}: {len(unused)} files")
        else:
            print(f"[OK] All DICOMs converted for patient {patient_dir.name}")

    def _apply_dwi_background_mask(self, output_dir: Path) -> None:
        """Use b1000-derived body mask for all DWI b-values to align backgrounds."""
        dwi_rules = [r for r in self.cfg.sequence_rules if (r.canonical_modality or r.output_modality).lower() == "dwi"]
        b_dirs = [p for p in output_dir.iterdir() if p.is_dir() and _is_float(p.name)]
        if not b_dirs:
            return
        b1000_dirs = [d for d in b_dirs if _is_b1000(d.name)]
        if not b1000_dirs:
            return
        b1000_dir = b1000_dirs[0]
        other_dirs = [d for d in b_dirs if d != b1000_dir]
        for b1000_file in sorted(b1000_dir.glob("*.nii*")):
            station = b1000_file.stem
            try:
                b1000_img = sitk.ReadImage(str(b1000_file))
                b1000_arr = sitk.GetArrayFromImage(b1000_img).astype(np.float32)
            except Exception:
                continue
            mask_img = compute_body_mask(b1000_img, smoothing_sigma=1.0, closing_radius=1, dilation_radius=1)
            mask = sitk.GetArrayFromImage(mask_img).astype(np.float32)
            masked_b1000 = b1000_arr * mask
            out_b1000 = sitk.GetImageFromArray(masked_b1000)
            out_b1000.CopyInformation(b1000_img)
            sitk.WriteImage(out_b1000, str(b1000_file), True)
            for b_dir in other_dirs:
                target = b_dir / f"{station}.nii.gz"
                if not target.exists():
                    continue
                try:
                    img = sitk.ReadImage(str(target))
                    arr = sitk.GetArrayFromImage(img).astype(np.float32)
                    if arr.shape != mask.shape:
                        continue
                    masked = arr * mask
                    out_img = sitk.GetImageFromArray(masked)
                    out_img.CopyInformation(img)
                    sitk.WriteImage(out_img, str(target), True)
                except Exception:
                    continue


def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False


def _is_b1000(val: str) -> bool:
    try:
        b = float(val)
        return abs(b - 1000.0) < 1e-3
    except ValueError:
        return False
