from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import SimpleITK as sitk

from Preprocessing.config import PipelineConfig, SequenceRule


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
            for key in ["2005|1011", "0018|9087"]:
                if reader.HasMetaDataKey(key):
                    value = reader.GetMetaData(key)
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
            rule = self.detector.match(instance.series_description, meta)
            if rule is None and self.interactive:
                rule = self._prompt_rule(instance)
            if rule is None:
                self._log_unknown(patient=instance.filepath.parents[1].name, instance=instance, metadata=meta)
                continue
            key = (rule.name, instance.series_uid, instance.b_value)
            grouped.setdefault(key, []).append(instance)
        return grouped

    def _write_series(
        self,
        rule: SequenceRule,
        series_uid: str,
        b_value: Optional[str],
        instances: Sequence[DicomInstance],
        patient_output: Path,
        station_idx: int,
    ) -> Path:
        reader = sitk.ImageSeriesReader()
        sorted_files = [str(inst.filepath) for inst in sorted(instances, key=lambda x: x.instance_number)]
        reader.SetFileNames(sorted_files)
        image = reader.Execute()
        if rule.target_orientation:
            try:
                image = sitk.DICOMOrient(image, rule.target_orientation)
            except Exception:
                pass
        patient_output.mkdir(parents=True, exist_ok=True)
        station_label = self._station_label(station_idx)
        suffix = f"_b{b_value}" if b_value else ""
        filename = f"{rule.output_modality}_{station_label}{suffix}.nii.gz"
        output_path = patient_output / filename
        sitk.WriteImage(image, str(output_path), True)
        return output_path

    def _station_label(self, idx: int) -> str:
        labels = self.cfg.station_labels
        if idx < len(labels):
            return f"{idx+1:02d}_{labels[idx]}"
        return f"{idx+1:02d}"

    def sort_and_convert(self, patient_dir: Path, output_dir: Path) -> List[Path]:
        """Return list of written NIfTI files."""
        instances = self._collect_instances(patient_dir)
        grouped = self._group_instances(instances)
        written: List[Path] = []
        for station_idx, ((rule_name, series_uid, b_value), items) in enumerate(sorted(grouped.items(), key=lambda x: x[0][1])):
            rule = next(r for r in self.cfg.sequence_rules if r.name == rule_name)
            modality_dir = output_dir / rule.output_modality
            written_path = self._write_series(rule, series_uid, b_value, items, modality_dir, station_idx)
            written.append(written_path)
        return written

