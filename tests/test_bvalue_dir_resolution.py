from __future__ import annotations

from pathlib import Path

from Preprocessing.register_F2A import _resolve_bvalue_dir as resolve_bvalue_dir_f2a
from Preprocessing.register_S2S import _resolve_bvalue_dir as resolve_bvalue_dir_s2s


def test_resolve_bvalue_dir_accepts_integer_like_float(tmp_path: Path) -> None:
    patient = tmp_path / "P001"
    (patient / "50").mkdir(parents=True)
    (patient / "1000").mkdir(parents=True)

    assert resolve_bvalue_dir_f2a(patient, "1000.0") == patient / "1000"
    assert resolve_bvalue_dir_s2s(patient, "1000.0") == patient / "1000"

