"""Tests for taper_lib on single Pauli strings and the TaperData metadata.

Formalizes the checks used while building the pipeline:

1. TaperData JSON round-trip is lossless;
2. each HF double's JW representative string tapers to ONE Pauli string with a
   +-1 sign (Clifford maps Pauli -> Pauli), and matches the known values;
3. tapering is verified operator-level: the tapered string's matrix equals the
   sector-restricted conjugated full string on the kept qubits (spot check via
   ansatz-energy tests in test_tapered_ansatz.py);
4. a Pauli that anticommutes with a symmetry generator is rejected.
"""

from __future__ import annotations

import json

import pytest

from conftest import MOLECULE

# Expected HF values (active space (6,4), doubles from UCCSD_Mole/HF.ipynb).
HF_DOUBLES = [(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)]
HF_EXPECTED_FULL_STRINGS = ["YZZXXZZX", "IYZXIXZX", "IIYXIIXX"]
HF_EXPECTED_TAPERED = [("YZZXZZ", 1), ("IYZIXZ", 1), ("IIYIIX", 1)]


def test_taper_data_json_round_trip(taper_lib, taper):
    clone = taper_lib.TaperData.from_dict(json.loads(json.dumps(taper.to_dict())))
    assert clone == taper


def test_taper_data_basic_structure(taper):
    assert taper.n_qubits_tapered == taper.n_qubits_full - 2
    assert len(taper.removed_qubits) == 2
    assert all(v in (-1, 1) for v in taper.tapering_values)
    assert len(taper.kept_qubits) == taper.n_qubits_tapered
    # Removed qubits are the last orbital of each spin block and unoccupied in HF.
    for q in taper.removed_qubits:
        assert taper.hf_bitstring_full[q] == "0"
    # Electron count is preserved on the kept register.
    assert taper.hf_bitstring_tapered.count("1") == taper.hf_bitstring_full.count("1")


@pytest.mark.skipif(MOLECULE != "HF", reason="expected values are for HF")
def test_hf_doubles_taper_to_expected_single_paulis(taper_lib, taper, gen):
    for double, expected_full, expected_tapered in zip(
        HF_DOUBLES, HF_EXPECTED_FULL_STRINGS, HF_EXPECTED_TAPERED
    ):
        full_string = "".join(gen.jw_string_for_double(taper.n_qubits_full, double))
        assert full_string == expected_full
        tapered_string, sign = taper_lib.taper_pauli_string(full_string, taper)
        assert (tapered_string, sign) == expected_tapered
        assert len(tapered_string) == taper.n_qubits_tapered


def test_tapered_string_support_keeps_shared_bridge(taper_lib, taper, gen):
    """All tapered doubles must share one vertical bridge pair (q, q + n/2):
    that is what guarantees ONE long-range RZX per double on the SAME pair."""
    half = taper.n_qubits_tapered // 2
    supports = []
    for double in HF_DOUBLES:
        full_string = "".join(gen.jw_string_for_double(taper.n_qubits_full, double))
        tapered_string, _sign = taper_lib.taper_pauli_string(full_string, taper)
        supports.append({q for q, p in enumerate(tapered_string) if p != "I"})
    common = set.intersection(*supports)
    bridge_candidates = [q for q in sorted(common) if q < half and (q + half) in common]
    assert bridge_candidates, f"no shared vertical bridge; common support = {sorted(common)}"


def test_non_symmetry_pauli_is_rejected(taper_lib, taper):
    """A single X anticommutes with the alpha-block Z parity -> not taperable."""
    bad = "X" + "I" * (taper.n_qubits_full - 1)
    with pytest.raises(ValueError):
        taper_lib.taper_pauli_string(bad, taper)
