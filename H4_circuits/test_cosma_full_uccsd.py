#!/usr/bin/env python3
"""Tests for COSMA-faithful full-linked H4 UCCSD pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cosma_uccsd.emit import coupling_map, cross_rungs, score_circuit
from cosma_uccsd.majorana import (
    excitation_generator_majorana,
    jw_odd_y_string,
    majorana_to_pauli,
)
from cosma_uccsd.mapping import build_mapping, paulis_commute
from cosma_uccsd.optimize import compile_path, map_all_excitations
from cosma_uccsd.schedule import (
    expand_operator_to_factors,
    factors_all_commute,
    topological_gray_schedule,
)
from cosma_uccsd.workload import FIXED_DOUBLE_PIDS, H4_LINKED_EX_OPS, load_h4_workload


def test_workload_has_all_linked_ops():
    wl = load_h4_workload()
    assert wl.pids == FIXED_DOUBLE_PIDS
    linked = wl.linked_excitations()
    assert len(linked) == 12  # 4 singles + 4 doubles-of-links
    assert sum(len(H4_LINKED_EX_OPS[p]) for p in FIXED_DOUBLE_PIDS) == 12


def test_jw_majorana_contains_odd_y():
    basis = build_mapping("JW", 8)
    ex = (2, 6, 5, 1)
    op = majorana_to_pauli(excitation_generator_majorana(ex), basis)
    odd = jw_odd_y_string(ex, 8)
    assert odd in op
    # Sign follows correct Pauli multiplication (XY=+iZ, …); older COSMA
    # port used a flipped symplectic phase and expected -0.125j.
    assert abs(op[odd] - (0.125j)) < 1e-12
    assert len(op) == 8
    facs = expand_operator_to_factors("t", 12, 0, "t12", op)
    assert factors_all_commute(facs)


def test_schedule_respects_noncommuting_order():
    # Build two anticommuting Paulis with forced order
    from cosma_uccsd.schedule import PauliFactor

    a = PauliFactor("a", 1, 0, "XIIIIIII", 1.0, "t1", 1j)
    b = PauliFactor("b", 2, 0, "YIIIIIII", 1.0, "t2", 1j)
    assert not paulis_commute(a.pauli, b.pauli)
    # If a precedes b in input and they anticommute, schedule must keep a before b
    out = topological_gray_schedule([a, b])
    assert [f.label for f in out] == ["a", "b"]


def test_pe_and_jkmn_map_all_excitations():
    wl = load_h4_workload()
    for enc in ("JW", "PE", "JKMN"):
        basis = build_mapping(enc, 8)
        facs, notes, exact = map_all_excitations(wl, basis)
        assert len(facs) == 96
        assert exact


def test_compile_8q_jw_scores():
    wl = load_h4_workload()
    basis = build_mapping("JW", 8)
    res = compile_path(wl, basis, n_qubits=8, path_name="8q", seeds=range(2))
    assert res.score.n_qubits == 8
    assert res.score.n_factors == 96
    assert res.score.total_2q > 0
    assert res.score.cross_chip_2q >= 0
    # All routed 2q must be on the cube coupling map — checked inside score via transpile


def test_compile_6q_jw_tapers():
    wl = load_h4_workload()
    basis = build_mapping("JW", 8)
    res = compile_path(wl, basis, n_qubits=6, path_name="6q", seeds=range(2))
    assert res.score.n_qubits == 6
    assert res.score.n_factors > 0
    assert all(len(p) == 6 for p in res.to_dict()["paulis"])


def test_allocation_cc_chunk_fix_present():
    """Source-level check that the COSMA allocator chunk cap was applied."""
    path = (
        HERE.parent
        / "COSMA_Communication-aware_Optimization_of_Fermionic_Simulation_Kernels_for_Modular_Quantum_Architectures"
        / "cosma"
        / "cpp"
        / "allocation.cc"
    )
    text = path.read_text(encoding="utf-8")
    assert "kMaxChunks" in text
    assert "std::min<i32>" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
