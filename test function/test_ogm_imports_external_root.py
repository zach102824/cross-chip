from __future__ import annotations

from pathlib import Path
import sys

import cirq
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_cursor_lib import load_observable_h, prepare_decomposed_ansatz_cirq
from shot_measurement import (
    _load_shadowgrouping_scheme,
    ensure_shadowgrouping_importable,
    estimate_energy_from_noisy_rho_shots,
    pauli_sum_to_int_observables,
)


SHADOWGROUPING_ROOT = Path("/Users/zacharyhe/shadowgrouping")
OGM_FILE = SHADOWGROUPING_ROOT / "haozhaowu/H4/hamil_class/ogm_outputs/OGM_H4_bond_2.0.txt"


def _h4_observable_subset() -> tuple[cirq.PauliSum, list[cirq.Qid]]:
    workspace = Path(__file__).resolve().parents[1]
    _, qubits = prepare_decomposed_ansatz_cirq(num_spatial_orbitals=4, num_layers=1)
    observable_h = load_observable_h(workspace, qubits, h_atom=4, bond_length=2.0)
    # Keep this lightweight while still matching H4 qubit shape expected by OGM data.
    terms = list(observable_h)[:4]
    return cirq.PauliSum.from_pauli_strings(terms), qubits


def test_ensure_shadowgrouping_importable_rejects_missing_external_root(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist_shadowgrouping"
    with pytest.raises(FileNotFoundError):
        ensure_shadowgrouping_importable(missing)


@pytest.mark.skipif(not SHADOWGROUPING_ROOT.exists(), reason="shadowgrouping external root not available")
def test_ensure_shadowgrouping_importable_accepts_external_root() -> None:
    ensure_shadowgrouping_importable(SHADOWGROUPING_ROOT)
    assert str(SHADOWGROUPING_ROOT.resolve()) in sys.path


@pytest.mark.skipif(
    not SHADOWGROUPING_ROOT.exists() or not OGM_FILE.exists(),
    reason="shadowgrouping OGM resources not available",
)
def test_load_ogm_scheme_from_external_root_and_sample_settings() -> None:
    ensure_shadowgrouping_importable(SHADOWGROUPING_ROOT)
    observable_h, qubits = _h4_observable_subset()
    observables_int, weights, _ = pauli_sum_to_int_observables(observable_h, qubits)

    method = _load_shadowgrouping_scheme(
        measurement_scheme="ogm",
        observables_int=observables_int,
        weights=weights,
        epsilon=0.1,
        ogm_file=OGM_FILE,
    )
    settings, _ = method.find_setting(N_samples=16)
    settings = np.asarray(settings, dtype=int)

    assert settings.shape[0] > 0
    assert settings.shape[-1] == len(qubits)


@pytest.mark.skipif(
    not SHADOWGROUPING_ROOT.exists() or not OGM_FILE.exists(),
    reason="shadowgrouping OGM resources not available",
)
def test_estimate_energy_ogm_external_root_returns_expected_keys() -> None:
    observable_h, qubits = _h4_observable_subset()
    n = len(qubits)
    # |00...0><00...0| density matrix
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[0, 0] = 1.0

    out = estimate_energy_from_noisy_rho_shots(
        rho,
        observable_h,
        qubits,
        measurement_scheme="ogm",
        num_shots=32,
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=7,
        ogm_file=OGM_FILE,
        shadowgrouping_root=SHADOWGROUPING_ROOT,
    )
    assert set(out.keys()) == {"energy_unmitigated", "energy_rem", "offset"}
    assert np.isfinite(out["energy_unmitigated"])
    assert np.isfinite(out["energy_rem"])


@pytest.mark.skipif(not SHADOWGROUPING_ROOT.exists(), reason="shadowgrouping external root not available")
def test_estimate_energy_ogm_requires_ogm_file() -> None:
    observable_h, qubits = _h4_observable_subset()
    n = len(qubits)
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[0, 0] = 1.0

    with pytest.raises(ValueError, match="requires ogm_file"):
        estimate_energy_from_noisy_rho_shots(
            rho,
            observable_h,
            qubits,
            measurement_scheme="ogm",
            num_shots=8,
            apply_readout_noise=False,
            apply_rem=False,
            sampling_seed=11,
            ogm_file=None,
            shadowgrouping_root=SHADOWGROUPING_ROOT,
        )
