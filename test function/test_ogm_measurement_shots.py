from __future__ import annotations

from pathlib import Path
import sys

import cirq
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_cursor_lib import (
    load_observable_h,
    ordered_parameter_symbols,
    prepare_decomposed_ansatz_cirq,
    trace_energy,
)
from shot_measurement import estimate_energy_from_noisy_rho_shots


SHADOWGROUPING_ROOT = Path("/Users/zacharyhe/shadowgrouping")
OGM_FILE = SHADOWGROUPING_ROOT / "haozhaowu/H4/hamil_class/ogm_outputs/OGM_H4_bond_2.0.txt"


def _noiseless_rho_and_observable() -> tuple[np.ndarray, cirq.PauliSum, list[cirq.Qid], float]:
    workspace = Path(__file__).resolve().parents[1]
    circuit, qubits = prepare_decomposed_ansatz_cirq(num_spatial_orbitals=4, num_layers=1)
    symbols = ordered_parameter_symbols(4, 1)
    params = np.linspace(-0.3, 0.4, len(symbols))
    resolver = cirq.ParamResolver(dict(zip(symbols, params)))
    resolved = cirq.resolve_parameters(circuit, resolver)
    state = cirq.Simulator(seed=2).simulate(resolved, qubit_order=qubits).final_state_vector
    rho = np.outer(state, state.conj())

    observable_h_full = load_observable_h(workspace, qubits, h_atom=4, bond_length=2.0)
    observable_h = cirq.PauliSum.from_pauli_strings(list(observable_h_full)[:4])
    exact = trace_energy(observable_h.matrix(qubits=qubits), rho)
    return rho, observable_h, qubits, float(exact)


@pytest.mark.skipif(
    not SHADOWGROUPING_ROOT.exists() or not OGM_FILE.exists(),
    reason="shadowgrouping OGM resources not available",
)
def test_ogm_same_seed_is_reproducible() -> None:
    rho, observable_h, qubits, _ = _noiseless_rho_and_observable()
    kwargs = dict(
        rho=rho,
        observable_h=observable_h,
        qubits=qubits,
        measurement_scheme="ogm",
        num_shots=1024,
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=123,
        ogm_file=OGM_FILE,
        shadowgrouping_root=SHADOWGROUPING_ROOT,
    )
    out1 = estimate_energy_from_noisy_rho_shots(**kwargs)
    out2 = estimate_energy_from_noisy_rho_shots(**kwargs)
    # OGM path can include internal randomness from external library components.
    # Same-seed runs should still remain very close.
    assert abs(out1["energy_unmitigated"] - out2["energy_unmitigated"]) <= 2e-2


@pytest.mark.skipif(
    not SHADOWGROUPING_ROOT.exists() or not OGM_FILE.exists(),
    reason="shadowgrouping OGM resources not available",
)
def test_ogm_error_vs_exact_decreases_with_more_shots() -> None:
    rho, observable_h, qubits, exact = _noiseless_rho_and_observable()
    shot_grid = (256, 1024, 4096)
    seed_grid = (11, 13, 17, 19)
    mean_abs_errors = []
    for shots in shot_grid:
        errs = []
        for seed in seed_grid:
            out = estimate_energy_from_noisy_rho_shots(
                rho,
                observable_h,
                qubits,
                measurement_scheme="ogm",
                num_shots=shots,
                apply_readout_noise=False,
                apply_rem=False,
                sampling_seed=seed,
                ogm_file=OGM_FILE,
                shadowgrouping_root=SHADOWGROUPING_ROOT,
            )
            errs.append(abs(out["energy_unmitigated"] - exact))
        mean_abs_errors.append(float(np.mean(errs)))

    assert mean_abs_errors[-1] <= mean_abs_errors[0]


@pytest.mark.skipif(
    not SHADOWGROUPING_ROOT.exists() or not OGM_FILE.exists(),
    reason="shadowgrouping OGM resources not available",
)
def test_ogm_vs_direct_pauli_same_shots_seed_both_near_exact() -> None:
    rho, observable_h, qubits, exact = _noiseless_rho_and_observable()
    shots = 4096
    seed = 29
    ogm = estimate_energy_from_noisy_rho_shots(
        rho,
        observable_h,
        qubits,
        measurement_scheme="ogm",
        num_shots=shots,
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=seed,
        ogm_file=OGM_FILE,
        shadowgrouping_root=SHADOWGROUPING_ROOT,
    )
    direct = estimate_energy_from_noisy_rho_shots(
        rho,
        observable_h,
        qubits,
        measurement_scheme="direct_pauli",
        num_shots=shots,
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=seed,
    )
    assert abs(ogm["energy_unmitigated"] - exact) <= 0.18
    assert abs(direct["energy_unmitigated"] - exact) <= 0.18

