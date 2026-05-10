from __future__ import annotations

from pathlib import Path
import sys

import cirq
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_cursor_lib import load_observable_h, ordered_parameter_symbols, prepare_decomposed_ansatz_cirq, trace_energy
from shot_measurement import estimate_energy_from_noisy_rho_shots


VQE_PARAMETERS_56 = np.array(
    [
        0.62769563,
        0.72445584,
        0.88167509,
        1.21377448,
        1.40410528,
        1.54272135,
        1.65876094,
        0.68703915,
        0.53914478,
        0.65488307,
        0.38242015,
        1.21990629,
        -0.37586606,
        -0.55407695,
        1.14433297,
        0.93390869,
        1.06005155,
        1.13875818,
        0.31887654,
        1.95665943,
        0.92840525,
        0.96902297,
        0.62448309,
        0.85426258,
        3.07549546,
        1.06236478,
        1.83665633,
        -0.07343972,
        1.05494565,
        0.51913651,
        0.8990249,
        -0.7786491,
        0.20801007,
        1.33467266,
        1.50009365,
        0.75667649,
        1.43658698,
        1.15788765,
        0.93250988,
        0.21547095,
        1.14415573,
        -0.34546883,
        1.06523003,
        1.45429555,
        0.63474669,
        1.32076148,
        0.51153084,
        1.32344213,
        0.64436686,
        1.11864704,
        0.46516075,
        1.52517135,
        1.25912853,
        0.6848312,
        -0.26316329,
        -0.07124165,
    ]
)


def test_ogm_direct_and_trace_agree_noiseless_h4() -> None:
    workspace = Path(__file__).resolve().parents[1]
    shadowgrouping_root = Path("/Users/zacharyhe/shadowgrouping")
    ogm_file = shadowgrouping_root / "haozhaowu/H4/hamil_class/ogm_outputs/OGM_H4_bond_2.0.txt"
    if not shadowgrouping_root.exists() or not ogm_file.exists():
        raise FileNotFoundError(
            "Missing shadowgrouping OGM resources. Expected "
            f"{shadowgrouping_root} and {ogm_file}."
        )

    num_spatial = 4
    ansatz_layers = 8
    num_shots = 8
    sampling_seed = 1234

    circuit, qubits = prepare_decomposed_ansatz_cirq(num_spatial, ansatz_layers)
    symbols = ordered_parameter_symbols(num_spatial, ansatz_layers)
    resolver = cirq.ParamResolver(dict(zip(symbols, VQE_PARAMETERS_56)))

    state = cirq.Simulator(seed=123).simulate(
        cirq.resolve_parameters(circuit, resolver), qubit_order=qubits
    ).final_state_vector
    rho = np.outer(state, state.conj())

    observable_h_full = load_observable_h(workspace, qubits, h_atom=4, bond_length=2.0)
    # Use a tiny H4-derived term subset so this correctness test stays fast.
    h_terms = list(observable_h_full)[:2]
    observable_h = cirq.PauliSum.from_pauli_strings(h_terms)
    hmat = observable_h.matrix(qubits=qubits)
    exact_trace = trace_energy(hmat, rho)

    ogm = estimate_energy_from_noisy_rho_shots(
        rho,
        observable_h,
        qubits,
        num_shots=num_shots,
        measurement_scheme="ogm",
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=sampling_seed,
        ogm_file=ogm_file,
        shadowgrouping_root=shadowgrouping_root,
    )
    direct = estimate_energy_from_noisy_rho_shots(
        rho,
        observable_h,
        qubits,
        num_shots=num_shots,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=sampling_seed,
    )

    ogm_energy = ogm["energy_unmitigated"]
    direct_energy = direct["energy_unmitigated"]

    # Single-seed shot estimates should both stay close to exact and to each other.
    tol = 3.5e-1
    assert abs(ogm_energy - exact_trace) <= tol
    assert abs(direct_energy - exact_trace) <= tol
    assert abs(ogm_energy - direct_energy) <= tol


def main() -> None:
    test_ogm_direct_and_trace_agree_noiseless_h4()
    print("PASS: OGM and direct agree with exact trace in noiseless H4 test.")


if __name__ == "__main__":
    main()
