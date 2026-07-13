"""Full-register vs tapered ansatz energy equivalence.

The pre-taper and post-taper single-string doubles ansatz must give the SAME
energy for the SAME parameters (tapering is a Clifford change of frame, not an
approximation):

    <HF| prod exp(+i t_k/2 P_k) H prod exp(-i t_k/2 P_k) |HF>
  == <HF~| prod exp(+i t_k/2 P~_k) H~ prod exp(-i t_k/2 P~_k) |HF~>

checked at random parameter values and at the independently optimized minimum.
"""

from __future__ import annotations

import numpy as np
import pytest
from openfermion.linalg import get_sparse_operator
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

from conftest import TEST_BONDS, hf_state_vector

HF_DOUBLES = [(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)]
POINTWISE_TOL = 1e-10  # Ha, random-theta energy match
OPTIMUM_TOL = 1e-6     # Ha, independently optimized minima match


def _ansatz_energy(h_sparse, reference, strings, signs, n_qubits, thetas, taper_lib):
    vec = reference.copy()
    for theta, sign, string in zip(thetas, signs, strings):
        pauli = get_sparse_operator(
            taper_lib.pauli_string_to_qubit_operator(string), n_qubits=n_qubits
        )
        vec = expm_multiply(-0.5j * float(theta) * float(sign) * pauli, vec)
    return float(np.real(np.vdot(vec, h_sparse @ vec)))


@pytest.fixture(scope="module")
def ansatz_setup(taper_lib, taper, gen, hamiltonians):
    full_strings = [
        "".join(gen.jw_string_for_double(taper.n_qubits_full, d)) for d in HF_DOUBLES
    ]
    tapered = [taper_lib.taper_pauli_string(s, taper) for s in full_strings]
    occ_full = [q for q, b in enumerate(taper.hf_bitstring_full) if b == "1"]
    occ_tap = [q for q, b in enumerate(taper.hf_bitstring_tapered) if b == "1"]
    return {
        "full_strings": full_strings,
        "full_signs": [1] * len(full_strings),
        "tap_strings": [s for s, _ in tapered],
        "tap_signs": [sg for _, sg in tapered],
        "psi_full": hf_state_vector(taper.n_qubits_full, occ_full),
        "psi_tap": hf_state_vector(taper.n_qubits_tapered, occ_tap),
    }


@pytest.mark.parametrize("bond", TEST_BONDS)
def test_energy_matches_at_random_parameters(
    taper_lib, taper, hamiltonians, ansatz_setup, bond
):
    full_op, tapered_op = hamiltonians[bond]
    h_full = get_sparse_operator(full_op, n_qubits=taper.n_qubits_full)
    h_tap = get_sparse_operator(tapered_op, n_qubits=taper.n_qubits_tapered)

    rng = np.random.default_rng(1234)
    for _trial in range(5):
        thetas = rng.uniform(-0.8, 0.8, size=len(HF_DOUBLES))
        e_full = _ansatz_energy(
            h_full, ansatz_setup["psi_full"], ansatz_setup["full_strings"],
            ansatz_setup["full_signs"], taper.n_qubits_full, thetas, taper_lib,
        )
        e_tap = _ansatz_energy(
            h_tap, ansatz_setup["psi_tap"], ansatz_setup["tap_strings"],
            ansatz_setup["tap_signs"], taper.n_qubits_tapered, thetas, taper_lib,
        )
        assert abs(e_full - e_tap) < POINTWISE_TOL, (
            f"bond {bond}, thetas {thetas}: {e_full!r} != {e_tap!r}"
        )


@pytest.mark.parametrize("bond", TEST_BONDS)
def test_optimized_minima_match(taper_lib, taper, hamiltonians, ansatz_setup, bond):
    full_op, tapered_op = hamiltonians[bond]
    h_full = get_sparse_operator(full_op, n_qubits=taper.n_qubits_full)
    h_tap = get_sparse_operator(tapered_op, n_qubits=taper.n_qubits_tapered)

    def objective(h, psi, strings, signs, nq):
        return lambda x: _ansatz_energy(h, psi, strings, signs, nq, x, taper_lib)

    x0 = np.zeros(len(HF_DOUBLES))
    res_full = minimize(
        objective(h_full, ansatz_setup["psi_full"], ansatz_setup["full_strings"],
                  ansatz_setup["full_signs"], taper.n_qubits_full),
        x0, method="BFGS", options={"gtol": 1e-9},
    )
    res_tap = minimize(
        objective(h_tap, ansatz_setup["psi_tap"], ansatz_setup["tap_strings"],
                  ansatz_setup["tap_signs"], taper.n_qubits_tapered),
        x0, method="BFGS", options={"gtol": 1e-9},
    )
    assert abs(res_full.fun - res_tap.fun) < OPTIMUM_TOL, (
        f"bond {bond}: optimized full {res_full.fun:.10f} != tapered {res_tap.fun:.10f}"
    )
