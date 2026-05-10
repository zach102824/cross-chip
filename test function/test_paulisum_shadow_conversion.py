from __future__ import annotations

import cirq
import numpy as np
import sys
from pathlib import Path
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shot_measurement import int_observable_to_pauli_string, pauli_sum_to_int_observables


def test_conversion_roundtrip() -> None:
    qubits = [cirq.LineQubit(i) for i in range(3)]
    observable = (
        0.5 * cirq.X(qubits[0]) * cirq.Z(qubits[2])
        + (-0.75) * cirq.Y(qubits[1])
        + 1.2 * cirq.I(qubits[0])
    )

    observables_int, weights, offset = pauli_sum_to_int_observables(observable, qubits)
    assert observables_int.shape == (2, 3)
    assert_allclose(np.sort(weights), np.sort(np.array([0.5, -0.75])), atol=0.0, rtol=0.0)
    assert_allclose(offset, 1.2, atol=1e-12, rtol=0.0)

    pauli_strings = [int_observable_to_pauli_string(row) for row in observables_int]
    assert set(pauli_strings) == {"XIZ", "IYI"}


def main() -> None:
    test_conversion_roundtrip()
    print("PASS: PauliSum <-> shadowgrouping integer conversion is correct.")


if __name__ == "__main__":
    main()
