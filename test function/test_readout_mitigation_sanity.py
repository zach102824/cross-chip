from __future__ import annotations

import numpy as np
import sys
from pathlib import Path
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shot_measurement import rem_z_vectors


def test_rem_unbiased_single_qubit_z() -> None:
    rng = np.random.default_rng(7)
    num_shots = 500_000

    # True Z expectation +0.4 means p(0)=0.7, p(1)=0.3
    p0_true = 0.7
    ideal_bits = (rng.random(num_shots) >= p0_true).astype(int)[:, None]

    p_0_success = np.array([0.88], dtype=float)
    p_1_success = np.array([0.93], dtype=float)
    p_flip_0_to_1 = 1.0 - p_0_success[0]
    p_flip_1_to_0 = 1.0 - p_1_success[0]

    noisy_bits = ideal_bits.copy()
    rands = rng.random(num_shots)
    flip01 = (ideal_bits[:, 0] == 0) & (rands < p_flip_0_to_1)
    flip10 = (ideal_bits[:, 0] == 1) & (rands < p_flip_1_to_0)
    noisy_bits[flip01, 0] = 1
    noisy_bits[flip10, 0] = 0

    rem = rem_z_vectors(p_0_success, p_1_success)
    mitigated_vals = np.where(noisy_bits[:, 0] == 0, rem[0, 0], rem[0, 1])
    mitigated_est = float(np.mean(mitigated_vals))

    exact_z = 2.0 * p0_true - 1.0
    assert_allclose(mitigated_est, exact_z, atol=3e-3, rtol=0.0)


def main() -> None:
    test_rem_unbiased_single_qubit_z()
    print("PASS: REM single-qubit sanity check passed.")


if __name__ == "__main__":
    main()
