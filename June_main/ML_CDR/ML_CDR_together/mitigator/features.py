"""Feature construction for the single GP (Section 4).

Each training/prediction row is one (circuit, observable) pair. The GP input is
three concatenated blocks:

  [ Fourier(angles) | o_noisy | Pauli(one-hot + summaries) ]

The kernel applies different sub-kernels to different blocks, so an index map of
which columns belong to which block is REQUIRED (see :func:`feature_index_map`).
"""

from __future__ import annotations

import numpy as np

from .config import MitigatorConfig


def _pauli_label(pauli) -> str:
    """Accept a PauliObservable, a per-qubit label string, or an int row."""
    if hasattr(pauli, "label"):
        return str(pauli.label)
    if isinstance(pauli, str):
        return pauli
    int_to_char = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    return "".join(int_to_char[int(v)] for v in np.asarray(pauli).ravel().tolist())


def encode_angles(theta: np.ndarray, max_harmonic: int) -> np.ndarray:
    """Fourier features ``[cos(k*theta_j), sin(k*theta_j)]`` for k=1..max_harmonic.

    Encoding sinusoids (never raw angles) gives the GP the correct periodic
    inductive bias: an ordinary RBF/Matern kernel on these features is
    automatically periodic in theta. Length = n_params * 2 * max_harmonic.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    if max_harmonic < 1:
        raise ValueError(f"max_harmonic must be >= 1, got {max_harmonic}.")
    parts = []
    for k in range(1, int(max_harmonic) + 1):
        parts.append(np.cos(k * theta))
        parts.append(np.sin(k * theta))
    # Interleave per-angle so columns group naturally; layout is consistent as
    # long as it matches feature_index_map (the exact order does not matter to the GP).
    return np.concatenate(parts, axis=0)


def encode_pauli(pauli, n_qubits: int, onehot: bool, summaries: bool) -> np.ndarray:
    """One-hot per-qubit [I,X,Y,Z] block (4*n_qubits) and/or [weight, n_XY, n_Z]."""
    label = _pauli_label(pauli)
    if len(label) != n_qubits:
        raise ValueError(f"Pauli label length {len(label)} != n_qubits {n_qubits}: {label!r}")

    blocks = []
    if onehot:
        oh = np.zeros((n_qubits, 4), dtype=float)
        idx = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        for q, ch in enumerate(label):
            oh[q, idx[ch]] = 1.0
        blocks.append(oh.ravel())
    if summaries:
        weight = float(sum(1 for ch in label if ch != "I"))
        n_xy = float(sum(1 for ch in label if ch in ("X", "Y")))
        n_z = float(sum(1 for ch in label if ch == "Z"))
        blocks.append(np.array([weight, n_xy, n_z], dtype=float))
    if not blocks:
        return np.zeros((0,), dtype=float)
    return np.concatenate(blocks, axis=0)


def angle_block_len(config: MitigatorConfig) -> int:
    return int(config.n_params * 2 * config.max_harmonic)


def noisy_block_len(config: MitigatorConfig) -> int:
    return 1 if config.include_noisy_feature else 0


def pauli_block_len(config: MitigatorConfig) -> int:
    length = 0
    if config.pauli_onehot:
        length += 4 * config.n_qubits
    if config.pauli_summaries:
        length += 3
    return int(length)


def feature_index_map(config: MitigatorConfig) -> dict:
    """{'angle': slice, 'noisy': slice, 'pauli': slice} over the feature vector.

    The kernel uses these to apply k_lin to the o_noisy column and k_base to the
    angle + Pauli columns (Design 2 active_dims).
    """
    a = angle_block_len(config)
    n = noisy_block_len(config)
    p = pauli_block_len(config)
    return {
        "angle": slice(0, a),
        "noisy": slice(a, a + n),
        "pauli": slice(a + n, a + n + p),
    }


def feature_dim(config: MitigatorConfig) -> int:
    return angle_block_len(config) + noisy_block_len(config) + pauli_block_len(config)


def build_feature_row(row: dict, config: MitigatorConfig) -> np.ndarray:
    """Build the GP input vector for one row {theta, pauli, o_noisy, (o_ideal)}."""
    parts = [encode_angles(row["theta"], config.max_harmonic)]
    if config.include_noisy_feature:
        parts.append(np.array([float(row["o_noisy"])], dtype=float))
    parts.append(
        encode_pauli(row["pauli"], config.n_qubits, config.pauli_onehot, config.pauli_summaries)
    )
    x = np.concatenate(parts, axis=0)
    expected = feature_dim(config)
    if x.shape[0] != expected:
        raise ValueError(f"Feature row has length {x.shape[0]}, expected {expected}.")
    return x


def build_feature_matrix(rows: list[dict], config: MitigatorConfig) -> np.ndarray:
    if not rows:
        return np.zeros((0, feature_dim(config)), dtype=float)
    return np.stack([build_feature_row(r, config) for r in rows], axis=0)
