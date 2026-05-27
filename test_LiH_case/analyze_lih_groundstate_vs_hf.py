#!/usr/bin/env python3
"""Analyze LiH ground-state structure relative to an HF reference.

This script loads a saved OpenFermion qubit Hamiltonian (`*_of.pkl`), computes
its exact ground state, and reports:
1) spin-orbital occupancies,
2) spin-orbital occupancy covariance and Pearson correlation matrices,
3) differences versus a configurable Hartree-Fock (HF) reference occupation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh


def load_qubit_hamiltonian(path: Path) -> QubitOperator:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, QubitOperator):
        raise TypeError(f"Expected QubitOperator in {path}, got {type(obj).__name__}.")
    return obj


def infer_num_qubits(qubit_hamiltonian: QubitOperator) -> int:
    if not qubit_hamiltonian.terms:
        return 1
    max_index = -1
    for term in qubit_hamiltonian.terms:
        for q, _ in term:
            if q > max_index:
                max_index = q
    return max_index + 1 if max_index >= 0 else 1


def sparse_hamiltonian_and_checks(
    qubit_hamiltonian: QubitOperator, n_qubits: int, hermiticity_tol: float
) -> tuple[csc_matrix, float]:
    h_sparse = get_sparse_operator(qubit_hamiltonian, n_qubits=n_qubits).tocsc()
    anti_herm = h_sparse - h_sparse.getH()
    anti_herm_norm = float(np.linalg.norm(anti_herm.toarray(), ord="fro"))
    if anti_herm_norm > hermiticity_tol:
        raise ValueError(
            f"Hamiltonian fails Hermiticity check: ||H - H^dagger||_F = {anti_herm_norm:.3e}"
        )
    return h_sparse, anti_herm_norm


def exact_ground_state(h_sparse: csc_matrix) -> tuple[float, np.ndarray]:
    dim = h_sparse.shape[0]
    if dim <= 1024:
        dense = h_sparse.toarray()
        evals, evecs = np.linalg.eigh(dense)
        gs_energy = float(np.real_if_close(evals[0]))
        gs_vec = np.asarray(evecs[:, 0], dtype=np.complex128)
        return gs_energy, gs_vec
    eigvals, eigvecs = eigsh(h_sparse, k=1, which="SA")
    gs_energy = float(np.real_if_close(eigvals[0]))
    gs_vec = np.asarray(eigvecs[:, 0], dtype=np.complex128)
    return gs_energy, gs_vec


def number_operator(index: int) -> QubitOperator:
    return QubitOperator((), 0.5) + QubitOperator(((index, "Z"),), -0.5)


def expectation_value(state: np.ndarray, operator: csc_matrix) -> float:
    vec = operator @ state
    val = np.vdot(state, vec)
    return float(np.real_if_close(val))


def occupancy_statistics(
    gs_vec: np.ndarray, n_qubits: int, pearson_eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_ops = [get_sparse_operator(number_operator(i), n_qubits=n_qubits).tocsc() for i in range(n_qubits)]
    occ = np.array([expectation_value(gs_vec, n_ops[i]) for i in range(n_qubits)], dtype=np.float64)

    n_occ = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        n_i_psi = n_ops[i] @ gs_vec
        for j in range(i, n_qubits):
            val = float(np.real_if_close(np.vdot(gs_vec, n_ops[j] @ n_i_psi)))
            n_occ[i, j] = val
            n_occ[j, i] = val

    cov = n_occ - np.outer(occ, occ)
    var = np.clip(np.diag(cov), a_min=0.0, a_max=None)
    denom = np.sqrt(np.outer(var, var))
    pearson = np.zeros_like(cov)
    mask = denom > pearson_eps
    pearson[mask] = cov[mask] / denom[mask]
    np.fill_diagonal(pearson, np.where(var > pearson_eps, 1.0, 0.0))
    return occ, cov, pearson


def parse_occupied_indices(
    n_spatial: int, n_fill: int, provided: list[int] | None, spin_label: str
) -> list[int]:
    if provided is None:
        occ = list(range(n_fill))
    else:
        occ = sorted(provided)
        if len(occ) != n_fill:
            raise ValueError(
                f"--occupied-{spin_label} must contain exactly {n_fill} entries, got {len(occ)}."
            )
    if any(i < 0 or i >= n_spatial for i in occ):
        raise ValueError(
            f"Invalid {spin_label} occupied spatial orbital in {occ}; valid range is [0, {n_spatial - 1}]."
        )
    if len(set(occ)) != len(occ):
        raise ValueError(f"Duplicate entries in --occupied-{spin_label}: {occ}")
    return occ


def hf_occupancy_vector(
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
    occupied_alpha: list[int] | None,
    occupied_beta: list[int] | None,
) -> np.ndarray:
    occ_alpha = parse_occupied_indices(n_spatial, n_alpha, occupied_alpha, "alpha")
    occ_beta = parse_occupied_indices(n_spatial, n_beta, occupied_beta, "beta")
    occ = np.zeros(2 * n_spatial, dtype=np.float64)
    for a in occ_alpha:
        occ[a] = 1.0
    for b in occ_beta:
        occ[n_spatial + b] = 1.0
    return occ


def hf_energy_from_pauli(
    qubit_hamiltonian: QubitOperator, hf_occ: np.ndarray, n_spatial: int
) -> float:
    energy = 0.0
    z_eigs = np.ones_like(hf_occ, dtype=np.float64)
    z_eigs[hf_occ > 0.5] = -1.0
    for term, coeff in qubit_hamiltonian.terms.items():
        term_val = 1.0
        for q, p in term:
            if p == "Z":
                term_val *= z_eigs[q]
            elif p in ("X", "Y"):
                term_val = 0.0
                break
            else:
                raise ValueError(f"Unsupported Pauli label '{p}' in term {term}.")
        energy += float(np.real_if_close(coeff)) * term_val
    return energy


def top_pair_entries(matrix: np.ndarray, topk: int) -> list[tuple[int, int, float]]:
    n = matrix.shape[0]
    entries: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            entries.append((i, j, float(matrix[i, j])))
    entries.sort(key=lambda x: abs(x[2]), reverse=True)
    return entries[:topk]


def spin_channel(i: int, j: int, n_spatial: int) -> str:
    i_alpha = i < n_spatial
    j_alpha = j < n_spatial
    if i_alpha and j_alpha:
        return "alpha-alpha"
    if (not i_alpha) and (not j_alpha):
        return "beta-beta"
    return "alpha-beta"


def channel_abs_summary(delta_matrix: np.ndarray, n_spatial: int) -> dict[str, float]:
    bins: dict[str, list[float]] = {"alpha-alpha": [], "beta-beta": [], "alpha-beta": []}
    n = delta_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            bins[spin_channel(i, j, n_spatial)].append(abs(float(delta_matrix[i, j])))
    out: dict[str, float] = {}
    for k, vals in bins.items():
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


def matrix_str(name: str, value: np.ndarray) -> str:
    arr = np.array2string(value, precision=6, floatmode="fixed", suppress_small=False)
    return f"{name}:\n{arr}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamiltonian-pkl", type=Path, required=True, help="Path to saved *_of.pkl Hamiltonian.")
    parser.add_argument("--n-spatial", type=int, default=4, help="Number of active spatial orbitals.")
    parser.add_argument("--n-alpha", type=int, default=1, help="Number of alpha electrons in HF reference.")
    parser.add_argument("--n-beta", type=int, default=1, help="Number of beta electrons in HF reference.")
    parser.add_argument(
        "--occupied-alpha",
        type=int,
        nargs="*",
        default=None,
        help="Explicit occupied alpha spatial orbital indices (overrides default lowest-energy filling).",
    )
    parser.add_argument(
        "--occupied-beta",
        type=int,
        nargs="*",
        default=None,
        help="Explicit occupied beta spatial orbital indices (overrides default lowest-energy filling).",
    )
    parser.add_argument("--topk", type=int, default=10, help="Show top-K largest pairwise differences.")
    parser.add_argument("--pearson-eps", type=float, default=1e-12, help="Small denominator threshold for Pearson.")
    parser.add_argument("--hermiticity-tol", type=float, default=1e-10, help="Hermiticity tolerance.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    h_path = args.hamiltonian_pkl.expanduser().resolve()
    qubit_h = load_qubit_hamiltonian(h_path)
    n_qubits = infer_num_qubits(qubit_h)
    if n_qubits != 2 * args.n_spatial:
        raise ValueError(
            f"Qubit count mismatch: inferred {n_qubits} from Hamiltonian, "
            f"but --n-spatial={args.n_spatial} implies {2 * args.n_spatial}."
        )
    if args.n_alpha < 0 or args.n_alpha > args.n_spatial:
        raise ValueError(f"--n-alpha must be in [0, {args.n_spatial}].")
    if args.n_beta < 0 or args.n_beta > args.n_spatial:
        raise ValueError(f"--n-beta must be in [0, {args.n_spatial}].")

    h_sparse, anti_herm_norm = sparse_hamiltonian_and_checks(
        qubit_h, n_qubits=n_qubits, hermiticity_tol=args.hermiticity_tol
    )
    gs_energy, gs_vec = exact_ground_state(h_sparse)
    gs_norm = float(np.linalg.norm(gs_vec))
    if not np.isclose(gs_norm, 1.0, atol=1e-10):
        raise ValueError(f"Ground-state vector is not normalized: ||psi|| = {gs_norm:.12f}")

    gs_occ, gs_cov, gs_pearson = occupancy_statistics(gs_vec, n_qubits=n_qubits, pearson_eps=args.pearson_eps)

    hf_occ = hf_occupancy_vector(
        n_spatial=args.n_spatial,
        n_alpha=args.n_alpha,
        n_beta=args.n_beta,
        occupied_alpha=args.occupied_alpha,
        occupied_beta=args.occupied_beta,
    )
    hf_cov = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    hf_pearson = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    hf_energy = hf_energy_from_pauli(qubit_h, hf_occ, n_spatial=args.n_spatial)

    delta_occ = gs_occ - hf_occ
    delta_cov = gs_cov - hf_cov
    delta_pearson = gs_pearson - hf_pearson
    top_cov = top_pair_entries(delta_cov, topk=args.topk)
    top_pearson = top_pair_entries(delta_pearson, topk=args.topk)
    cov_channel = channel_abs_summary(delta_cov, n_spatial=args.n_spatial)
    pearson_channel = channel_abs_summary(delta_pearson, n_spatial=args.n_spatial)

    print("=== LiH Ground-State vs HF Analysis ===")
    print(f"Hamiltonian: {h_path}")
    print(f"Num qubits: {n_qubits}")
    print(f"Num Pauli terms: {len(qubit_h.terms)}")
    print()
    print("--- Energies (Hartree) ---")
    print(f"Exact ground-state energy: {gs_energy:.12f}")
    print(f"HF product-state energy:   {hf_energy:.12f}")
    print(f"Correlation energy (GS-HF): {gs_energy - hf_energy:.12f}")
    print()
    print("--- Per spin-orbital occupancy ---")
    print("index spin  spatial  GS_occ        HF_occ        delta")
    for i in range(n_qubits):
        spin = "a" if i < args.n_spatial else "b"
        spatial = i if i < args.n_spatial else i - args.n_spatial
        print(
            f"{i:>3d}   {spin}      {spatial:>2d}   "
            f"{gs_occ[i]: .10f}  {hf_occ[i]: .10f}  {delta_occ[i]: .10f}"
        )
    print()
    print(matrix_str("GS occupancy covariance matrix", gs_cov))
    print()
    print(matrix_str("GS occupancy Pearson matrix", gs_pearson))
    print()
    print("--- Top |delta covariance| pairs (GS - HF) ---")
    for i, j, val in top_cov:
        print(f"({i}, {j}) [{spin_channel(i, j, args.n_spatial)}] : {val:+.10f}")
    print()
    print("--- Top |delta Pearson| pairs (GS - HF) ---")
    for i, j, val in top_pearson:
        print(f"({i}, {j}) [{spin_channel(i, j, args.n_spatial)}] : {val:+.10f}")
    print()
    print("--- Mean |delta| by spin channel ---")
    print(
        "covariance  : "
        f"aa={cov_channel['alpha-alpha']:.10f}, "
        f"bb={cov_channel['beta-beta']:.10f}, "
        f"ab={cov_channel['alpha-beta']:.10f}"
    )
    print(
        "pearson     : "
        f"aa={pearson_channel['alpha-alpha']:.10f}, "
        f"bb={pearson_channel['beta-beta']:.10f}, "
        f"ab={pearson_channel['alpha-beta']:.10f}"
    )
    print()
    print("--- Validation ---")
    print(f"Hermiticity ||H-H^dagger||_F: {anti_herm_norm:.3e}")
    print(f"Ground-state norm:            {gs_norm:.12f}")

    if args.json_out is not None:
        json_path = args.json_out.expanduser().resolve()
        payload: dict[str, Any] = {
            "input": {
                "hamiltonian_pkl": str(h_path),
                "n_spatial": int(args.n_spatial),
                "n_alpha": int(args.n_alpha),
                "n_beta": int(args.n_beta),
                "occupied_alpha": None if args.occupied_alpha is None else [int(x) for x in args.occupied_alpha],
                "occupied_beta": None if args.occupied_beta is None else [int(x) for x in args.occupied_beta],
                "topk": int(args.topk),
            },
            "energies_hartree": {
                "ground_state": float(gs_energy),
                "hf_product_state": float(hf_energy),
                "correlation_energy_gs_minus_hf": float(gs_energy - hf_energy),
            },
            "occupancy": {
                "ground_state": gs_occ.tolist(),
                "hf": hf_occ.tolist(),
                "delta": delta_occ.tolist(),
            },
            "covariance": {
                "ground_state": gs_cov.tolist(),
                "hf": hf_cov.tolist(),
                "delta": delta_cov.tolist(),
                "mean_abs_delta_by_spin_channel": cov_channel,
            },
            "pearson": {
                "ground_state": gs_pearson.tolist(),
                "hf": hf_pearson.tolist(),
                "delta": delta_pearson.tolist(),
                "mean_abs_delta_by_spin_channel": pearson_channel,
            },
            "top_pairs": {
                "delta_covariance": [
                    {"i": int(i), "j": int(j), "spin_channel": spin_channel(i, j, args.n_spatial), "value": float(v)}
                    for i, j, v in top_cov
                ],
                "delta_pearson": [
                    {"i": int(i), "j": int(j), "spin_channel": spin_channel(i, j, args.n_spatial), "value": float(v)}
                    for i, j, v in top_pearson
                ],
            },
            "validation": {
                "hermiticity_fro_norm": float(anti_herm_norm),
                "ground_state_norm": float(gs_norm),
            },
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON report: {json_path}")


if __name__ == "__main__":
    main()
