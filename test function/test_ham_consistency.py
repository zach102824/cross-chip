from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import eigsh


PAULI_INT_TO_CHAR = {0: "I", 1: "X", 2: "Y", 3: "Z"}
VALID_PAULI_CHARS = {"I", "X", "Y", "Z"}


def load_pickle_operator(path: Path) -> QubitOperator:
    with path.open("rb") as f:
        op = pickle.load(f)
    if not isinstance(op, QubitOperator):
        raise TypeError(f"Expected QubitOperator in {path}, got {type(op)!r}")
    return op


def load_number_convention(path: Path) -> QubitOperator:
    op = QubitOperator()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid number-convention line {line_no} in {path}")

            coeff = float(parts[0])
            pauli_ints = [int(x) for x in parts[1:]]
            terms = []
            for qubit_idx, pauli_int in enumerate(pauli_ints):
                if pauli_int not in PAULI_INT_TO_CHAR:
                    raise ValueError(
                        f"Unknown Pauli int {pauli_int} on line {line_no} in {path}"
                    )
                pauli_char = PAULI_INT_TO_CHAR[pauli_int]
                if pauli_char != "I":
                    terms.append((qubit_idx, pauli_char))
            op += QubitOperator(tuple(terms), complex(coeff))
    return op


def load_pauli_convention(path: Path) -> QubitOperator:
    op = QubitOperator()
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 2 != 0:
        raise ValueError(f"Expected even number of non-empty lines in {path}")

    for idx in range(0, len(lines), 2):
        pauli_str = lines[idx]
        coeff_raw = lines[idx + 1]
        if any(ch not in VALID_PAULI_CHARS for ch in pauli_str):
            raise ValueError(f"Invalid pauli string '{pauli_str}' in {path}")

        coeff = complex(coeff_raw)
        terms = []
        for qubit_idx, pauli_char in enumerate(pauli_str):
            if pauli_char != "I":
                terms.append((qubit_idx, pauli_char))
        op += QubitOperator(tuple(terms), coeff)
    return op


def lowest_eigs(op: QubitOperator, k: int = 5) -> np.ndarray:
    sparse = get_sparse_operator(op).tocsc()
    dim = sparse.shape[0]
    if dim == 0:
        raise ValueError("Empty Hamiltonian matrix.")

    k_eff = min(k, dim)
    if dim <= 2 or k_eff == dim:
        dense = sparse.toarray()
        eigvals = np.linalg.eigvalsh(dense)
        return np.sort(np.real_if_close(eigvals))[:k_eff]

    eigvals, _ = eigsh(sparse, k=k_eff, which="SA")
    return np.sort(np.real_if_close(eigvals))


def compare_eigenvalues(
    pkl_op: QubitOperator,
    num_op: QubitOperator,
    pauli_op: QubitOperator,
    k: int = 5,
    atol: float = 1e-5,
) -> dict[str, np.ndarray]:
    eig_pkl = lowest_eigs(pkl_op, k=k)
    eig_num = lowest_eigs(num_op, k=k)
    eig_pauli = lowest_eigs(pauli_op, k=k)

    assert_allclose(eig_pkl, eig_num, atol=atol, rtol=0.0)
    assert_allclose(eig_pkl, eig_pauli, atol=atol, rtol=0.0)
    assert_allclose(eig_num, eig_pauli, atol=atol, rtol=0.0)

    return {"pkl": eig_pkl, "num": eig_num, "pauli": eig_pauli}


def ham_eigenvalue_consistency(
    pkl_path: Path,
    number_path: Path,
    pauli_path: Path,
    k: int = 5,
    atol: float = 1e-5,
) -> dict[str, np.ndarray]:
    pkl_op = load_pickle_operator(pkl_path)
    num_op = load_number_convention(number_path)
    pauli_op = load_pauli_convention(pauli_path)
    return compare_eigenvalues(pkl_op, num_op, pauli_op, k=k, atol=atol)


def test_ham_eigenvalue_consistency_default_paths() -> None:
    pkl_path, number_path, pauli_path = default_paths()
    ham_eigenvalue_consistency(
        pkl_path=pkl_path,
        number_path=number_path,
        pauli_path=pauli_path,
        k=5,
        atol=1e-5,
    )


def default_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    ham_dir = root / "Pauli_Ham"
    return (
        ham_dir / "H4_bond_2.pkl",
        ham_dir / "H4_bond_2_number_convention.txt",
        ham_dir / "H4_bond_2_pauli_convention.txt",
    )


def main() -> None:
    pkl_default, num_default, pauli_default = default_paths()
    parser = argparse.ArgumentParser(
        description="Check eigenvalue consistency across Hamiltonian file formats."
    )
    parser.add_argument("--pkl", type=Path, default=pkl_default, help="Path to .pkl file")
    parser.add_argument(
        "--number",
        type=Path,
        default=num_default,
        help="Path to number-convention .txt file",
    )
    parser.add_argument(
        "--pauli",
        type=Path,
        default=pauli_default,
        help="Path to pauli-convention .txt file",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of smallest eigenvalues")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    args = parser.parse_args()

    eigs = ham_eigenvalue_consistency(
        pkl_path=args.pkl,
        number_path=args.number,
        pauli_path=args.pauli,
        k=args.k,
        atol=args.atol,
    )
    print("PASS: Eigenvalues match across pickle, number, and pauli formats.")
    print(f"Lowest {len(eigs['pkl'])} eigenvalues:")
    print(eigs["pkl"])


if __name__ == "__main__":
    main()
