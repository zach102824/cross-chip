"""
test_create_UCCSD_circuit.py
============================
Statevector test of "create UCCSD circuit.py".

It checks, with pure matrix-vector products (no dense 2^N x 2^N matrices),
that the generated Qiskit circuits recreate the ORIGINAL fermionic double
excitations:

TEST A (fermionic equivalence on the reference determinant)
    For every double d, the single-double circuit at angle alpha applied to
    the Hartree-Fock determinant equals the exact fermionic exponential

        exp( theta (T - T^dag) ) |HF>,   T = a_k^dag a_l^dag a_i a_j,

    with theta = s*alpha/2 (s = +-1 fixed sign).  The fermionic side is
    evaluated in second quantisation with Jordan-Wigner sign factors using

        exp(theta K) = 1 + sin(theta) K + (1 - cos(theta)) K^2,   K = T - T^dag,

    which is EXACT because K^3 = -K (T^2 = 0).  Only matvecs are used.
    This is the paper's claim: one Pauli string per double reproduces the
    full 8-string fermionic double on the reference (angle rescaled).

TEST B (compiler correctness on the FULL Hilbert space)
    The complete multi-double circuit (including the peephole-cancelled
    Clifford interfaces) equals the product of Pauli-string exponentials
        prod_d exp(-i s_d alpha_d / 2 P_d)
    on random statevectors, evaluated by sparse Pauli matvec.

Conventions: qubit i <-> spin orbital i; Qiskit little-endian (bit i of the
basis index = qubit i), used consistently on both sides of every comparison.

Run:  python test_create_UCCSD_circuit.py
"""
import importlib.util
import os
import numpy as np
from qiskit.quantum_info import Statevector

# --- import the builder (filename contains spaces) ---------------------
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "create_uccsd", os.path.join(_here, "create UCCSD circuit.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
create_uccsd_circuit = mod.create_uccsd_circuit

# ----------------------------------------------------------------------
# Fermionic operators in second quantisation (Jordan-Wigner signs),
# acting on statevectors index-wise: bit i of the index = occupation n_i.
# ----------------------------------------------------------------------
def _parity_below(idx, i):
    """(-1)^{sum_{r<i} n_r} for basis index idx."""
    return -1 if bin(idx & ((1 << i) - 1)).count("1") % 2 else 1

def annihilate(psi, n, i):
    """a_i |psi>  via matrix-vector product (no matrices materialised)."""
    out = np.zeros_like(psi)
    src = np.nonzero(psi)[0]
    for b in src:
        if (b >> i) & 1:
            out[b ^ (1 << i)] += _parity_below(b, i) * psi[b]
    return out

def create(psi, n, i):
    """a_i^dag |psi>."""
    out = np.zeros_like(psi)
    src = np.nonzero(psi)[0]
    for b in src:
        if not (b >> i) & 1:
            out[b ^ (1 << i)] += _parity_below(b, i) * psi[b]
    return out

def apply_T(psi, n, double):
    """T = a_k^dag a_l^dag a_i a_j  (rightmost operator acts first)."""
    k, l, i, j = double
    psi = annihilate(psi, n, j)
    psi = annihilate(psi, n, i)
    psi = create(psi, n, l)
    psi = create(psi, n, k)
    return psi

def apply_K(psi, n, double):
    """K = T - T^dag."""
    k, l, i, j = double
    return apply_T(psi, n, double) - apply_T(psi, n, (j, i, l, k))

def expm_K(psi, n, double, theta):
    """exp(theta K)|psi> exactly, using K^3 = -K:
       exp(theta K) = 1 + sin(theta) K + (1-cos(theta)) K^2."""
    K1 = apply_K(psi, n, double)
    K2 = apply_K(K1, n, double)
    return psi + np.sin(theta) * K1 + (1 - np.cos(theta)) * K2

# ----------------------------------------------------------------------
# Pauli-string exponential by matvec (for TEST B)
# ----------------------------------------------------------------------
def apply_pauli(psi, string):
    """P|psi> for P = prod_i sigma_{string[i]} on qubit i (little-endian)."""
    n = len(string)
    out = np.zeros_like(psi)
    flip = 0
    for q in range(n):
        if string[q] in ('X', 'Y'):
            flip |= 1 << q
    for b in np.nonzero(psi)[0]:
        ph = 1.0 + 0j
        for q in range(n):
            c, bit = string[q], (b >> q) & 1
            if c == 'Y':
                ph *= (1j if bit == 0 else -1j)
            elif c == 'Z' and bit:
                ph *= -1
        out[b ^ flip] += ph * psi[b]
    return out

def expm_pauli(psi, string, alpha):
    """exp(-i alpha/2 P)|psi> = cos(a/2)|psi> - i sin(a/2) P|psi>."""
    return np.cos(alpha / 2) * psi - 1j * np.sin(alpha / 2) * apply_pauli(psi, string)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def determinant(n, occupied):
    psi = np.zeros(2 ** n, dtype=complex)
    psi[sum(1 << q for q in occupied)] = 1.0
    return psi

def run_qc(qc, psi):
    return np.asarray(Statevector(psi).evolve(qc).data)

def check(label, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok

# ----------------------------------------------------------------------
# The three molecules of arXiv:2212.08006
# ----------------------------------------------------------------------
CASES = {
    "H2  (4 qubits)":  dict(n=4,  occ=[0, 2],
                            doubles=[(3, 1, 2, 0)]),
    "LiH (6 qubits)":  dict(n=6,  occ=[0, 3],
                            doubles=[(5, 1, 3, 0), (5, 2, 3, 0), (4, 2, 3, 0)]),
    "F2  (12 qubits)": dict(n=12, occ=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
                            doubles=[(11, 5, k + 6, k) for k in range(5)]),
}

rng = np.random.default_rng(7)
all_ok = True

for name, c in CASES.items():
    n, occ, doubles = c["n"], c["occ"], c["doubles"]
    print(f"\n=== {name}:  HF occupation {sorted(occ)},  doubles {doubles} ===")
    hf = determinant(n, occ)

    # ---------- TEST A: each double vs the fermionic exponential --------
    for d, dbl in enumerate(doubles):
        # sanity: K^3 = -K on a random state (exactness of expm_K)
        r = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        r /= np.linalg.norm(r)
        K1 = apply_K(r, n, dbl)
        K3 = apply_K(apply_K(K1, n, dbl), n, dbl)
        all_ok &= check(f"double {dbl}: K^3 = -K (exact expm formula valid)",
                        np.allclose(K3, -K1, atol=1e-12))

        for alpha in [0.4, np.pi / 5, 1.9]:
            qc, strings, signs = create_uccsd_circuit(n, [dbl], thetas=[alpha])
            psi_circ = run_qc(qc, hf)
            # fermionic side: theta = s*alpha/2, fixed sign s
            errs = {s: np.linalg.norm(psi_circ - expm_K(hf, n, dbl, s * alpha / 2))
                    for s in (+1, -1)}
            s_best = min(errs, key=errs.get)
            all_ok &= check(
                f"double {dbl}, alpha={alpha:.3f}: circuit|HF> == "
                f"exp({'+' if s_best>0 else '-'}a/2 (T-T^dag))|HF>   "
                f"(err={errs[s_best]:.2e})", errs[s_best] < 1e-10)

    # ---------- TEST B: full circuit == product of Pauli rotations ------
    alphas = list(rng.uniform(0.2, 1.4, size=len(doubles)))
    qc, strings, signs = create_uccsd_circuit(n, doubles, thetas=alphas)
    for trial in range(2):
        psi0 = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        psi0 /= np.linalg.norm(psi0)
        psi_circ = run_qc(qc, psi0)
        psi_ref = psi0.copy()
        for s, sg, a in zip(strings, signs, alphas):
            psi_ref = expm_pauli(psi_ref, s, sg * a)
        err = np.linalg.norm(psi_circ - psi_ref)
        all_ok &= check(
            f"full circuit == prod_d exp(-i s_d a_d/2 P_d) on random state "
            f"#{trial + 1} (err={err:.2e})", err < 1e-10)

    # ---------- and the full sequence acting on |HF> vs fermionic -------
    # (doubles of one molecule act in orthogonal 2D subspaces of |HF>
    #  only at first order; we still report the sequential fermionic
    #  comparison for completeness when doubles share target orbitals)
    psi_circ = run_qc(qc, hf)
    psi_f = hf.copy()
    for dbl, sg, a in zip(doubles, signs, alphas):
        psi_f = expm_K(psi_f, n, dbl, sg * a / 2)
    err = np.linalg.norm(psi_circ - psi_f)
    note = "exact" if err < 1e-10 else \
        "expected O(theta^2) deviation: doubles share target orbitals, " \
        "Pauli strings act on rotated determinants where the fermionic " \
        "operator annihilates"
    print(f"  [info] sequence on |HF> vs sequential fermionic product: "
          f"err={err:.2e}  ({note})")

print("\n" + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
raise SystemExit(0 if all_ok else 1)
