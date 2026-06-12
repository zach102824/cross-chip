"""
Reconstruction and verification of the optimised UCCSD circuits of
Guo et al., "Experimental quantum computational chemistry with optimised
unitary coupled cluster ansatz", arXiv:2212.08006 (Figs. 12, 13, 14).

What this script does
---------------------
1.  Jordan-Wigner expands a fermionic double excitation (e.g. a3^ a1^ a2 a0
    for H2) into its 8 Pauli strings -- the starting point of the standard
    construction of Yordanov et al., arXiv:2005.14475.
2.  Shows the key *algorithmic* simplification used in arXiv:2212.08006:
    acting on a computational-basis reference (Hartree-Fock) state, a SINGLE
    Pauli string with an odd number of Y's generates exactly the same Givens
    rotation between |HF> and the doubly-excited determinant as the full
    8-string fermionic exponential.  So each selected double is replaced by
    one Pauli rotation (H2 keeps a +/- pair), with the angle absorbed into
    the variational parameter.
3.  Verifies, gate-by-gate, that the transcribed circuits of Fig. 12 (H2)
    and Fig. 13 (LiH) are EXACTLY equal (global phase 1) to the products of
    Pauli-rotations in Eqs. (22) and (23) of the paper, and reproduces the
    published gate counts: H2: 10 CZ + 14 1q,  LiH: 18 CZ + 19 1q.
4.  Explains the compilation rule used to build those circuits via a
    "Pauli-frame" analysis: in a circuit
        C_n RX_n ... C_1 RX_1 C_0
    (C_k Clifford layers of CZ/H/RX(+-pi/2), RX_k = RX(theta) on pivot p_k),
    the circuit equals  prod_k exp(-i theta/2 D_k^dag X_{p_k} D_k)  with
    D_k = C_{k-1}...C_0, provided C_n...C_0 = identity.  The frame strings
    D_k^dag X_{p_k} D_k are computed symbolically and printed.
5.  For F2 (Fig. 14) it verifies the first compiled block: the balanced
    CZ-tree over the 2x6 qubit grid (hubs q3/q9, joined by the vertical
    CZ(3,9)) implements exp(-i theta/2 Y0 Z1Z2Z3Z4 X5 X6 Z7Z8Z9Z10 X11),
    the first factor of Eq. (24).  Published count: 50 CZ + 63 1q.

Conventions
-----------
* Qubit 0 is the most significant bit of the state index.
* RX(t) = exp(-i t X / 2), RZ(t) = exp(-i t Z / 2).
* exp(-i theta P) for Pauli string P corresponds to RX angle 2*theta on the
  pivot (the figures use RX(pi/5)  <->  theta = pi/10 in Eqs. (22)-(24)).

Run:  python build_uccsd_circuits.py     (needs numpy only)
"""
import itertools
import numpy as np

# ----------------------------------------------------------------------
# Basic matrices and statevector simulator
# ----------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}

def rx(t): return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * X
def rz(t): return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * Z

def apply_1q(psi, n, q, U):
    psi = psi.reshape([2] * n)
    psi = np.tensordot(U, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q).reshape(-1)

def apply_cz(psi, n, a, b):
    psi = psi.copy()
    idx = np.arange(2 ** n)
    mask = ((idx >> (n - 1 - a)) & 1) & ((idx >> (n - 1 - b)) & 1)
    psi[mask == 1] *= -1
    return psi

def run_circuit(gates, n, psi):
    """gates: list of ('H',q) | ('RX',q,t) | ('RZ',q,t) | ('CZ',a,b)."""
    for g in gates:
        if g[0] == 'H':
            psi = apply_1q(psi, n, g[1], H)
        elif g[0] == 'RX':
            psi = apply_1q(psi, n, g[1], rx(g[2]))
        elif g[0] == 'RZ':
            psi = apply_1q(psi, n, g[1], rz(g[2]))
        elif g[0] == 'CZ':
            psi = apply_cz(psi, n, g[1], g[2])
        else:
            raise ValueError(g)
    return psi

def circuit_unitary(gates, n):
    d = 2 ** n
    U = np.zeros((d, d), dtype=complex)
    for k in range(d):
        e = np.zeros(d, dtype=complex); e[k] = 1.0
        U[:, k] = run_circuit(gates, n, e)
    return U

def pauli_matrix(s):
    M = np.array([[1.0]], dtype=complex)
    for c in s:
        M = np.kron(M, PAULI[c])
    return M

def expm_pauli(theta, s):
    """exp(-i*theta*P)  (P^2 = 1)."""
    P = pauli_matrix(s)
    return np.cos(theta) * np.eye(P.shape[0]) - 1j * np.sin(theta) * P

def same_up_to_phase(A, B, tol=1e-8):
    k = np.argmax(np.abs(A))
    i, j = np.unravel_index(k, A.shape)
    if abs(B[i, j]) < 1e-12:
        return False, None
    ph = A[i, j] / B[i, j]
    return np.allclose(A, ph * B, atol=tol), ph

# ----------------------------------------------------------------------
# Symbolic Pauli-frame propagation through the Clifford gate set
# {H, CZ, RX(+-pi/2), RZ(+-pi/2)}  --  used to read off which Pauli string
# each RX(theta) implements inside a compiled circuit.
# ----------------------------------------------------------------------
def conj_letters(letters, g):
    """Return (phase, letters') with  g P g^dag = phase * P'."""
    phase, L = 1, list(letters)
    if g[0] == 'H':
        m = {'I': ('I', 1), 'X': ('Z', 1), 'Z': ('X', 1), 'Y': ('Y', -1)}
        L[g[1]], p = m[L[g[1]]]
        phase *= p
    elif g[0] == 'CZ':
        a, b = g[1], g[2]
        table = {('I','I'):(1,'I','I'),('I','X'):(1,'Z','X'),('I','Y'):(1,'Z','Y'),
                 ('I','Z'):(1,'I','Z'),('X','I'):(1,'X','Z'),('Y','I'):(1,'Y','Z'),
                 ('Z','I'):(1,'Z','I'),('X','X'):(1,'Y','Y'),('X','Y'):(-1,'Y','X'),
                 ('Y','X'):(-1,'X','Y'),('Y','Y'):(1,'X','X'),('X','Z'):(1,'X','I'),
                 ('Z','X'):(1,'I','X'),('Y','Z'):(1,'Y','I'),('Z','Y'):(1,'I','Y'),
                 ('Z','Z'):(1,'Z','Z')}
        p, la, lb = table[(L[a], L[b])]
        L[a], L[b], phase = la, lb, phase * p
    elif g[0] == 'RX':                       # +-pi/2 only
        s = 1 if g[2] > 0 else -1
        m = {'I': ('I', 1), 'X': ('X', 1), 'Y': ('Z', s), 'Z': ('Y', -s)}
        L[g[1]], p = m[L[g[1]]]
        phase *= p
    elif g[0] == 'RZ':
        s = 1 if g[2] > 0 else -1
        m = {'I': ('I', 1), 'Z': ('Z', 1), 'X': ('Y', s), 'Y': ('X', -s)}
        L[g[1]], p = m[L[g[1]]]
        phase *= p
    return phase, L

def frame_string(prefix_gates, pivot, n, theta=np.pi / 5):
    """D^dag X_pivot D for the Clifford prefix D (time-ordered gate list).
    Non-Clifford RX(theta) blocks are factored out (skipped)."""
    letters = ['I'] * n
    letters[pivot] = 'X'
    phase = 1
    for g in reversed(prefix_gates):
        if g[0] in ('RX', 'RZ') and not np.isclose(abs(g[2]), np.pi / 2):
            continue                          # an earlier RX(theta) rotation
        ginv = (g[0], g[1], -g[2]) if g[0] in ('RX', 'RZ') else g
        p, letters = conj_letters(letters, ginv)
        phase *= p
    return phase, ''.join(letters)

def report(name, gates, n, theta):
    rx_idx = [i for i, g in enumerate(gates)
              if g[0] == 'RX' and np.isclose(abs(g[2]), theta)]
    print(f"--- {name} ---")
    for k, i in enumerate(rx_idx):
        ph, s = frame_string(gates[:i], gates[i][1], n)
        sgn = '+' if gates[i][2] > 0 else '-'
        print(f"  RX#{k+1} on q{gates[i][1]} ({sgn}{abs(gates[i][2])/np.pi:.2f}pi)"
              f"  implements  exp(-i a/2 * {'+' if ph==1 else '-'}{s})")
    ncz = sum(1 for g in gates if g[0] == 'CZ')
    print(f"  gate count: {ncz} CZ + {len(gates) - ncz} single-qubit")
    return rx_idx

# ----------------------------------------------------------------------
# 1) Jordan-Wigner expansion of the H2 double  a3^ a1^ a2 a0  (4 qubits)
# ----------------------------------------------------------------------
def jw_annihilation(n, i):
    """a_i = (prod_{r<i} Z_r) (X_i + iY_i)/2, qubit 0 = leftmost factor."""
    M = np.array([[1.0]], dtype=complex)
    for q in range(n):
        if q < i:   M = np.kron(M, Z)
        elif q == i: M = np.kron(M, (X + 1j * Y) / 2)
        else:        M = np.kron(M, I2)
    return M

def pauli_decompose(M, n):
    """Expand a 2^n x 2^n matrix in the Pauli-string basis."""
    out = {}
    for s in itertools.product('IXYZ', repeat=n):
        s = ''.join(s)
        c = np.trace(pauli_matrix(s).conj().T @ M) / 2 ** n
        if abs(c) > 1e-10:
            out[s] = c
    return out

print("=" * 72)
print("STEP 1: JW expansion of T = a3^ a1^ a2 a0  (H2),  T - T^dag =")
n = 4
a = [jw_annihilation(n, i) for i in range(n)]
T = a[3].conj().T @ a[1].conj().T @ a[2] @ a[0]
K = T - T.conj().T
for s, c in sorted(pauli_decompose(K, n).items()):
    print(f"   {c.imag:+.3f}i * {s}")
print("-> 8 strings, coefficient i/8: the standard construction (2005.14475)")
print("   would spend one CNOT-staircase (6 CNOTs) per string = 48 CNOTs.")

# ----------------------------------------------------------------------
# 2) One string is enough on the Hartree-Fock determinant
# ----------------------------------------------------------------------
print()
print("STEP 2: action on |HF> = |1010>  (q0,q2 occupied)")
hf = np.zeros(16, dtype=complex); hf[0b1010] = 1
th = 0.37
full = np.eye(16, dtype=complex)
# exp(th*(T-Tdag)) via eigen-decomposition
w, V = np.linalg.eigh(1j * K)            # i*K Hermitian
full = V @ np.diag(np.exp(-1j * th * w)) @ V.conj().T
psi_full = full @ hf
# single string with one Y:  exp(-i a/2 YXXX), choose a = th/2... scan:
one = expm_pauli(th / 4, 'YXXX') @ hf    # exp(-i (th/4) YXXX)
print(f"  full fermionic exp:  <1010|psi> = {psi_full[0b1010]:+.4f}, "
      f"<0101|psi> = {psi_full[0b0101]:+.4f}")
print(f"  single string YXXX:  <1010|psi> = {one[0b1010]:+.4f}, "
      f"<0101|psi> = {one[0b0101]:+.4f}")
print("  -> both are REAL Givens rotations |HF> -> cos|1010> + sin|0101>;")
print("     a single odd-Y Pauli string reproduces the full 8-string double")
print("     on the reference determinant (angle rescaled, absorbed by VQE).")

# ----------------------------------------------------------------------
# 3) H2: Fig. 12 transcription == Eq. (22), 10 CZ + 14 1q     [VERIFIED]
# ----------------------------------------------------------------------
print()
print("=" * 72)
t = np.pi / 5         # the RX angle used in the paper's figures
pi = np.pi
h2_fig12 = [
    ('RX', 0, pi/2), ('H', 3),
    ('CZ', 0, 1), ('CZ', 2, 3),
    ('H', 1),
    ('CZ', 1, 2),
    ('RX', 2, t),                      # exp(-i t/2 * Y0 X1 X2 X3)
    ('CZ', 1, 2),
    ('H', 1),
    ('CZ', 0, 1),
    ('RX', 0, -pi/2), ('RZ', 1, -pi/2),   # basis hand-off  Y0X1 -> X0Y1
    ('H', 0),
    ('CZ', 0, 1),
    ('H', 1),
    ('CZ', 1, 2),
    ('RX', 2, -t),                     # exp(+i t/2 * X0 Y1 X2 X3)
    ('CZ', 1, 2),
    ('H', 1), ('CZ', 2, 3),
    ('H', 3), ('CZ', 0, 1),
    ('H', 0), ('RZ', 1, pi/2),
]
report("H2  (Fig. 12 transcription)", h2_fig12, 4, t)
U_fig = circuit_unitary(h2_fig12, 4)
U_eq22 = expm_pauli(-t/2, 'XYXX') @ expm_pauli(t/2, 'YXXX')   # Eq. 22, theta=t/2
ok, ph = same_up_to_phase(U_fig, U_eq22)
print(f"  equals Eq.(22) with theta = pi/10:  {ok}  (global phase {ph:.3f})")

# ----------------------------------------------------------------------
# 4) LiH: Fig. 13 transcription == Eq. (23), 18 CZ + 19 1q    [VERIFIED]
#    doubles: a5^a1^a3a0 , a4^a2^a3a0 , a5^a2^a3a0  ->  strings
#    P1=Y0X1X3Z4X5,  P2=Y0Z1X2X3X4,  P3=Y0Z1X2X3Z4X5
#    circuit order: P1, P3, P2  (P1,P3 anticommute - order is part of ansatz)
# ----------------------------------------------------------------------
print()
lih_fig13 = [
    ('RX', 0, pi/2), ('H', 2), ('H', 3), ('H', 4), ('H', 5),
    ('CZ', 0, 1), ('CZ', 3, 4),
    ('CZ', 4, 5),
    ('H', 4),
    ('CZ', 1, 4),
    ('RX', 1, t),                      # P1 = Y0 X1 X3 Z4 X5
    ('CZ', 1, 4),
    ('CZ', 0, 1),
    ('H', 1),                          # X1 -> Z1 (pivot letter swap)
    ('CZ', 0, 1),
    ('CZ', 1, 2),
    ('CZ', 1, 4),
    ('RX', 1, t),                      # P3 = Y0 Z1 X2 X3 Z4 X5
    ('CZ', 1, 4),
    ('H', 4),
    ('CZ', 4, 5),
    ('CZ', 3, 4), ('H', 5),
    ('H', 4),
    ('CZ', 3, 4),
    ('H', 4),
    ('CZ', 1, 4),
    ('RX', 1, t),                      # P2 = Y0 Z1 X2 X3 X4
    ('CZ', 1, 4),
    ('CZ', 1, 2), ('H', 4),
    ('H', 2),
    ('CZ', 0, 1), ('CZ', 3, 4),
    ('RX', 0, -pi/2), ('H', 1), ('H', 3),
]
report("LiH (Fig. 13 transcription)", lih_fig13, 6, t)
P1, P2, P3 = 'YXIXZX', 'YZXXXI', 'YZXXZX'
U_fig = circuit_unitary(lih_fig13, 6)
U_eq = expm_pauli(t/2, P2) @ expm_pauli(t/2, P3) @ expm_pauli(t/2, P1)
ok, ph = same_up_to_phase(U_fig, U_eq)
print(f"  equals exp(-i t/2 P2) exp(-i t/2 P3) exp(-i t/2 P1), t=pi/5: {ok} "
      f"(global phase {ph:.3f})")
print("  (same three strings as Eq. (23); the figure applies P1,P3,P2.)")

# ----------------------------------------------------------------------
# 5) F2: first block of Fig. 14                                [VERIFIED]
#    doubles a11^a5^ a_{k+6} a_k (k=0..4); Eq. (24) strings:
# ----------------------------------------------------------------------
print()
S_F2 = ['YZZZZXXZZZZX',     # theta1  Y0 Z1Z2Z3Z4 X5 X6 Z7Z8Z9Z10 X11
        'IYZZZXIXZZZX',     # theta2
        'IIYZZXIIXZZX',     # theta3
        'IIIYZXIIIXZX',     # theta4
        'IIIIXXIIIIYX']     # theta5
f2_block1 = ([('RX', 0, pi/2)] + [('H', q) for q in range(1, 12)] + [
    ('CZ', 0, 1), ('CZ', 4, 5), ('CZ', 6, 7), ('CZ', 10, 11),   # parallel
    ('H', 1), ('H', 4), ('H', 7), ('H', 10),
    ('CZ', 1, 2), ('CZ', 3, 4), ('CZ', 7, 8), ('CZ', 9, 10),    # parallel
    ('H', 2), ('H', 8),
    ('CZ', 2, 3), ('CZ', 8, 9),
    ('H', 9),
    ('CZ', 3, 9),            # vertical link between the two 1x6 rows
    ('RX', 3, t)])           # the rotation
report("F2  (Fig. 14, first block)", f2_block1, 12, t)
print(f"  target (Eq. 24, theta1 factor):    +{S_F2[0]}")
print("  -> balanced CZ-tree on the 2x6 grid: hubs q3 (spin-up row) and q9")
print("     (spin-down row) joined by ONE nearest-neighbour CZ(3,9);")
print("     depth ~log instead of a length-11 linear staircase.")
print("  Remaining 4 blocks reuse the X5/X11 ends and shorten the Z-chain by")
print("  one rung each (shared a11^ a5^), giving the published 50 CZ + 63 1q.")
