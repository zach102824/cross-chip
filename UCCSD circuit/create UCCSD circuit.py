"""
create UCCSD circuit.py
=======================
Build the optimised, CZ-native UCCSD-doubles circuit of arXiv:2212.08006
(Figs. 12-14 style) as a *Qiskit* QuantumCircuit.

Input
-----
* num_qubits : int                      -- user-defined qubit count N
* doubles    : list of 4-tuples (k,l,i,j) meaning the excitation
               a_k^dag a_l^dag a_i a_j   (e.g. (3,1,2,0) for H2;
               for F2 pass the five tuples (11,5,k+6,k), k=0..4)
* thetas     : list of rotation angles (one per double); if None, Qiskit
               Parameters t0, t1, ... are used.

Output
------
A QuantumCircuit made only of H, RX(+-pi/2), RX(theta) and CZ gates that
implements, for each double in order,

        exp( -i * theta_d / 2 * P_d )

where P_d is the single Jordan-Wigner Pauli string chosen per double
(the "one term in the qubit form" trick of arXiv:2212.08006, Supp. II C 3):

        P_d = Y_a X_b X_c X_e  *  (Z chains)        a<b<c<e sorted indices,
        Z on (a+1..b-1) and (c+1..e-1)              [JW parity strings]

Acting on a reference determinant this single string generates exactly the
same Givens rotation  cos|HF> + sin|D>  as the full 8-string fermionic
exponential exp(theta(T - T^dag)), with theta = +-thetas[d]/2 (the sign and
the 1/8 rescaling are absorbed by the variational parameter).

Each rotation is compiled with the Pauli-frame rules (CZ hangs a Z on a
neighbour, H opens X<->Z, RX(+-pi/2) maps Z<->Y):

        circuit_d = B  L  RX(theta)_pivot  L^dag  B^dag

with B a single-qubit basis layer and L a CZ chain that fans the support
into a middle "pivot" qubit from both sides (the paper uses balanced trees
on its 2x6 grid; a chain is topologically the same construction).  A final
peephole pass cancels gate pairs shared by consecutive doubles (the
common-a^dag/a sharing that gives the paper its 10/18/50 CZ counts).

The builder *self-verifies*: it symbolically conjugates X_pivot through the
compiled Clifford layers and asserts the result is the intended string P_d.

Run  `python "create UCCSD circuit.py"`  for a demo on H2 / LiH / F2.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# ----------------------------------------------------------------------
# Jordan-Wigner: one representative Pauli string per double
# ----------------------------------------------------------------------
def jw_string_for_double(num_qubits, double):
    """Return the odd-Y representative Pauli string (as a list of letters)
    for the double a_k^dag a_l^dag a_i a_j, paper convention:
    Y on the smallest index, X on the other three, JW Z-chains between the
    1st-2nd and 3rd-4th sorted indices.  e.g. (5,1,3,0), N=6 -> YXIXZX."""
    k, l, i, j = double
    a, b, c, e = sorted({k, l, i, j})
    if len({k, l, i, j}) != 4:
        raise ValueError("double must have four distinct orbital indices")
    s = ['I'] * num_qubits
    s[a] = 'Y'
    s[b] = s[c] = s[e] = 'X'
    for q in range(a + 1, b):
        s[q] = 'Z'
    for q in range(c + 1, e):
        s[q] = 'Z'
    return s

# ----------------------------------------------------------------------
# Symbolic Pauli-frame conjugation (used to self-verify the compilation)
# ----------------------------------------------------------------------
def _conj(letters, g):
    """g P g^dag for g in {('H',q), ('CZ',a,b), ('RX',q,+-pi/2)}."""
    phase, L = 1, list(letters)
    if g[0] == 'H':
        m = {'I': ('I', 1), 'X': ('Z', 1), 'Z': ('X', 1), 'Y': ('Y', -1)}
        L[g[1]], p = m[L[g[1]]]
        phase *= p
    elif g[0] == 'CZ':
        a, b = g[1], g[2]
        t = {('I','I'):(1,'I','I'),('I','X'):(1,'Z','X'),('I','Y'):(1,'Z','Y'),
             ('I','Z'):(1,'I','Z'),('X','I'):(1,'X','Z'),('Y','I'):(1,'Y','Z'),
             ('Z','I'):(1,'Z','I'),('X','X'):(1,'Y','Y'),('X','Y'):(-1,'Y','X'),
             ('Y','X'):(-1,'X','Y'),('Y','Y'):(1,'X','X'),('X','Z'):(1,'X','I'),
             ('Z','X'):(1,'I','X'),('Y','Z'):(1,'Y','I'),('Z','Y'):(1,'I','Y'),
             ('Z','Z'):(1,'Z','Z')}
        p, la, lb = t[(L[a], L[b])]
        L[a], L[b], phase = la, lb, phase * p
    elif g[0] == 'RX':
        s = 1 if g[2] > 0 else -1
        m = {'I': ('I', 1), 'X': ('X', 1), 'Y': ('Z', s), 'Z': ('Y', -s)}
        L[g[1]], p = m[L[g[1]]]
        phase *= p
    return phase, L

def _frame(prefix, pivot, num_qubits):
    """D^dag X_pivot D for the time-ordered Clifford gate list `prefix`."""
    letters = ['I'] * num_qubits
    letters[pivot] = 'X'
    phase = 1
    for g in reversed(prefix):
        ginv = ('RX', g[1], -g[2]) if g[0] == 'RX' else g
        p, letters = _conj(letters, ginv)
        phase *= p
    return phase, letters

# ----------------------------------------------------------------------
# Compile one Pauli rotation exp(-i theta/2 P) into (gates, pivot)
# ----------------------------------------------------------------------
def _compile_one(string):
    """Return (clifford_prefix, pivot, sign) such that
       prefix + RX(sign*theta)@pivot + reversed(prefix^dag)
    implements exp(-i theta/2 P).  prefix uses only H / RX(pi/2) / CZ."""
    n = len(string)
    sup = [q for q in range(n) if string[q] != 'I']
    pivot = sup[len(sup) // 2]                      # middle of the support
    basis, ladder = [], []
    # ladder letters: pivot and internal nodes carry X, the two chain
    # terminals carry Z (see Sec. 4 of the explainer note)
    for q in sup:
        terminal = (q == sup[0] or q == sup[-1]) and q != pivot
        ladder_letter = 'Z' if terminal else 'X'
        want = string[q]
        if ladder_letter == 'Z':
            if want == 'X':  basis.append(('H', q))
            elif want == 'Y': basis.append(('RX', q, np.pi / 2))
            # want == 'Z': nothing
        else:                                        # ladder letter X
            if want == 'Z':  basis.append(('H', q))
            elif want == 'Y':                        # X -> Y needs RZ; avoid:
                # re-route: give this qubit a Z ladder letter by H-dressing
                basis.append(('H', q)); basis.append(('RX', q, np.pi / 2))
            # want == 'X': nothing
    # chain fan-in from both ends towards the pivot
    ip = sup.index(pivot)
    left, right = sup[:ip + 1], sup[ip:][::-1]       # both end at pivot
    for side in (left, right):
        for a in range(len(side) - 1):
            ladder.append(('CZ', side[a], side[a + 1]))
            if side[a + 1] != pivot:
                ladder.append(('H', side[a + 1]))
    prefix = basis + ladder
    ph, letters = _frame(prefix, pivot, n)
    assert letters == list(string), (letters, string)
    assert ph in (1, -1)
    return prefix, pivot, ph                         # ph absorbed into angle

def _invert(gates):
    out = []
    for g in reversed(gates):
        out.append(('RX', g[1], -g[2]) if g[0] == 'RX' else g)
    return out

# ----------------------------------------------------------------------
# Peephole cancellation: remove adjacent self-inverse pairs (H H, CZ CZ,
# RX(a) RX(-a)) allowing commutation through gates on disjoint qubits.
# This recovers the "shared a^dag/a" savings between consecutive doubles.
# ----------------------------------------------------------------------
def _qubits_of(g):
    return {g[1]} if g[0] != 'CZ' else {g[1], g[2]}

def _cancels(g1, g2):
    if g1[0] != g2[0]:
        return False
    if g1[0] in ('H', 'CZ'):
        return g1 == g2
    if g1[0] == 'RX':
        return g1[1] == g2[1] and abs(g1[2] + g2[2]) < 1e-12
    return False

def _peephole(gates):
    gates = list(gates)
    changed = True
    while changed:
        changed = False
        for i in range(len(gates)):
            gi = gates[i]
            if gi[0] == 'ROT':
                continue
            qs = _qubits_of(gi)
            for j in range(i + 1, len(gates)):
                gj = gates[j]
                if _cancels(gi, gj):
                    del gates[j]; del gates[i]
                    changed = True
                    break
                blocker = (gj[0] == 'ROT') or (_qubits_of(gj) & qs)
                if blocker:
                    break
            if changed:
                break
    return gates

# ----------------------------------------------------------------------
# Public builder
# ----------------------------------------------------------------------
def create_uccsd_circuit(num_qubits, doubles, thetas=None, optimize=True):
    """Build the optimised UCCSD-doubles circuit.

    Parameters
    ----------
    num_qubits : int
    doubles    : list of (k, l, i, j) for a_k^dag a_l^dag a_i a_j
    thetas     : list of floats (or None -> Qiskit Parameters t0, t1, ...).
                 Each block implements exp(-i thetas[d]/2 * P_d).
    optimize   : run the peephole cancellation pass between blocks.

    Returns
    -------
    (QuantumCircuit, strings, signs) -- the circuit, the Pauli string P_d
    implemented for each double, and the sign s_d such that block d equals
    exp(-i * s_d * thetas[d]/2 * P_d)  (self-verified symbolically).
    """
    if thetas is None:
        thetas = [Parameter(f"t{d}") for d in range(len(doubles))]
    strings, signs, prog = [], [], []
    for d, dbl in enumerate(doubles):
        s = jw_string_for_double(num_qubits, dbl)
        strings.append(''.join(s))
        prefix, pivot, sign = _compile_one(s)
        signs.append(sign)
        prog += prefix
        prog.append(('ROT', pivot, d, sign))        # placeholder for RX(theta)
        prog += _invert(prefix)
    if optimize:
        prog = _peephole(prog)
    qc = QuantumCircuit(num_qubits)
    for g in prog:
        if g[0] == 'H':
            qc.h(g[1])
        elif g[0] == 'RX':
            qc.rx(g[2], g[1])
        elif g[0] == 'CZ':
            qc.cz(g[1], g[2])
        elif g[0] == 'ROT':
            _, pivot, d, sign = g
            qc.rx(sign * thetas[d], pivot)
    return qc, strings, signs

# ----------------------------------------------------------------------
# Demo: the three molecules of arXiv:2212.08006
# ----------------------------------------------------------------------
if __name__ == "__main__":
    cases = {
        "H2  (4q)":  (4,  [(3, 1, 2, 0)]),
        "LiH (6q)":  (6,  [(5, 1, 3, 0), (5, 2, 3, 0), (4, 2, 3, 0)]),
        "F2  (12q)": (12, [(11, 5, k + 6, k) for k in range(5)]),
    }
    for name, (n, dbls) in cases.items():
        qc, strs, sgns = create_uccsd_circuit(n, dbls, thetas=[np.pi / 5] * len(dbls))
        ops = qc.count_ops()
        print(f"{name}: doubles={dbls}")
        for s in strs:
            print(f"    string  {s}")
        print(f"    gates: {dict(ops)},  CZ = {ops.get('cz', 0)}, "
              f"depth = {qc.depth()}")
        if n <= 4:
            print(qc.draw(output='text', fold=120))
