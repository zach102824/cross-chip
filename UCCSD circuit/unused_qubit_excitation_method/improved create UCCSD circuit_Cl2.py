"""
improved create UCCSD circuit .py
=================================
Improved version of "create UCCSD circuit.py" that reproduces the hand-tuned
gate counts of arXiv:2212.08006 (H2: 10 CZ, LiH: 18 CZ, F2: 50 CZ).

Same API:  create_uccsd_circuit(num_qubits, doubles, thetas, ...)
           -> (QuantumCircuit, strings, signs)

Where the extra efficiency comes from (all four ingredients are GENERAL,
none is molecule-specific):

1.  TWO-ROW HUB LADDERS instead of a single chain.
    The qubits encode spin-up orbitals on row 0..N/2-1 and spin-down on
    N/2..N-1 (the paper's 2 x N/2 grid).  Every double excites within the
    rows, so the support of each Pauli string splits into an alpha part and
    a beta part.  Each row fans into a row "hub" by a CZ/H chain, the beta
    hub is opened with one H and linked to the pivot (= alpha hub) by a
    single CZ -- vertically adjacent on the chip ((p, p+N/2)).  This is the
    tree layout visible in Figs. 13/14.

2.  HUB CONTINUITY.  The hub of a row is kept on the same qubit for
    consecutive doubles whenever it lies in both supports.  Shared subtrees
    (= shared a^dag/a indices, e.g. a11^ a5^ in every F2 double) then sit on
    identical gates, which the cancellation pass removes; un-shared parts
    are peeled/rebuilt only where the strings differ.

3.  CZ-CZ COMMUTATION in the cancellation pass.  All CZ gates commute with
    each other (they are Z-diagonal), even when they share a qubit.  The
    original peephole only commuted gates with disjoint supports and
    therefore missed e.g.  CZ(1,4) ... CZ(0,1) ... CZ(1,4)  cancellations.
    With this rule the pass finds exactly the interface Cliffords of the
    paper (e.g. LiH:  CZ14 CZ01 H1 CZ01 CZ12 CZ14).

4.  ADJACENT-OVERLAP ORDERING ('order="auto"').  The doubles are ordered to
    maximise the letter overlap of consecutive strings (brute force, the
    selected pools are tiny).  For LiH this turns Eq.(23)'s order
    (P1,P2,P3) into the figure's order (P1,P3,P2): P1/P3 and P3/P2 overlap
    on 4 qubits each, P1/P2 only on 2.

5.  Optional 'pair=True': emit the exact two-string qubit-excitation pair
        exp(-i t/2 Y_a X_b X_c X_e ...) exp(+i t/2 X_a Y_b X_c X_e ...)
    per double (what Fig. 12 does for H2 -> 10 CZ).  The default single
    string is cheaper (6 CZ) and variationally equivalent on a determinant.

Every block is still verified symbolically (Pauli-frame conjugation), and
after optimisation the whole program is re-verified: each RX implements the
intended string and the residual Clifford is the identity.  Run the file
for a demo + statevector check against the product of Pauli exponentials.
"""
import itertools
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

try:                                  # only needed for the hardware-aware build
    import networkx as nx
except ImportError:                   # pragma: no cover
    nx = None

# ----------------------------------------------------------------------
# Jordan-Wigner representative strings (same as the original file)
# ----------------------------------------------------------------------
def jw_string_for_double(num_qubits, double, y_pos=0):
    """Odd-Y representative for a_k^dag a_l^dag a_i a_j.
    y_pos selects which of the four support qubits carries the Y."""
    k, l, i, j = double
    sup = sorted({k, l, i, j})
    if len(sup) != 4:
        raise ValueError("double must have four distinct orbital indices")
    s = ['I'] * num_qubits
    for m, q in enumerate(sup):
        s[q] = 'Y' if m == y_pos else 'X'
    for q in range(sup[0] + 1, sup[1]):
        s[q] = 'Z'
    for q in range(sup[2] + 1, sup[3]):
        s[q] = 'Z'
    return s

# ----------------------------------------------------------------------
# Symbolic Pauli-frame conjugation
# ----------------------------------------------------------------------
def _conj(letters, g):
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

def _frame(prefix, pivot, n):
    """D^dag X_pivot D for time-ordered Clifford prefix (ROT/RX(t) skipped)."""
    letters = ['I'] * n
    letters[pivot] = 'X'
    phase = 1
    for g in reversed(prefix):
        if g[0] == 'ROT':
            continue
        if g[0] == 'RX' and not np.isclose(abs(g[2]), np.pi / 2):
            continue
        ginv = ('RX', g[1], -g[2]) if g[0] == 'RX' else g
        p, letters = _conj(letters, ginv)
        phase *= p
    return phase, letters

# ----------------------------------------------------------------------
# IMPROVEMENT 1+2: two-row hub ladder with hub continuity
# ----------------------------------------------------------------------
_BASIS_CANDIDATES = ([], [('H',)], [('RX', np.pi/2)], [('RX', -np.pi/2)],
                     [('H',), ('RX', np.pi/2)], [('RX', np.pi/2), ('H',)],
                     [('H',), ('RX', -np.pi/2)], [('RX', -np.pi/2), ('H',)])

def _basis_for(lad, want, q):
    """Smallest 1q gate list b (time order, applied BEFORE the ladder) with
    conj of `want` through b^dag ... = `lad` up to +-1 phase."""
    for cand in _BASIS_CANDIDATES:
        gates = [(c[0], q) if c[0] == 'H' else (c[0], q, c[1]) for c in cand]
        letters, ph = [want], 1
        for g in gates:                      # frame conj order: b1 then b2 ...
            ginv = ('RX', 0, -g[2]) if g[0] == 'RX' else ('H', 0)
            p, letters = _conj(letters, ginv)
            ph *= p
        if letters[0] == lad and ph in (1, -1):
            return gates
    raise RuntimeError(f"no basis gate for {lad} -> {want}")

def _row_chain(nodes, hub):
    """CZ/H fan-in of `nodes` (sorted) into hub: chains from both ends."""
    gates = []
    left = [q for q in nodes if q < hub] + [hub]
    right = [q for q in nodes if q > hub][::-1] + [hub]
    for side in (left, right):
        for a in range(len(side) - 1):
            gates.append(('CZ', side[a], side[a + 1]))
            if side[a + 1] != hub:
                gates.append(('H', side[a + 1]))
    return gates

def _compile_tworow(string, n, hub_hint=None):
    """Compile exp(-i t/2 P) with the two-row hub layout.
    Returns (prefix, pivot, sign): prefix+RX(sign*t)+prefix^dag = the block."""
    half = n // 2
    sup = [q for q in range(n) if string[q] != 'I']
    alpha = [q for q in sup if q < half]
    beta = [q for q in sup if q >= half]
    if not alpha or not beta:                 # degenerate: single chain
        row = alpha or beta
        pivot = hub_hint if hub_hint in row else row[len(row) // 2]
        ladder = _row_chain(row, pivot)
        hubs = [pivot]
    else:
        pivot = hub_hint if hub_hint in alpha else alpha[len(alpha) // 2]
        # beta hub: vertically below the pivot if possible, else nearest
        want_b = pivot + half
        hub_b = want_b if want_b in beta else min(beta, key=lambda q: abs(q - want_b))
        ladder = _row_chain(alpha, pivot) + _row_chain(beta, hub_b)
        ladder.append(('H', hub_b))           # open the beta hub
        ladder.append(('CZ', pivot, hub_b))   # single vertical link
        hubs = [pivot, hub_b]
    # ladder letters seen at each qubit (conjugate X_pivot through the ladder)
    _, lad = _frame(ladder, pivot, n)
    basis = []
    for q in sup:
        basis += _basis_for(lad[q], string[q], q)
    prefix = basis + ladder
    ph, letters = _frame(prefix, pivot, n)
    assert letters == list(string), (''.join(letters), string)
    assert ph in (1, -1)
    return prefix, pivot, ph

def _invert(gates):
    return [('RX', g[1], -g[2]) if g[0] == 'RX' else g for g in reversed(gates)]

# ----------------------------------------------------------------------
# IMPROVEMENT 3: peephole with CZ-CZ commutation
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

def _commute(g1, g2):
    """Sound (sufficient) commutation rules."""
    if g2[0] == 'ROT' or g1[0] == 'ROT':
        return False
    if not (_qubits_of(g1) & _qubits_of(g2)):
        return True
    if g1[0] == 'CZ' and g2[0] == 'CZ':
        return True                          # all Z-diagonal gates commute
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
            for j in range(i + 1, len(gates)):
                gj = gates[j]
                if _cancels(gi, gj):
                    del gates[j]; del gates[i]
                    changed = True
                    break
                if not _commute(gi, gj):
                    break
            if changed:
                break
    return gates

# ----------------------------------------------------------------------
# IMPROVEMENT 4: order the doubles by adjacent string overlap
# ----------------------------------------------------------------------
def _overlap(s1, s2):
    return sum(1 for a, b in zip(s1, s2) if a == b and a != 'I')

def _auto_order(strings):
    m = len(strings)
    if m <= 2 or m > 7:
        return list(range(m))
    best, best_score = list(range(m)), -1
    for perm in itertools.permutations(range(m)):
        sc = sum(_overlap(strings[perm[a]], strings[perm[a + 1]])
                 for a in range(m - 1))
        if sc > best_score:
            best, best_score = list(perm), sc
    return best

# ----------------------------------------------------------------------
# Whole-program verification (after optimisation)
# ----------------------------------------------------------------------
def _verify_program(prog, n, expected):
    """expected: list of (string, sign) in program order.  Checks each ROT's
    frame and that the residual Clifford is the identity."""
    k = 0
    for i, g in enumerate(prog):
        if g[0] == 'ROT':
            ph, letters = _frame(prog[:i], g[1], n)
            s, sgn = expected[k]
            assert ''.join(letters) == s and ph == sgn, \
                (k, ''.join(letters), ph, s, sgn)
            k += 1
    cliff = [g for g in prog if g[0] != 'ROT']
    for q in range(n):
        for axis in 'XZ':
            letters = ['I'] * n
            letters[q] = axis
            ph, L = 1, letters
            for g in reversed(cliff):
                ginv = ('RX', g[1], -g[2]) if g[0] == 'RX' else g
                p, L = _conj(L, ginv)
                ph *= p
            target = ['I'] * n
            target[q] = axis
            assert L == target and ph == 1, "residual Clifford != identity"

# ----------------------------------------------------------------------
# Initial state preparation (Hartree-Fock or paper Eq. (6) multireference)
# ----------------------------------------------------------------------
def initial_state_circuit(num_qubits, n_electrons=None, occupied=None,
                          beta=None):
    """Reference-state preparation circuit (paper qubit layout: spin-up
    orbitals on 0..N/2-1, spin-down on N/2..N-1, lowest MO first).

    Parameters
    ----------
    n_electrons : total electron count; occupied orbitals default to the
                  lowest eta = n_electrons//2 in each spin row
    occupied    : explicit list of occupied spin orbitals (overrides
                  n_electrons); HF determinant = X on each of them
    beta        : if not None, prepare the multireference state of
                  arXiv:2212.08006 Eq. (6),
                      (|HF> - beta * a^dag_{eta+N/2} a^dag_eta
                              a_{eta+N/2-1} a_{eta-1} |HF>) / sqrt(1+beta^2),
                  via Ry(-2 atan beta) on q_{eta-1} followed by 3 CNOTs and
                  the X layer (Eqs. (7)-(8), Fig. 10).  Requires the default
                  closed-shell occupation (give n_electrons, not occupied).
    """
    half = num_qubits // 2
    if occupied is None:
        if n_electrons is None:
            raise ValueError("give n_electrons or occupied")
        eta = n_electrons // 2
        occupied = list(range(eta)) + list(range(half, half + eta))
    qc = QuantumCircuit(num_qubits)
    if beta is None:                          # plain HF determinant
        for q in sorted(occupied):
            qc.x(q)
        return qc
    if n_electrons is None:
        raise ValueError("multireference prep needs n_electrons")
    eta = n_electrons // 2
    assert sorted(occupied) == list(range(eta)) + list(range(half, half + eta)), \
        "multireference prep assumes the closed-shell HF occupation"
    # paper Eq. (7): Ry |0> = (|0> - beta |1>)/sqrt(1+beta^2) on q_{eta-1}
    qc.ry(-2 * np.arctan(beta), eta - 1)
    # paper Eq. (8), rightmost CNOT acts first
    qc.cx(eta - 1, eta)
    qc.cx(eta, half + eta - 1)
    qc.cx(half + eta - 1, half + eta)
    for q in list(range(eta)) + list(range(half, half + eta)):
        qc.x(q)
    return qc

# ----------------------------------------------------------------------
# Public builder
# ----------------------------------------------------------------------
def create_uccsd_circuit(num_qubits, doubles, thetas=None, optimize=True,
                         order='auto', pair=False,
                         init_state=None, n_electrons=None, occupied=None,
                         beta=None):
    """Improved optimised UCCSD-doubles circuit (see module docstring).

    Parameters
    ----------
    num_qubits : int  (spin-up orbitals on 0..N/2-1, spin-down on N/2..N-1)
    doubles    : list of (k, l, i, j) for a_k^dag a_l^dag a_i a_j
    thetas     : list of floats or None (-> Qiskit Parameters)
    optimize   : run the CZ-commuting cancellation pass
    order      : 'auto' (overlap-maximising) or 'given'
    pair       : emit the exact 2-string qubit-excitation pair per double
                 (Fig. 12 style; doubles the rotation count)
    init_state : None (bare ansatz), 'hf' (X-gate determinant) or
                 'multiref' (paper Eq. (6); needs beta).  The preparation
                 is prepended before the ansatz, separated by a barrier.
    n_electrons / occupied / beta : forwarded to initial_state_circuit()

    Returns (QuantumCircuit, strings, signs, theta_idx): rotation k (in
    circuit order, possibly reordered by 'auto') implements
        exp(-i * signs[k] * thetas[theta_idx[k]] / 2 * strings[k]);
    with pair=True each double contributes two consecutive rotations
    (the second with the opposite sign: the qubit-excitation pair).
    """
    if thetas is None:
        thetas = [Parameter(f"t{d}") for d in range(len(doubles))]
    raw = [''.join(jw_string_for_double(num_qubits, d)) for d in doubles]
    idx = _auto_order(raw) if order == 'auto' else list(range(len(doubles)))

    # build the rotation list: (string, theta_index, theta_sign_multiplier)
    rots = []
    for d in idx:
        rots.append((raw[d], d, +1))
        if pair:                              # partner string: Y moved 0 -> 1
            s2 = ''.join(jw_string_for_double(num_qubits, doubles[d], y_pos=1))
            rots.append((s2, d, -1))          # opposite angle: qubit excitation

    prog, expected, hub = [], [], None
    for (s, d, mult) in rots:
        prefix, pivot, ph = _compile_tworow(s, num_qubits, hub_hint=hub)
        hub = pivot                           # hub continuity
        prog += prefix
        # block = exp(-i a/2 * ph * P) with a = mult*ph*theta
        #       = exp(-i mult*theta/2 * P)
        prog.append(('ROT', pivot, d, mult * ph))
        prog += _invert(prefix)
        expected.append((s, ph))
    if optimize:
        prog = _peephole(prog)
    _verify_program(prog, num_qubits, expected)

    qc = QuantumCircuit(num_qubits)
    if init_state is not None:
        if init_state == 'hf':
            prep = initial_state_circuit(num_qubits, n_electrons=n_electrons,
                                         occupied=occupied)
        elif init_state == 'multiref':
            prep = initial_state_circuit(num_qubits, n_electrons=n_electrons,
                                         occupied=occupied, beta=beta)
        else:
            raise ValueError(f"unknown init_state {init_state!r}")
        qc.compose(prep, inplace=True)
        qc.barrier()
    strings = [r[0] for r in rots]
    signs = [r[2] for r in rots]              # effective sign: exp(-i s*t/2 P)
    theta_idx = [r[1] for r in rots]
    for g in prog:
        if g[0] == 'H':
            qc.h(g[1])
        elif g[0] == 'RX':
            qc.rx(g[2], g[1])
        elif g[0] == 'CZ':
            qc.cz(g[1], g[2])
        else:
            _, pivot, d, a_sgn = g
            qc.rx(a_sgn * thetas[d], pivot)
    return qc, strings, signs, theta_idx

# ----------------------------------------------------------------------
# Diagram helpers (shared by the generation pipeline below)
# ----------------------------------------------------------------------
# Qiskit "iqp" palette, hard-coded so the look is identical regardless of the
# installed qiskit version: salmon H, blue X/CX, maroon rotations, light-blue
# CZ/RZX links -- same palette as June_main/circuits2read/plot_hf_circuits_compare.ipynb.
IQP_STYLE = {
    "name": "iqp",
    "displaycolor": {
        "h": ["#FA4D56", "#000000"],
        "x": ["#002D9C", "#FFFFFF"],
        "rx": ["#9F1853", "#FFFFFF"],
        "ry": ["#9F1853", "#FFFFFF"],
        "rz": ["#33B1FF", "#000000"],
        "rzx": ["#33B1FF", "#000000"],
        "cz": ["#33B1FF", "#000000"],
        "cx": ["#002D9C", "#000000"],
    },
}


def qc_from_logical_gates(gates, num_qubits):
    """Rebuild a Qiskit circuit from a saved logical/decomposed gate list
    (the same JSON schema produced by uccsd_circuit_io.save_circuit_json),
    so it can be drawn.  Symbolic rotations keep their parameter name."""
    qc = QuantumCircuit(num_qubits)
    params: dict[str, Parameter] = {}

    def angle(g):
        if "param" in g:
            p = params.setdefault(g["param"], Parameter(g["param"]))
            return float(g.get("coeff", 1.0)) * p
        return float(g.get("value", g.get("angle", 0.0)))

    for g in gates:
        op = g["op"].lower()
        qs = g["qubits"]
        if op == "x":
            qc.x(qs[0])
        elif op == "h":
            qc.h(qs[0])
        elif op == "cx":
            qc.cx(qs[0], qs[1])
        elif op == "cz":
            qc.cz(qs[0], qs[1])
        elif op == "rx":
            qc.rx(angle(g), qs[0])
        elif op == "ry":
            qc.ry(angle(g), qs[0])
        elif op == "rz":
            qc.rz(angle(g), qs[0])
        elif op == "rzx":
            qc.rzx(angle(g), qs[0], qs[1])
        else:
            raise ValueError(f"unsupported gate op {op!r}")
    return qc


def fuse_cz_rot_cz_to_rzx(gates):
    """Fuse every  CZ(a,b) . RX(theta)@q . CZ(a,b)  sandwich (q in {a,b}) into a
    single continuous-angle native RZX gate.

    Why this is exact.  CZ is symmetric and Z-diagonal, so it conjugates the
    X on the rotated qubit into an X (x) Z two-body Pauli:
        CZ(a,b) . exp(-i th/2 X_q) . CZ(a,b)  ==  exp(-i th/2  X_q (x) Z_other)
    and that is precisely Qiskit's  RZX(th)  with the Z on `other` and the X on
    `q`, i.e.  rzx(th, other, q)  (verified by Operator equality up to global
    phase).  Each fused block therefore turns 2 CZ + 1 single-qubit rotation
    into ONE continuous-theta RZX, keeping the angle symbolic.

    The sandwich is exactly what every double leaves behind on the alpha<->beta
    vertical link: the ladder ends on CZ(pivot, hub_b), the RX(theta) pivot
    rotation follows, and the inverse ladder re-applies CZ(pivot, hub_b); the
    ROT in between blocks the two CZs from cancelling in `_peephole`.

    Operates on the JSON logical gate-list (the schema of
    uccsd_circuit_io.circuit_to_logical_gates).  Gates on qubits disjoint from
    {a,b} between the two CZs are left untouched (they commute through).
    Returns a new list; rotations may be symbolic (param/coeff) or numeric.
    """
    gates = [dict(g) for g in gates]
    changed = True
    while changed:
        changed = False
        n = len(gates)
        for i in range(n):
            g = gates[i]
            if g["op"] != "cz":
                continue
            pair = set(g["qubits"])
            a, b = g["qubits"]
            mid_rot = mid_idx = close_idx = None
            ok = True
            for j in range(i + 1, n):
                gj = gates[j]
                touched = set(gj["qubits"]) & pair
                if not touched:
                    continue                      # disjoint -> commutes through
                if gj["op"] == "cz" and set(gj["qubits"]) == pair:
                    close_idx = j
                    break
                if (gj["op"] == "rx" and len(gj["qubits"]) == 1
                        and gj["qubits"][0] in pair and mid_rot is None):
                    mid_rot, mid_idx = gj, j
                    continue
                ok = False                        # anything else blocks the fuse
                break
            if not (ok and close_idx is not None and mid_rot is not None):
                continue
            q = mid_rot["qubits"][0]
            other = b if q == a else a
            rzx = {"op": "rzx", "qubits": [other, q]}     # Z on other, X on q
            if "param" in mid_rot:
                rzx["param"] = mid_rot["param"]
                rzx["coeff"] = float(mid_rot.get("coeff", 1.0))
            else:
                rzx["angle"] = float(mid_rot.get("value", mid_rot.get("angle", 0.0)))
            if g.get("cross_chip") or gates[close_idx].get("cross_chip"):
                rzx["cross_chip"] = True
            for idx in sorted((close_idx, mid_idx, i), reverse=True):
                del gates[idx]
            gates.insert(i, rzx)
            changed = True
            break
    return gates


def save_circuit_diagram(gates, num_qubits, out_png, title):
    """Draw a logical/decomposed gate list as a folded-out Qiskit mpl PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qc = qc_from_logical_gates(gates, num_qubits)
    # fold=-1 -> one continuous row (no wrapping), matching the reference PNGs.
    fig = qc.draw(output="mpl", style=IQP_STYLE, fold=-1)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ----------------------------------------------------------------------
# Molecule case runner
# ----------------------------------------------------------------------
class MoleculeCircuitRunner:
    """Build, simplify and draw the UCCSD-doubles circuit for a named molecule.

    The molecule -> (num_qubits, n_electrons, doubles) registry lives in
    ``CASES`` so re-generating everything is a one-liner:

        MoleculeCircuitRunner().run("HF")          # one molecule
        MoleculeCircuitRunner().run_all()          # HF, Cl2, Br2

    For each molecule it writes, into June_main/circuits2read/:
        <tag>.json / <tag>_circuit.png                 -- bare ansatz (CZ form)
    and prints the gate count.

    The doubles are the shared-creation set (N/2-1, N-1, k+N/2, k) picked by the
    top-k MP2 ranking in the matching UCCSD_Mole notebook (HF: 3, Cl2: 5,
    Br2: 7), listed in the notebooks' op_terms order.
    """

    CASES = {
        "HF": dict(
            molecule="HF", bond_length=1.0, num_qubits=8, n_electrons=6,
            # UCCSD_Mole/HF.ipynb, active_space=(6, 4) -> 8 qubits, top-3 doubles.
            doubles=[(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)],
            pair=False,
        ),
        "Cl2": dict(
            molecule="Cl2", bond_length=1.0, num_qubits=12, n_electrons=10,
            # UCCSD_Mole/Cl2.ipynb, active_space=(10, 6) -> 12 qubits, top-5 doubles
            # (the notebook table shows the top 4; the 5th completes the n_o=5 set).
            doubles=[(5, 11, 8, 2), (5, 11, 7, 1), (5, 11, 9, 3),
                     (5, 11, 10, 4), (5, 11, 6, 0)],
            pair=False,
        ),
        "Br2": dict(
            molecule="Br2", bond_length=1.0, num_qubits=16, n_electrons=14,
            # UCCSD_Mole/Br2.ipynb, active_space=(14, 8) -> 16 qubits, top-7 doubles.
            doubles=[(7, 15, 12, 4), (7, 15, 11, 3), (7, 15, 14, 6),
                     (7, 15, 13, 5), (7, 15, 10, 2), (7, 15, 9, 1),
                     (7, 15, 8, 0)],
            pair=False,
        ),
    }

    def __init__(self, out_dir=None):
        self._dir = Path(__file__).resolve().parent
        if out_dir is None:
            out_dir = self._dir.parent.parent / "June_main" / "circuits2read"
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        import sys
        if str(self._dir.parent) not in sys.path:
            sys.path.insert(0, str(self._dir.parent))
        import uccsd_circuit_io as cio
        self._cio = cio

    @staticmethod
    def _tag(cfg):
        return f"{cfg['molecule']}_{cfg['num_qubits']}q_{len(cfg['doubles'])}doubles"

    def _verify_fusion(self, bare_gates, fused_gates, num_qubits, seed=0):
        """Bind random angles and confirm fused == bare up to global phase
        (statevector evolution from a random state; cheap even at 16 qubits)."""
        from qiskit.quantum_info import random_statevector

        qc_bare = qc_from_logical_gates(bare_gates, num_qubits)
        qc_fused = qc_from_logical_gates(fused_gates, num_qubits)
        # Each circuit rebuilds its own Parameter objects, so bind by name.
        rng = np.random.default_rng(seed)
        by_name = {p.name: float(rng.uniform(-np.pi, np.pi))
                   for p in qc_bare.parameters}
        qc_bare = qc_bare.assign_parameters(
            {p: by_name[p.name] for p in qc_bare.parameters})
        qc_fused = qc_fused.assign_parameters(
            {p: by_name[p.name] for p in qc_fused.parameters})
        sv0 = random_statevector(2 ** num_qubits, seed=seed + 1)
        v1 = sv0.evolve(qc_bare).data
        v2 = sv0.evolve(qc_fused).data
        idx = int(np.argmax(np.abs(v1)))
        phase = v1[idx] / v2[idx]
        if not np.allclose(v1, phase * v2, atol=1e-8):
            raise AssertionError("RZX fusion changed the unitary!")

    def run(self, name):
        """Generate JSON + PNG for one molecule (key of CASES). Returns paths."""
        import matplotlib
        matplotlib.use("Agg")
        cio = self._cio
        cfg = self.CASES[name]
        tag = self._tag(cfg)
        num_qubits = cfg["num_qubits"]
        doubles = cfg["doubles"]
        n_spatial = num_qubits // 2

        # ---- bare ansatz (init_state=None -> no prep gates) ----
        thetas = [Parameter(f"t{d}") for d in range(len(doubles))]
        qc, strings, signs, theta_idx = create_uccsd_circuit(
            num_qubits, doubles, thetas=thetas, optimize=True,
            order="auto", pair=cfg["pair"], init_state=None,
        )
        bare_gates = cio.circuit_to_logical_gates(qc, num_qubits)
        bare_path = self.out_dir / f"{tag}.json"
        cio.save_circuit_json(
            bare_path, molecule=cfg["molecule"], bond_length=cfg["bond_length"],
            num_qubits=num_qubits, n_spatial=n_spatial,
            n_electrons=cfg["n_electrons"], doubles=doubles, signs=signs,
            theta_idx=theta_idx, logical_gates=bare_gates,
            init_state=None, beta=None,
        )
        save_circuit_diagram(bare_gates, num_qubits,
                             self.out_dir / f"{tag}_circuit.png", title=tag)

        n_cz = sum(1 for g in bare_gates if g["op"] == "cz")
        print(
            f"[{tag}] qubits={num_qubits} doubles={len(doubles)} "
            f"params={len(set(theta_idx))}\n"
            f"    bare : {len(bare_gates):3d} gates (CZ={n_cz})\n"
            f"    wrote {bare_path.name} (+ {tag}_circuit.png)"
        )
        return {"bare_json": bare_path}

    def run_all(self):
        return {name: self.run(name) for name in self.CASES}


# ======================================================================
# HARDWARE-AWARE Cl2 build (minimise CROSS-CHIP / high-error 2q gates)
# ======================================================================
# Physical connectivity from the lab drawing: three 2x2 squares wired in a line
# A - B - C, joined by exactly two CROSS-CHIP bridges (the high-error edges).
# Labels below are *physical* qubit indices (the drawing's "example" labelling).
#
#        8 -- 9 ===(bridge)=== 10 -- 11          A = {6,7,8,9}
#        |    |                |     |           B = {4,5,10,11}
#        7 -- 6                4 --- 5           C = {0,1,2,3}
#                             ||(bridge)
#                              3 -- 0
#                              |    |
#                              2 -- 1
#
# Two facts make the naive circuit expensive (and were proven by exhaustive
# search in the _embed_search*.py helpers):
#   * The fermionic Jordan-Wigner doubles carry long Z-strings (weight up to 12),
#     forcing a single linear chain whose busy middle must straddle a bridge.
#   * No relabelling / double-reordering / placement can push the cross-chip
#     count below 16 for that fermionic chain -- it is a structural floor.
#
# So this build switches to the *qubit-excitation* form of the same doubles
# (the hardware-efficient ansatz of arXiv:2212.08006, the paper this generator
# targets): the JW Z-strings are dropped, so each double is a LOCAL 4-qubit
# Pauli rotation on {k, k+6, 5, 11}.  We then:
#   1. keep the shared hub rung (orbitals 5,11) on block B, straddling both
#      bridges, so every satellite block is one hop from the hub;
#   2. move each satellite double's two orbitals to the bridge boundary using
#      only IN-BLOCK swaps (never on a bridge);
#   3. cross a bridge only for that double's parity tree (gather + scatter).
# A cancellation pass then removes the redundant CNOTs at the double interfaces.
#
# Result (verified numerically below): total 2q 58 -> 45 and cross-chip 16 -> 8.

CL2_CHIP_EDGES = [tuple(sorted(e)) for e in [
    (8, 9), (7, 8), (6, 7), (6, 9),       # square A
    (10, 11), (4, 10), (5, 11), (4, 5),   # square B
    (0, 3), (2, 3), (0, 1), (1, 2),       # square C
    (9, 10), (3, 4),                      # CROSS-CHIP bridges (high error)
]]
CL2_BRIDGES = {(9, 10), (3, 4)}

# orbital -> physical qubit.  Hub rung 5/11 sits on the two bridge corners of B;
# rung 4/10 fills B; the other four rungs (k, k+6) live in A and C.
CL2_PLACEMENT = {5: 4, 11: 10, 4: 5, 10: 11,
                 9: 9, 3: 6, 8: 8, 2: 7,        # rungs {3,9},{2,8} -> block A
                 7: 3, 1: 0, 6: 1, 0: 2}        # rungs {1,7},{0,6} -> block C
# emission order of the doubles (index into the doubles list); grouping each
# block's two doubles maximises interface cancellation.
CL2_DOUBLE_ORDER = (1, 4, 2, 0, 3)


def _cl2_local_pauli(double):
    """Z-string-free (qubit-excitation) representative of `double`:
    {physical-orbital : 'X'/'Y'} on exactly the four support orbitals."""
    s = jw_string_for_double(12, double, y_pos=0)
    return {q: c for q, c in enumerate(s) if c in ('X', 'Y')}


class _Cl2HardwareBuilder:
    """Bridge-aware emitter for the qubit-excitation Cl2 ansatz."""

    def __init__(self, placement):
        if nx is None:
            raise ImportError("networkx is required for the hardware-aware build")
        self.G = nx.Graph(CL2_CHIP_EDGES)
        self.G_inblock = nx.Graph([e for e in CL2_CHIP_EDGES
                                   if e not in CL2_BRIDGES])
        self.pos = dict(placement)                       # orbital -> phys
        self.inv = {p: o for o, p in placement.items()}  # phys -> orbital
        self.qc = QuantumCircuit(12)
        self._swaps = []                                 # for layout restore

    def _swap(self, p, q):
        self.qc.swap(p, q)
        o1, o2 = self.inv[p], self.inv[q]
        self.inv[p], self.inv[q] = o2, o1
        self.pos[o1], self.pos[o2] = q, p
        self._swaps.append((p, q))

    def restore_layout(self):
        """Undo all swaps so each orbital ends on its initial physical qubit;
        the interface-cancellation pass removes these again for free."""
        for p, q in reversed(self._swaps):
            self._swap(p, q)
        self._swaps.clear()

    def _move(self, orb, target):
        """Slide `orb` to physical `target` along in-block (non-bridge) edges."""
        for nxt in nx.shortest_path(self.G_inblock, self.pos[orb], target)[1:]:
            self._swap(self.pos[orb], nxt)

    def _connect(self, orbs):
        """Make the physical positions of `orbs` a connected chip subgraph using
        only in-block swaps (the hub rung 5/11 is the fixed anchor)."""
        cluster = {self.pos[5], self.pos[11]}
        for orb in orbs:
            if orb in (5, 11) or self.pos[orb] in cluster:
                continue
            cands = [n for c in cluster for n in self.G.neighbors(c)
                     if n not in cluster
                     and nx.has_path(self.G_inblock, self.pos[orb], n)]
            target = min(cands, key=lambda t: nx.shortest_path_length(
                self.G_inblock, self.pos[orb], t))
            self._move(orb, target)
            cluster.add(self.pos[orb])

    def rotation(self, double, theta, pauli_evo=False):
        letters = _cl2_local_pauli(double)
        self._connect(list(letters))
        sup = {self.pos[o]: L for o, L in letters.items()}
        sub = self.G.subgraph(sup)
        assert nx.is_connected(sub), (double, list(sub.nodes))
        if pauli_evo:                              # exact reference block
            from qiskit.circuit.library import PauliEvolutionGate
            from qiskit.quantum_info import Pauli
            lbl = ['I'] * 12
            for p, L in sup.items():
                lbl[p] = L
            self.qc.append(PauliEvolutionGate(Pauli(''.join(reversed(lbl))),
                                              time=theta / 2), range(12))
            return
        root = self.pos[5]
        tree = nx.bfs_tree(sub, root)
        leaves_first = [c for c in nx.dfs_postorder_nodes(tree, root) if c != root]
        gather = [(c, next(tree.predecessors(c))) for c in leaves_first]
        for p, L in sup.items():                  # basis change letter -> Z
            if L == 'X':
                self.qc.h(p)
            else:
                self.qc.sdg(p); self.qc.h(p)
        for c, par in gather:
            self.qc.cx(c, par)
        self.qc.rz(theta, root)                    # exp(-i theta/2 Z_root)
        for c, par in reversed(gather):
            self.qc.cx(c, par)
        for p, L in sup.items():                   # undo basis change
            if L == 'X':
                self.qc.h(p)
            else:
                self.qc.h(p); self.qc.s(p)


def _cl2_angle_info(angle):
    """(param_name, coeff) for a symbolic angle, or (None, value) if numeric."""
    try:
        return None, float(angle)
    except (TypeError, RuntimeError):
        pass
    params = list(getattr(angle, "parameters", []))
    if len(params) != 1:
        raise ValueError(f"unsupported angle {angle!r}")
    p = params[0]
    return p.name, float(angle.bind({p: 1.0})) - float(angle.bind({p: 0.0}))


def cl2_hardware_logical_gates(qc):
    """Flatten the hardware circuit into a JSON-friendly gate list; every 2q gate
    carries a picture-based ``cross_chip`` flag (True on the high-error bridges)."""
    gates = []
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(w).index for w in inst.qubits]
        if name == "barrier":
            continue
        if len(qs) == 2:
            entry = {"op": name, "qubits": qs,
                     "cross_chip": tuple(sorted(qs)) in CL2_BRIDGES}
        elif name in ("rx", "ry", "rz"):
            pname, coeff = _cl2_angle_info(inst.operation.params[0])
            entry = {"op": name, "qubits": qs}
            if pname is None:
                entry["value"] = coeff
            else:
                entry["param"], entry["coeff"] = pname, coeff
        else:                                     # h, x, s, sdg, ...
            entry = {"op": name, "qubits": qs}
        gates.append(entry)
    return gates


def save_cl2_hardware_json(path, qc, info, doubles):
    """Write the hardware-aware Cl2 circuit as a logical-gate JSON (same schema
    as the rest of the pipeline, with bridge-based cross_chip flags)."""
    import json
    gates = cl2_hardware_logical_gates(qc)
    param_names = sorted({g["param"] for g in gates if "param" in g},
                         key=lambda s: (len(s), s))
    payload = {
        "molecule": "Cl2",
        "bond_length": 1.0,
        "num_qubits": 12,
        "n_spatial": 6,
        "n_electrons": 10,
        "ansatz": "qubit_excitation",
        "init_state": None,
        "doubles": [list(map(int, d)) for d in doubles],
        "signs": [1.0] * len(doubles),
        "theta_idx": list(range(len(doubles))),
        "param_names": param_names,
        "chip_edges": [list(e) for e in CL2_CHIP_EDGES],
        "cross_chip_bridges": [list(e) for e in sorted(CL2_BRIDGES)],
        "initial_layout": {str(o): p for o, p in sorted(CL2_PLACEMENT.items())},
        "final_layout": {str(o): p for o, p in info["final_layout"].items()},
        "total_2q": info["total_2q"],
        "cross_chip_2q": info["cross_chip"],
        "edge_load": {f"{a}_{b}": c for (a, b), c in info["edge_load"].items()},
        "gates": gates,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _cl2_counts(qc):
    from collections import Counter
    load = Counter()
    for inst in qc.data:
        if inst.operation.num_qubits == 2:
            p, q = (qc.find_bit(w).index for w in inst.qubits)
            load[tuple(sorted((p, q)))] += (3 if inst.operation.name == "swap" else 1)
    return load


def build_cl2_hardware_circuit(doubles, thetas=None, *, placement=None,
                               order=CL2_DOUBLE_ORDER, cancel=True, restore=False):
    """Hardware-aware qubit-excitation circuit for the Cl2 doubles.

    Returns (QuantumCircuit, info) with info['total_2q'], info['cross_chip'],
    info['edge_load'] and info['final_layout'] (orbital -> physical qubit at the
    END of the circuit).  All 2q gates lie on real chip edges -- no SWAP routing
    is ever required.

    restore=False (default): leave the cheap in-block swaps un-undone; the ansatz
        ends on a permuted layout (use info['final_layout'] at read-out).  This
        is the low-count form (total 2q ~49, cross-chip 8).
    restore=True: append inverse swaps so each orbital returns to its initial
        qubit (exact product, fixed layout) at the cost of more 2q gates.
    """
    if placement is None:
        placement = CL2_PLACEMENT
    if thetas is None:
        thetas = [Parameter(f"t{d}") for d in range(len(doubles))]
    b = _Cl2HardwareBuilder(placement)
    for i in order:
        b.rotation(doubles[i], thetas[i])
    if restore:
        b.restore_layout()
    qc = b.qc
    if cancel:
        from qiskit import transpile
        bg = ["cx", "rz", "rx", "ry", "h", "s", "sdg", "x"]
        # decompose SWAPs to CX first so the interface CNOTs can cancel, then a
        # correctness-preserving peephole/commutation pass.  (optimization_level
        # 3's block re-synthesis is avoided: it is lossy for wide circuits in
        # this qiskit version -- see verify_cl2_hardware.)
        qc = transpile(qc, basis_gates=bg, optimization_level=0, seed_transpiler=0)
        qc = transpile(qc, basis_gates=bg, optimization_level=2, seed_transpiler=0)
    load = _cl2_counts(qc)
    info = {
        "total_2q": int(sum(load.values())),
        "cross_chip": int(sum(c for e, c in load.items() if e in CL2_BRIDGES)),
        "edge_load": dict(sorted(load.items())),
        "final_layout": dict(sorted(b.pos.items())),
    }
    return qc, info


def verify_cl2_hardware(doubles, *, placement=None, order=CL2_DOUBLE_ORDER,
                        restore=False, seed=0, atol=1e-7):
    """Numerically confirm the emitted hardware circuit implements the intended
    product of qubit-excitation rotations  prod_k exp(-i theta_k/2 P_local_k).

    The reference is built with the *same* swap schedule but with each block
    replaced by an exact PauliEvolutionGate, so it ends on the same (possibly
    permuted) layout and is directly comparable on a random input state."""
    from qiskit.quantum_info import Statevector
    if placement is None:
        placement = CL2_PLACEMENT
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, size=len(doubles))
    qc, _ = build_cl2_hardware_circuit(
        doubles, thetas=list(th), placement=placement, order=order,
        cancel=True, restore=restore)
    rb = _Cl2HardwareBuilder(placement)           # exact reference, same swaps
    for i in order:
        rb.rotation(doubles[i], th[i], pauli_evo=True)
    if restore:
        rb.restore_layout()
    ref = rb.qc
    vec = rng.standard_normal(2 ** 12) + 1j * rng.standard_normal(2 ** 12)
    sv = Statevector(vec / np.linalg.norm(vec))
    v1, v2 = sv.evolve(qc).data, sv.evolve(ref).data
    idx = int(np.argmax(np.abs(v2)))
    return bool(np.allclose(v1 * (v2[idx] / v1[idx]), v2, atol=atol))


def report_cl2_hardware(out_dir=None):
    """Build + verify the hardware-aware Cl2 circuit, print the comparison vs the
    connectivity-agnostic fermionic baseline, and (if out_dir given) save a PNG
    diagram next to the pipeline's other circuits."""
    from collections import Counter
    doubles = MoleculeCircuitRunner.CASES["Cl2"]["doubles"]

    # baseline (current file): fermionic chain, identity placement orbital i -> i
    qc0, *_ = create_uccsd_circuit(12, doubles, optimize=True, order="auto",
                                   pair=False, init_state=None)
    base = Counter()
    for inst in qc0.data:
        if inst.operation.name == "cz":
            a, b = (qc0.find_bit(w).index for w in inst.qubits)
            base[tuple(sorted((a, b)))] += 1
    base_total = sum(base.values())
    base_cross = sum(c for e, c in base.items() if e in CL2_BRIDGES)

    ok = verify_cl2_hardware(doubles)
    qc, info = build_cl2_hardware_circuit(doubles)
    print("Cl2 cross-chip optimisation (chip = three 2x2 squares A-B-C in a line)")
    print(f"  baseline (fermionic chain): total 2q = {base_total:3d}, "
          f"cross-chip = {base_cross}")
    print(f"  hardware-aware (qubit-exc): total 2q = {info['total_2q']:3d}, "
          f"cross-chip = {info['cross_chip']}   verified={ok}")
    print(f"  bridge load: " + ", ".join(
        f"{e}:{info['edge_load'].get(e, 0)}" for e in sorted(CL2_BRIDGES)))
    print(f"  final orbital->physical layout (read-out map): {info['final_layout']}")

    if out_dir is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        png = out / "Cl2_12q_5doubles_hardware_circuit.png"
        fig = qc.draw(output="mpl", style=IQP_STYLE, fold=-1)
        fig.suptitle(f"Cl2 hardware-aware (qubit-exc): {info['total_2q']} 2q, "
                     f"{info['cross_chip']} cross-chip", fontsize=14)
        fig.savefig(png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        json_path = save_cl2_hardware_json(
            out / "Cl2_12q_5doubles_hardware.json", qc, info, doubles)
        print(f"  wrote {png.name} + {json_path.name}")
    return qc, info


# Run this file directly to (re)generate HF / Cl2 / Br2 circuits and diagrams.
if __name__ == "__main__":
    runner = MoleculeCircuitRunner()
    runner.run_all()
    print()
    report_cl2_hardware(out_dir=runner.out_dir)
