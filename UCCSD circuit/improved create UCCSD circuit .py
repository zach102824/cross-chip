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
# Generation pipeline
# ----------------------------------------------------------------------
# Run this file directly to turn the doubles you picked (in CASES below) into:
#   1.  <tag>.json                              -- bare ansatz, NO init-state prep
#   2.  <tag>_circuit.png                       -- its Qiskit diagram
#   3.  <tag>_decomposed_simplified.json        -- cross-chip CZ -> RZX + peephole
#   4.  <tag>_decomposed_simplified_circuit.png -- its Qiskit diagram
# All four files are written to June_main/circuits2read/.
if __name__ == "__main__":
    import importlib.util
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    import uccsd_circuit_io as cio

    # "simplify decomposed circuit.py" has spaces in its name -> load by path.
    _spec = importlib.util.spec_from_file_location(
        "simplify_decomposed_circuit", str(_THIS_DIR / "simplify decomposed circuit.py")
    )
    simplify = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(simplify)

    OUT_DIR = _THIS_DIR.parent / "June_main" / "circuits2read"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # INPUT: the doubles to simulate (pick them in the UCCSD_Mole notebook,
    # then paste here).  NO initial-state prep is added, so the diagrams show
    # the bare ansatz -- no X / Ry / CX prep layer.
    #   tag           : output filename stem (also the diagram title)
    #   num_qubits    : spin-up on 0..N/2-1, spin-down on N/2..N-1
    #   doubles       : list of (k, l, i, j) for a_k^dag a_l^dag a_i a_j
    #   pair          : True -> exact two-string-per-double (Fig.12 style)
    #   molecule/bond_length/n_electrons : metadata only (no prep is built)
    # ------------------------------------------------------------------
    CASES = {
        "F2_12q_5doubles": dict(
            molecule="F2",
            bond_length=1.0,
            num_qubits=12,
            n_electrons=10,
            # Top-5 MP2-ranked doubles from UCCSD_Mole/F2.ipynb (active_space=(10, 6)).
            # Same set as the notebook's op_terms, listed in the paper's nested order
            # (a11^ a5^ a_{k+6} a_k, k=0..4) so the shared a11^a5^ subtree cancels and
            # the circuit reaches the published 50 CZ (Fig. 14).
            doubles=[(11, 5, k + 6, k) for k in range(5)],
            pair=False,
        ),
        # "br2_16q_3doubles": dict(
        #     molecule="Br2",
        #     bond_length=2.3,
        #     num_qubits=16,
        #     n_electrons=14,
        #     doubles=[(7, 15, 12, 4), (7, 15, 11, 3), (7, 15, 14, 6)],
        #     pair=False,
        # ),
    }

    for tag, cfg in CASES.items():
        num_qubits = cfg["num_qubits"]
        doubles = cfg["doubles"]
        n_spatial = num_qubits // 2

        # ---- Step 2: bare ansatz (init_state=None -> no prep gates) ----
        thetas = [Parameter(f"t{d}") for d in range(len(doubles))]
        qc, strings, signs, theta_idx = create_uccsd_circuit(
            num_qubits,
            doubles,
            thetas=thetas,
            optimize=True,
            order="auto",
            pair=cfg["pair"],
            init_state=None,
        )
        logical_gates = cio.circuit_to_logical_gates(qc, num_qubits)
        logical_path = OUT_DIR / f"{tag}.json"
        cio.save_circuit_json(
            logical_path,
            molecule=cfg["molecule"],
            bond_length=cfg["bond_length"],
            num_qubits=num_qubits,
            n_spatial=n_spatial,
            n_electrons=cfg["n_electrons"],
            doubles=doubles,
            signs=signs,
            theta_idx=theta_idx,
            logical_gates=logical_gates,
            init_state=None,
            beta=None,
        )
        n_cz = sum(1 for g in logical_gates if g["op"] == "cz")
        n_cross = sum(1 for g in logical_gates if g["op"] == "cz" and g["cross_chip"])
        print(
            f"[{tag}] logical: {logical_path.name}  "
            f"(qubits={num_qubits}, CZ={n_cz}, cross-chip CZ={n_cross}, "
            f"params={len(set(theta_idx))})"
        )
        logical_png = save_circuit_diagram(
            logical_gates, num_qubits, OUT_DIR / f"{tag}_circuit.png", title=tag
        )
        print(f"[{tag}] logical diagram: {logical_png.name}")

        # ---- Step 3: cross-chip CZ -> RZX + peephole, then diagram ----
        decomposed_path = simplify.simplify_file(logical_path)
        dec_data = cio.load_circuit_json(decomposed_path)
        dec_png = save_circuit_diagram(
            dec_data["gates"],
            num_qubits,
            OUT_DIR / f"{decomposed_path.stem}_circuit.png",
            title=f"{tag} (cross-chip CZ -> RZX)",
        )
        print(f"[{tag}] decomposed diagram: {dec_png.name}")
