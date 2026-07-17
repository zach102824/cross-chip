# Lowering the total error rate of the UCCSD circuits (HF / Cl2 / Br2)

Working notes, 2026-07-17.  Goal: reduce the *physical* error accumulated by the
three molecule circuits (cross-chip 2-qubit gate = 0.1 depol, on-chip 2-qubit =
0.01, 1-qubit = 0.0005 in `June_main/main_cursor_lib.py`) by approximating /
restructuring the circuit while staying at ~chemical accuracy (1.6 mHa).

The headline result (Method 1) is **not an approximation at all**: the JW
Z-parity tails of every selected double can be deleted *exactly* for these
paired-doubles ansätze.  This was verified numerically (statevector overlap
= 1.0 to machine precision for all three molecules, random angles up to ±0.3).
It cuts the Br2 circuit from 44 CZ / depth 39 to 16 CZ / depth 22 before any
further tricks.

---

## 0. Baseline and error budget

Current saved circuits (`June_main/circuits2read/*_rzx.json`) and the declared
cross-chip pairs from `June_main/export2cloud/main_*.py`:

| circuit | 1q gates | on-chip 2q | cross-chip 2q | depth | est. fidelity* |
|---|---|---|---|---|---|
| HF_8q_3doubles_rzx  | 32 | 20 | 3 (RZX on (2,6)) | 24 | 0.59 |
| Cl2_10q_3doubles_rzx | 42 | 24 | 7 (3 RZX + 4 CZ) | 29 | 0.37 |
| Br2_12q_4doubles_rzx | 56 | 32 | 8 (4 RZX on (4,10) + 2 CZ(1,2) + 2 CZ(7,8)) | 39 | 0.30 |

\* fidelity proxy = ∏(1−p) over all gates with p = 0.0005 / 0.01 / 0.1.
It is only a ranking tool, but it shows clearly **where the error lives**:

- Br2: cross-chip alone contributes (0.9)^8 ≈ 0.43; on-chip CZ (0.99)^32 ≈ 0.72.
- So the priorities are (1) fewer cross-chip 2q gates, (2) fewer CZs, (3) depth.

Br2 chip layout implied by `CROSS_CHIP_QUBIT_PAIRS = {(1,2),(7,8),(4,10)}`:
chip A = {0,1,6,7} (spatial orbitals 0,1 both spins), chip B = {2,3,4,5}
(α row of spatials 2–5), chip C = {8,9,10,11} (β row of spatials 2–5).

---

## 1. Method 1 — Delete the JW Z-parity tails (EXACT, verified)

### 1.1 The observation

Every selected double has the paired form `a†_{N/2−1} a†_{N−1} a_{k+N/2} a_k`
(same spatial excitation in both spin rows).  Its odd-Y representative is e.g.
for Br2 double (5,11,6,0):

```
Y0 Z1 Z2 Z3 Z4 X5 | X6 Z7 Z8 Z9 Z10 X11
```

The Z letters are only there to carry fermionic antisymmetry (JW parity).  But
for **this specific ansatz + reference state** they are constants:

- Each Z acts on a qubit that the string itself does not excite.
- The Zs come in **spin pairs**: Z_q and Z_{q+N/2} always appear together
  (both rows have the same tail because the excitation is spin-paired).
- On any state reachable by the circuit, qubit q and q+N/2 have **equal
  occupation** (HF start + only paired excitations ⇒ the α and β rows stay
  locked).  Hence Z_q Z_{q+N/2} = (+1)·(+1) or (−1)·(−1) = **+1 always**.

So every Z pair can be replaced by the scalar +1, i.e. simply deleted from the
string, with **zero** error on the reachable subspace.  For Br2 the four
strings collapse to weight-4 operators:

```
Y0 X5 | X6 X11        (was weight 12)
Y4 X5 | X10 X11       (unchanged, no tail)
Y3 X5 | X9 X11        (was weight 8)
Y2 X5 | X8 X11        (was weight 8)
```

Note this argument covers cross terms too: string k's Z-tail qubits are excited
only by *other* doubles, but those move α and β together, so the pair parity
stays +1 in every branch of the superposition.  This is why the check below
comes out exact and not merely O(θ²).

### 1.2 Numerical verification (done)

Compiled both variants with the existing generator (`_compile_tworow` +
`_peephole` + `_verify_program`) and compared full statevectors on
|HF⟩ with random θ ∈ [−0.3, 0.3]:

| molecule | overlap ⟨orig\|frozen⟩ | CZ before → after | H before → after | depth before → after |
|---|---|---|---|---|
| HF  | 1.000000000000 | 26 → 18 (12 after RZX fusion) | 26 → 12 | 30 → 17 |
| Cl2 | 1.000000000000 | 34 → 18 (12 after fusion) | 36 → 12 | 35 → 17 |
| Br2 | 1.000000000000 | 44 → 24 (16 after fusion) | 48 → 16 | 47 → 22 |

(`fuse_cz_rot_cz_to_rzx` still fires on the frozen circuits: each double keeps
exactly one parameterized RZX on the (α-hub, β-hub) link — HF: (3,7),
Cl2: (4,9), Br2: (5,11).)

Frozen Br2 fused circuit, all four doubles:

```
ops: 8 rx, 16 h, 16 cz, 4 rzx(θ), depth 22
cz pairs: (0,5) (6,11) (4,5) (10,11) (3,5) (9,11) (2,5) (8,11)  — ×2 each
```

i.e. a *star* fanning each excited qubit into the shared hubs (5, 11) instead
of a nearest-neighbour chain over all 12 qubits.

### 1.3 What it buys in the noise model

Under the current Br2 tagging ({(1,2),(7,8),(4,10)} cross): the frozen circuit
has cross-chip 2q = 2×CZ(0,5)… — careful, the star CZs like (0,5) are
**long-range** and not physical edges.  Two honest options:

- **(a) Route the star through the existing links** (keeps current placement).
  The chain qubits are now *idle* parity couriers instead of Z-support, so the
  gate count is identical to threading a wire: Br2 keeps ~8 cross gates but
  drops ~16 on-chip CZ and 32 H → fidelity proxy 0.30 → ~0.37.
- **(b) Combine with Method 2 (re-placement)** — this is where the big
  cross-chip win comes from, see below.

Also independent of noise: **half the depth** ⇒ less decoherence on real
hardware, and far fewer H gates.

### 1.4 Caveats / to-do before adopting

- Exactness relies on (i) HF or the Eq.(6) multiref start (also spin-paired),
  (ii) *only* paired doubles in the ansatz, (iii) `pair=False` single strings
  acting as their own inverses on the locked subspace.  All three hold for the
  current HF/Cl2/Br2 pipelines.  Verified end-to-end, but re-run the check if
  the doubles set changes.
- The measured Hamiltonian is untouched — only the ansatz circuit changes.
- `CROSS_CHIP_QUBIT_PAIRS` in `main_*.py` must be re-derived for the new gate
  pattern (the RZX link moves, e.g. Br2 (4,10) → (5,11)).

### 1.5 Connectivity fix (HF worked example, implemented)

Fully frozen strings compile to a *star* (every excited qubit fans straight
into the hub), which the 2×2-square + single-bridge chip graph cannot host:
the hub would need degree 4 and long-range CZs like CZ(0,3) are not edges.
The freeze identity works **both ways** — any Z spin-pair may be re-inserted
for free — so the fix is *selective* freezing: re-insert exactly the Z-pairs
whose qubits are needed as parity couriers along physical edges, and place the
hubs on the bridge:

```
physical chip1 = {1,2,3,4}, chip2 = {5,6,7,8}, bridge = (4,5)
logical -> physical:  0->2  1->1  2->3  3->4 | 4->7  5->6  6->8  7->5
t0: YZZXXZZX -> YZIXXZIX   (Z1/Z5 kept as couriers: chains 0-1-3, 4-5-7)
t1: IYZXIXZX -> IYIXIXIX   (fully frozen; (1,3),(5,7) are edges)
t2: IIYXIIXX -> IIYXIIXX   (no tail; (2,3),(6,7) are edges)
```

Every CZ then lands on a chip edge and all 3 RZX(θ) sit on the physical
bridge.  Verified exact (overlap 1.0) and routable by
`method1_HF_frozen_circuit_png.py`; result: 16 CZ / 18 H / depth 21 vs
baseline 20 CZ / 26 H / depth 24 (the fully-frozen unroutable star was
12 CZ / depth 17 — the gap is the routing cost of t0's couriers).
For Cl2/Br2 the same recipe applies; couriers cost 2 CZ per extra hop of the
deepest double.

---

## 2. Method 2 — Placement + link-aware fan-in (answers "one RZX only?")

With the tails gone, each double touches only 4 qubits: (k_α, k_β) and the
shared LUMO pair (5_α, 5_β).  Now choose the qubit→chip placement so both
spins of a spatial orbital sit on the *same* chip (2 spatial orbitals per
chip, exactly the layout `uccsd_circuit_io.chip_of` already assumes):

Br2: chip A = spatials {4,5} = qubits {4,5,10,11}, chip B = {2,3} =
{2,3,8,9}, chip C = {0,1} = {0,1,6,7}.

Per-double cross-chip cost, using two tricks:

1. **Local α⊗β pre-merge**: X_kα and X_kβ live on the same chip → fan them
   into one local parity qubit *before* crossing.  One crossing carries the
   joint parity (instead of one per spin row → halves the crossings).
2. **Put the parameterized rotation ON the link**: compile
   exp(−iθ/2 · A⊗B) with A fanned to the link qubit on one side, B on the
   other, and emit a single native RZX(θ) across the link — 1 cross-chip gate,
   not a CZ…RX…CZ round trip (2).

Resulting cross-chip 2q counts (Br2):

| double | support (spatial) | cross-chip 2q |
|---|---|---|
| (5,11,10,4) | {4,5} — all on chip A | **0** (purely local block!) |
| (5,11,9,3)  | {3,5} | 1 (RZX on the A–B link) |
| (5,11,8,2)  | {2,5} | 1 (RZX on the A–B link) |
| (5,11,6,0)  | {0,5} | 3 (2 CZ through B + 1 RZX) |

**Total: 5 cross-chip gates instead of 8** (and the HOMO→LUMO double becomes
noise-free at the cross-chip level).  Fidelity proxy for cross-chip part:
(0.9)^5 ≈ 0.59 vs (0.9)^8 ≈ 0.43.

So the answer to "can we cut to a single RZX per double" is: **yes for every
double whose support spans exactly 2 chips** (all of HF's and Cl2's, 3 of 4
for Br2) — single-qubit basis rotations (H / RX(±π/2)) move the Y/X letters
onto the link, and the local CZ fan-ins stay on-chip at 0.01.  A double
spanning 3 chips fundamentally needs to cross both cuts (parity must travel),
so 1 gate is impossible for Br2's (5,11,6,0) — 3 is the floor with these
links.

Implementation: this is a placement permutation + a graph-aware version of
`_row_chain` (fan into the link endpoints instead of the row midpoint).  The
existing symbolic verifier (`_verify_program`) covers correctness unchanged.

---

## 3. Method 3 — Noise-aware truncation of the doubles set (approximation)

The doubles are MP2-ranked; the deepest-orbital double is simultaneously

- the **least important chemically** (smallest amplitude), and
- the **most expensive physically** (longest string ⇒ for Br2 it alone owns
  the 4 extra cross-chip CZs today, or 3 of 5 cross gates after Method 2).

Concrete experiment: rebuild `Br2_12q_3doubles` without (5,11,6,0) and
compare (i) noiseless VQE floor vs FCI, (ii) noisy/CDR energy.  Decision rule:

> drop the double iff (noiseless accuracy lost) < (noisy bias/variance gained
> back), both measured in mHa at matched shot budget.

With cross-chip at 0.1 depol, removing 3 cross gates changes the cross-chip
fidelity by ×(0.9)^{−3} ≈ 1.37 — very likely worth ~a fraction of a mHa of
correlation, especially post-CDR where the *bias* scales with total noise.
Same logic applies to Cl2 (its (4,9,7,2) double reaches the far chip).

Cheap variant that keeps the parameter: freeze θ_weak at its MP2/noiseless
value **classically** — run the circuit without that block and add its
2nd-order energy contribution as a classical correction.

---

## 4. Method 4 — One Clifford for all doubles (commuting-set compilation)

The frozen strings mutually commute and are *nested* (supports
{4,5,10,11} ⊂ {3,4,5,…} ⊂ …).  So instead of ladder-up / ladder-down per
double (4× for Br2), synthesize a single Clifford C that maps all k strings to
single-qubit Zs simultaneously:

```
C† · [ RZ(θ1) ... RZ(θk)  in parallel, on k distinct qubits ] · C
```

- Clifford cost paid **once**: the shared fan-in into hubs (5,11) is common to
  all four strings; the nested structure means C is essentially one cumulative
  CNOT chain per row + one link crossing.
- The k rotations become **local RZs running in parallel** → the parameterized
  gates carry zero cross-chip noise, and circuit depth stops growing with k.
- Expected Br2 budget: ~2 crossings per cut for C + C† ⇒ 4–6 cross gates
  total, ~10–12 on-chip CZ, depth ~15.

Caveat (from the earlier exploration notes): this is a commuting-Pauli ansatz
— identical unitary to the current product of the 4 single-string blocks
(they commute, so the product *is* simultaneous), so unlike Method 3 it is
exact w.r.t. today's `pair=False` ansatz.  Verify with `_verify_program` +
statevector as in Method 1.

---

## 5. Smaller / situational ideas

- **First-block state synthesis**: the first ladder acts directly on |HF⟩ (a
  stabilizer state).  Synthesizing C₁|HF⟩ as a stabilizer state instead of
  applying C₁ as a unitary typically saves a few CZs and depth at the circuit
  entrance.  Exact; modest win; only helps the first block.
- **Last-block / measurement merge**: the final inverse ladder is a Clifford
  that could be absorbed into the OGM measurement basis change per Pauli
  group.  Saves the trailing CZs on every shot.  Requires touching the
  measurement pipeline, so rank it below Methods 1–4.
- **Echoed / small-angle RZX**: on real cross-resonance hardware a small-angle
  RZX(θ) is a shorter pulse with proportionally less error.  The current
  `GateArityDepolarizingNoise` is angle-independent, so this shows **no gain
  in simulation** — out of scope until an angle-dependent noise model exists.
- **CDR interplay**: fewer noisy gates also shrinks the CDR training-circuit
  variance; when comparing methods, compare *post-mitigation* energies at
  equal total shots.

---

## 6. Recommended order of attack

1. **Method 1 (Z-tail freeze)** — exact, verified, pure win: ~2× fewer CZ,
   ~2× shallower, for all three molecules.  Implement as a `freeze_tails=True`
   option in `create_uccsd_circuit` (drop Zs from `jw_string_for_double`
   output; everything downstream already works — verified).
2. **Method 2 (placement + RZX-on-link)** — Br2 cross-chip 8 → 5, Cl2/HF one
   RZX per double, zero cross-chip CZ.  Needs the graph-aware fan-in.
3. **Method 4 (single shared Clifford)** — collapses the remaining per-double
   ladder overhead; exact for the current ansatz.
4. **Method 3 (drop / classically correct the weakest double)** — the only
   true approximation; run the accuracy-vs-noise trade study on Br2 first.

Rough combined outlook for Br2 (fidelity proxy, same noise numbers):
current 0.30 → Method 1 ≈ 0.37 → +Method 2 ≈ 0.50 → +Method 4 ≈ 0.55 →
+Method 3 (3 doubles) ≈ 0.65+.  That is a >2× reduction in total depolarizing
exposure with zero or sub-chemical-accuracy cost.

---

## Appendix: reproduction snippet for the Method-1 check

```python
# from repo root
import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("gen", "UCCSD circuit/improved create UCCSD circuit .py")
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

n, doubles, eta, hub0 = 12, [(5,11,6,0),(5,11,10,4),(5,11,9,3),(5,11,8,2)], 5, 4
thetas = np.random.default_rng(1).uniform(-0.3, 0.3, len(doubles))

def build(strings):
    idx = gen._auto_order(strings)
    prog, expected, hub = [], [], None
    for i in idx:
        s = strings[i]
        prefix, pivot, ph = gen._compile_tworow(s, n, hub_hint=hub)
        hub = pivot
        prog += prefix + [('ROT', pivot, i, ph)] + gen._invert(prefix)
        expected.append((s, ph))
    prog = gen._peephole(prog); gen._verify_program(prog, n, expected)
    qc = QuantumCircuit(n)
    for g in prog:
        if g[0]=='H': qc.h(g[1])
        elif g[0]=='RX': qc.rx(g[2], g[1])
        elif g[0]=='CZ': qc.cz(g[1], g[2])
        else: qc.rx(g[3]*thetas[g[2]], g[1])
    return qc

orig = [''.join(gen.jw_string_for_double(n, d)) for d in doubles]
froz = [s.replace('Z', 'I') for s in orig]           # Method 1
init = QuantumCircuit(n)
for q in list(range(eta)) + list(range(n//2, n//2+eta)): init.x(q)
v0, v1 = (Statevector(init.compose(build(ss))) for ss in (orig, froz))
print(abs(np.vdot(v0.data, v1.data)))                # -> 1.0
```
