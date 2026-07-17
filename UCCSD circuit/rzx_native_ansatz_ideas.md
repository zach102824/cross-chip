# Hardware-efficient, UCCSD-inspired ansätze with native cross-chip RZX

Proposal / exploration notes. Goal: design circuits that reach the accuracy of our
reduced-doubles UCCSD (HF / Cl2 / Br2 pipelines) with **far fewer two-qubit gates**,
by treating the continuous-angle cross-chip `RZX(θ)` as a *first-class native gate*
instead of something CZs get decomposed into.

---

## 1. Setting

### 1.1 Hardware

![qubit connectivity](../qubit_connectivity/qubit_connectivity.png)

- Four chips of 4 qubits each (2×2 squares): {1,2,3,4}, {5,6,7,8}, {9,10,11,12}, {13,14,15,16}.
- Three **cross-chip links**: 4–5 (top row), 7–9 (bottom row), 11–13 (top row).
  Note they alternate top / bottom / top.
- On the cross-chip links the native two-qubit gate is `RZX(θ) = exp(-i θ/2 · Z⊗X)`
  with a **continuous angle**. This is unusual: most platforms only expose fixed
  entanglers, so a θ-dependent two-qubit interaction normally costs *two* fixed
  gates (`CZ · RX(θ) · CZ`). Here it costs **one**, and (cross-resonance-style)
  a small-angle RZX is a *shorter pulse*, so error likely scales with |θ| —
  small-amplitude doubles are extra cheap in noise.

### 1.2 What we already have (baseline to beat)

`improved create UCCSD circuit .py` compiles each double as a CZ/H hub-ladder around
one `RX(θ)` pivot, with hub continuity + peephole cancellation, then
`fuse_cz_rot_cz_to_rzx` fuses the innermost `CZ · RX(θ) · CZ` into one `RZX(θ)`.
Current saved circuits (`June_main/circuits2read/`):

| circuit | qubits | doubles | CZ | param. RZX |
|---|---|---|---|---|
| HF_8q_3doubles_rzx | 8 | 3 | 20 | 3 |
| Cl2_10q_3doubles_rzx | 10 | 3 | 28 | 3 |
| Br2_12q_4doubles_rzx | 12 | 4 | 36 | 4 |

### 1.3 The physics we can exploit

From `UCCSD_Mole/HF.ipynb` (and the Cl2/Br2 analogues):

- The reduced-ansatz convergence scan shows **one paired HOMO→LUMO double already
  captures most of the correlation**; 2–3 doubles reach ~FCI in these CAS spaces.
  We only go to 3–4 doubles so the ansatz *touches all qubits*.
- Equivalently: the FCI vector is dominated by a **handful of determinants**
  (|HF⟩ plus a few paired doubles). The target state is *sparse*.
- All selected doubles share the same creation pair `(N/2−1, N−1)`
  (e.g. Br2: `a5† a11† aj ai`), i.e. they all promote into the same LUMO pair.

So the true question is not "how do we compile UCCSD cheaply" but
**"what is the cheapest native-gate circuit whose image contains this sparse,
number-conserving manifold?"** Several directions below attack this.

---

## 2. Idea A — Pivot-on-the-link compilation (lowest risk, extends current code)

**Claim:** with the right orbital→qubit placement, the *one* θ-carrying gate of every
double can be made the *cross-chip* gate, and everything else stays on-chip.

- The fused `RZX(θ)` always sits on the (pivot, β-hub) pair of the ladder. Today the
  pivot is chosen by `hub0` / midpoint heuristics with no knowledge of the chip graph.
- Change `_compile_tworow` to take the hardware edge list and pick
  `(pivot, hub_b)` to be a **cross-chip edge** whenever the string's support spans
  chips (it always does, since the shared creation pair (N/2−1, N−1) sits on the last
  chip while the annihilated orbitals walk across the others).
- Then re-route the CZ fan-in chains along actual graph paths (currently they assume
  a line `q, q+1`; the real graph is 2×2 squares chained by single links, so the
  chains need SWAP-free routing along the squares — the 2-row layout already matches
  if we map α on the top physical row and β on the bottom).
- **Placement co-design:** the mapping `spin orbital → physical qubit` is a free
  discrete variable (8! per row, but heavily constrained). Search it (brute force /
  annealing, tiny space) to minimize (i) number of cross-chip CZs in the Clifford
  ladders, and (ii) total CZ count after the peephole pass. The alternating
  top/bottom cross links matter here: chip0↔chip1 and chip2↔chip3 talk through the
  α row, chip1↔chip2 through the β row, so the α/β assignment per chip is itself a
  choice.

**Deliverable:** same energies as today's circuits by construction, but with
cross-chip two-qubit count = **exactly one parameterized RZX per double** and zero
cross-chip CZs. Easy to verify with the existing `_verify_program` machinery.

## 3. Idea B — Compile the *set* of doubles jointly (shared-Clifford diagonalization)

Today each double pays its own ladder up/down; hub continuity only cancels the shared
subtree between *adjacent* blocks. Stronger idea:

- Check whether the k selected JW strings **mutually commute** (they share the
  creation pair, supports overlap on exactly {N/2−1, N−1}; for the odd-Y
  representatives this looks like it holds — verify symbolically with `_conj`).
- If yes, there is a single Clifford `C` that diagonalizes all k strings at once:
  the circuit becomes `C† · [RZ(θ₁) … RZ(θk) in parallel] · C`.
  The Clifford cost is paid **once**, not once per double, and the k rotations run
  in parallel → depth ~independent of k.
- Synthesize `C` under the chip graph (this is a small stabilizer-tableau problem;
  qiskit's `Clifford` synthesis or hand construction from the shared ladder).
- If the strings don't all commute, partition into commuting layers and do this per
  layer.

**Expected win:** CZ count drops from O(k · support) to O(support) + k parallel
rotations. For Br2 that could plausibly mean ~15–20 CZ + 4 RZX instead of 36 + 4.

## 4. Idea C — Sparse-determinant state preparation (drop UCC form entirely)

Since the target is a superposition of ~k determinants, prepare it *directly*,
Givens-style — this generalizes the paper's Eq. (6)–(8) multireference prep that we
already implement (`initial_state_circuit`: one `Ry(β)` + 3 CX for |HF⟩ − β|D₁⟩).

- A superposition `c₀|HF⟩ + Σᵢ cᵢ|Dᵢ⟩` over k paired-double determinants costs a
  **cascade of k controlled/Givens rotations**, each followed by a CX fan-out that
  moves the excited pair into place. Each determinant differs from |HF⟩ by moving
  2 electrons into the same LUMO pair — the fan-outs largely overlap, so the CX
  cost is shared.
- Parameters = the k amplitudes, optimized variationally exactly like the θs now.
  On this determinant manifold the reduced UCCSD and the linear superposition span
  the same states (this is why `pair=False` works in the current code), so **the
  energy floor is identical to the k-doubles UCCSD** — but the circuit is a state
  prep, not k conjugated Pauli exponentials.
- Route the CX/Givens ladders along the chip graph; the rotation that crosses chips
  becomes a native RZX pair (a Givens rotation = 2 RZX + single-qubit Cliffords).

**Expected win:** roughly one entangling "rung" per determinant; plausibly the
cheapest circuit that can hit FCI in these CAS spaces. First experiment: HF (8q),
k=2 — count gates and check statevector overlap with the FCI vector from tencirchem.

## 5. Idea D — Native-pool adaptive ansatz (most exploratory)

Flip the direction: instead of compiling chemistry operators into gates, grow a
circuit from a pool of gates the hardware likes, and let symmetry constraints do the
chemistry.

- **Pool:** `RZX(θ)` on each physical edge (all edges, but cross-chip ones are
  first-class), plus single-qubit `RZ/RX`. Optionally the number-conserving
  composite: Givens/exchange rotation `exp(-iθ(XX+YY)/4)` on an edge
  (= 2 RZX + 1q Cliffords) and the paired-double composite from Idea C.
- **Growth rule:** ADAPT-VQE gradient selection (⟨[H, A]⟩ per pool element), so the
  circuit stays as short as the physics allows — the HF.ipynb result says it should
  stop after very few layers.
- **Symmetry:** bare RZX breaks particle number. Either (a) restrict the pool to the
  number-conserving composites, or (b) allow raw RZX and add a penalty
  `λ⟨(N̂−N)²⟩` to the cost function / post-select on Hamming weight. Comparing (a)
  vs (b) is itself an interesting result: raw RZX layers are 2× cheaper per layer
  but may need more layers.

**Expected win:** unknown — that's the point. It gives a data-driven lower bound on
"how much entanglement does this molecule actually need on this graph".

## 6. Idea E — Paired (hard-core boson) encoding, 1 qubit per spatial orbital

All dominant doubles are *paired* (kα,kβ → aα,aβ). In the HCB encoding a spatial
orbital is one qubit and a paired double becomes a plain **2-qubit Givens rotation**
— no JW Z-strings at all.

- Br2's (10,6) CAS → 6 qubits instead of 12; the whole ansatz is k Givens rotations
  routed along a path. Each Givens = 2 RZX-equivalents.
- Cost: the pUCCD energy floor is above full UCCSD (no unpaired excitations). Two
  mitigations: (i) our own scans show the paired doubles dominate, so the gap may be
  below the noise floor anyway; (ii) use the freed qubits to *enlarge the active
  space* (16 physical qubits → 16 spatial orbitals) — a bigger CAS at pUCCD level
  may beat a small CAS at UCCSD level. That trade-off is a nice publishable
  question on its own.

## 7. Idea F — Squeeze the parameter/measurement side (orthogonal, cheap)

- Energy is a trigonometric polynomial in each θ → **Rotosolve-style** sequential
  minimization (3 energy evaluations per parameter per sweep, no gradients), which
  is far more noise-robust than optimizing 4 parameters jointly on hardware.
- Small-|θ| RZX pulses are shorter → initialize at MP2 amplitudes (already small)
  and keep the optimizer in the small-angle regime; combine with the existing CDR
  pipeline (`June_main/export2cloud/main_Br2.py`) which already parameterizes RZX
  with sympy symbols.

---

## 8. Evaluation protocol (common to all ideas)

1. **Exactness / floor:** statevector energy vs active-space FCI (tencirchem, as in
   the notebooks). Target: within 1.6 mHa, same as the current reduced-doubles scans.
2. **Cost metrics:** total 2Q gates, **cross-chip 2Q gates** (the expensive ones),
   circuit depth, parameter count. Baseline = table in §1.2.
3. **Noise:** the existing June_main Cirq pipeline with higher depolarizing on
   cross-chip gates; compare energy-after-CDR across ansätze at matched shot budget.
4. **Robustness:** repeat over the bond-length grid (the doubles are fixed once at
   the reference geometry, as in the notebooks) — a good ansatz should not need
   re-selection along the curve.

## 9. Suggested order of attack

| priority | idea | risk | expected payoff |
|---|---|---|---|
| 1 | A: pivot-on-the-link + placement search | low (extends existing compiler) | all cross-chip 2Q = 1 RZX per double |
| 2 | B: joint diagonalization of the doubles set | medium | ~2× fewer CZs, depth independent of k |
| 3 | C: sparse-determinant Givens prep | medium | likely the global minimum gate count |
| 4 | E: HCB / pUCCD encoding | medium | halves qubits or doubles active space |
| 5 | D: adaptive native-pool ansatz | high | data-driven "how cheap can it get" |
| — | F: rotosolve + small-angle RZX | low | free noise reduction on top of any of the above |

Ideas A→B→C form a natural sequence: each reuses the verification machinery
(`_verify_program`, statevector checks) already in the repo, and each strictly
reduces the two-qubit budget while keeping the same variational manifold.
