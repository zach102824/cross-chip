# Cross-chip machine: minimizing inter-chip error + which algorithms fit

Brainstorm, 2026-07-31.

Setting: a multi-chip machine where each chip is an NxN local lattice (or
possibly denser; this must be specified), and chips are joined by a sparse,
flexibly selected set of cross-chip links. In the 2x2 example in
`qubit_connectivity/all_connection_cases.txt`, every saturated configuration is
a perfect matching selected from row/column-aligned candidates. The file does
not by itself establish that links can be reconfigured *during a circuit*.
Cross-chip 2Q gates have error `p_x` much larger than intra-chip error `p_i`.
The current `June_main/main_cursor_lib.py` represents this with
`CZ_CROSS_CHIP_TAG` in `GateArityDepolarizingNoise`; older/archived code also
contains `cz_high`/`cz_normal` location-aware models. Neither model yet includes
the complete Bell-generation, reset, feedforward, latency, and memory process.

Two questions:

1. **How** to use the machine so that cross-chip error hurts as little as possible.
2. **What** problems/algorithms are naturally shaped for this architecture.



## Audit note

The first draft had several overclaims that are corrected below:

- Offline Bell-pair generation/TeleGate is not inherently more accurate than a
direct cross-chip gate; it wins only through heralding, reuse, distillation,
latency hiding, or a different physical link primitive.
- The stated DEJMPS equation was not a Werner-state equation, the claimed
`2ε²` suppression was wrong, and the `p_x/p_i ≳ 5–10` threshold was invented
from an inadequate scalar model.
- Independent CZ/CNOT gate cuts cost `9^k`, but joint LOCC gate cutting can
reduce this to `(2^(k+1)-1)^2 = O(4^k)` with ancillary-memory requirements.
Wire cutting separately has `16^k` local-only and `4^k` LOCC scaling.
- An NxN nearest-neighbor chip is not dense/all-to-all, so local routing cannot
be treated as free.
- Virtual distillation suppresses non-dominant eigencomponents, not all noise,
and its noisy cross-copy measurement can remove the benefit.

The strongest new directions after the audit are **QST/dynamic state
migration**, **cross-chip-awar**`e shared Clifford/parity synthesis, terminal
symmetry verification, and cross-budgeted ansatz design—not distillation
alone.`

---



## Part 1 — Strategies to minimize cross-chip interaction error



### 1.0 Revised core principle: make cross-chip communication sparse and reusable

A direct cross-chip CZ inside the algorithm circuit puts error `p_x` directly
on *data* qubits. A useful option is to use cross links to create entanglement
between dedicated communication qubits ahead of time, then consume that
resource using intra-chip gates, measurement, and classical feedforward.

This is **not automatically more accurate**. If a raw Bell pair is created by
the same noisy deterministic cross-chip gate, gate teleportation adds local
gates and measurements and is normally worse than applying that cross gate
directly. It becomes attractive only when at least one of these is true:

- **The physical entanglement-generation mechanism is heralded.** An ordinary
third-qubit parity check does not certify an unknown Bell pair for free;
useful heralding must come from the link mechanism or from a real
detection/distillation protocol using extra pairs.
- **Several remote gates reuse one ebit**, or a logical qubit is teleported once
and then participates in many local gates.
- **Distillation improves the link state enough** to offset its local-gate,
measurement, memory, and discarded-pair costs.
- **Scheduling freedom.** Bell-pair generation can run in parallel with intra-chip
computation if spare communication qubits exist and memory decoherence is
slow enough. This hides latency, not infidelity.



### 1.1 Entanglement-mediated remote gates (telegate / EJPP)

The standard protocol (Eisert–Jacobs–Papadopoulos–Plenio, "EJPP"): one shared
Bell pair + 2 classical bits + local gates implements a remote CNOT/CZ between
a qubit on chip A and a qubit on chip B.

Cost per remote CZ:

- 1 ebit (one cross-chip Bell pair between comm qubits a', b')
- ~2 local CNOTs, 2 measurements, feedforward Pauli corrections
- Resulting gate infidelity ≈ (Bell pair infidelity) + (a few × `p_i`) + measurement error

The raw-pair protocol is therefore a communication primitive, not an error
mitigation method by itself. Its fidelity must be compared end-to-end with a
direct cross-chip gate under the actual link and measurement noise.

**Ebit packing / embedding.** One Bell pair can mediate *multiple* gates if
they form a bipartite, uninterrupted distributable packet sharing a control.
The EJPP cat-entangler can stay open across the burst: entangle once, apply the
remote controlled gates, then disentangle. A bipartite Pauli-rotation packet
can therefore sometimes use one ebit, but a generic Pauli ladder does not do so
automatically; multipartite or interrupted packets require more. In a QAOA
cost layer, cut ZZ terms incident on one qubit can share an ebit only when their
targets are on the same remote chip and no incompatible operation interrupts
the packet.

This is a compilation problem: "minimize ebits consumed" instead of "minimize
cross-chip gates," and the literature (Andrés-Martínez & Heunen hypergraph
partitioning; Wu et al. entanglement-efficient distribution) reduces it to
hypergraph min-cut. Worth implementing a small version for our ansätze.

### 1.2 Entanglement distillation: buy fidelity with cheap local gates

When intra-chip gates are substantially cleaner than cross-chip gates,
distillation can exchange n noisy Bell pairs for one better pair using only
local 2Q gates and measurement (BBPSSW / DEJMPS). Intra-chip routing overhead
must still be included for a nearest-neighbor NxN lattice.

The result depends strongly on the Bell-pair noise model. For a binary
phase-flip/dephased Bell state, an ideal recurrence step gives

```
F' ≈ F² / (F² + (1-F)²)
```

so input infidelity ε becomes approximately ε². The earlier version of this
note incorrectly called this a Werner-state formula and wrote `2ε²`.

For an isotropic Werner state, the ideal recurrence formula is instead

```
F' = [F² + (1-F)²/9] /
     [F² + 2F(1-F)/3 + 5(1-F)²/9]
```

and the success probability is the denominator. Near F=1, the output
infidelity is approximately `(2/3)ε`, not quadratic. DEJMPS is still useful,
especially for biased/Bell-diagonal noise, but the channel structure matters.

One round uses one bilateral CNOT (one local CNOT on each chip), one measured
qubit on each chip, plus local basis rotations. It consumes two raw pairs per
attempt; expected raw-pair consumption is `2/P_success`, and the ideal yield is
`P_success/2`. Near F=1 the success probability is close to one, while the
yield is at most about one-half. These are different quantities.

There is therefore no justified universal threshold such as
`p_x / p_i ≳ 5–10`. The breakeven depends on the full cross-link Pauli/coherent
error channel, local CNOT and measurement errors, memory error while waiting,
pair-generation rate, and whether the objective is fidelity, latency, or total
shots. Measuring this breakeven remains a good experiment, but it must simulate
the full protocol rather than use the earlier scalar estimate.

**Entanglement pumping variant** if qubits are scarce: use one storage qubit
and one reusable communication qubit per node, reducing memory from exponential
to constant. It is sequential, and a failed later round can discard accumulated
progress.

### 1.3 Architecture co-design: exploit the *flexible* connectivity

This machine has a knob most papers don't: the cross-chip matching is
choosable per circuit (72 candidate edges, perfect matchings in the 4-chip
case). Candidate links are not independently selectable: different chip pairs
compete for the same physical qubits. So compilation is a globally coupled
optimization:

```
(qubit placement on chips) × (cross-link matching) × (routing/scheduling)
```

Concrete formulation:

1. Build the algorithm's interaction graph G (vertices = logical qubits).
  Use calibrated gate-count/error weights for physical execution and a
   separate angle-dependent quasiprobability cost for circuit cutting;
   `sum(|θ|)` is not a universal communication-error weight.
2. Partition G into chips minimizing a weighted cut (METIS/KL/spectral), while
  retaining a real intra-chip routing cost. An NxN nearest-neighbor grid is
   not dense/all-to-all; SWAP depth can be significant.
3. Given the partition, choose a legal cross-chip configuration with a global
  constrained maximum-weight matching or ILP, weighted by per-link calibrated
   error if links are heterogeneous.
4. Residual cut edges not covered by the matching get routed to a covered link
  or handled by teleportation/cutting. The compiler must price the added
   SWAPs and idle decoherence rather than assuming they are cheap.

Extra idea: **route everything through the best link.** If calibration shows
one cross link is much better than others, it can be worth paying intra-chip
SWAPs to funnel all traffic through it (depth ↑, fidelity ↑). The tradeoff
depth-vs-fidelity is measurable in our noise model.

Extra idea: **time-multiplexed matchings.** If the hardware can be re-matched
between circuit layers (even slowly, between shots/batches), different layers
of the circuit could use different matchings — e.g., Trotter step 1 uses the
A-B/C-D matching, step 2 uses A-C/B-D. Worth asking the hardware team what
the reconfiguration timescale actually is; it changes the compiler design a lot.

### 1.4 TeleData: move a qubit once, then compute locally

The earlier draft focused too much on gate teleportation. A different primitive
can be better: move the *state* of a logical qubit by coherent QST or
teleportation to the chip where its next block of partners live, execute the
whole block locally, and optionally move it back. This is TeleData/dynamic
remapping.

- An isolated remote controlled gate favors a direct gate or TeleGate.
- A burst of many interactions involving the same qubit can favor TeleData:
one ebit moves the qubit, after which all covered gates are local. Returning
to the original chip costs another ebit only if the later schedule requires it.
- Capacity matters: the destination needs a free data qubit, and movement can
create a new set of remote interactions elsewhere.

The best compiler should therefore choose jointly among direct cross gates,
TeleGate packets, TeleData moves, SWAP routing, and circuit cuts. A static
interaction-graph min-cut misses this temporal structure; the relevant object
is the time-ordered circuit or a sequence of interaction graphs.

### 1.5 Design the algorithm around an explicit cross-chip budget

Compiling a generic UCCSD or hardware-efficient ansatz after the fact may be
the wrong level of attack. For VQE, make cross-chip communication a variational
resource:

```
operator score = predicted energy improvement
                 - λ × expected cross-chip error/cost
```

A chip-aware ADAPT-VQE can build rich fragment-local states first, then admit
only the inter-fragment Pauli generators whose measured gradients justify
their communication cost. Similarly, a modular ansatz can contain many local
layers but only `b` carefully placed cross-chip ZZ/Pauli-gadget layers.

This is physically meaningful: cross-chip gates control the Schmidt rank
available across the partition. A small number of well-chosen inter-chip
entanglers can add much more expressive power than many indiscriminate ones.
The scientific question becomes a Pareto curve—energy error versus noisy
cross-chip budget—rather than only minimizing gates for a fixed circuit.

For chemistry simulation, also co-optimize the fermion-to-qubit mapping,
ordering of Pauli gadgets, parity-tree root/topology, and qubit allocation.
Different CNOT trees implement the same Pauli rotation but can have very
different numbers of cross-chip edges.

**Repository-specific strongest compiler idea: shared Clifford/parity frames.**
`H4_circuits/build_h4_circuit.py` already verifies that six Pauli rotations
commute, have GF(2) rank four, and can share one `V ... V†` Clifford frame.
Instead of compiling each Pauli gadget independently, synthesize the common
frame to minimize cross-chip GF(2) parity rank, routed cross edges, and legal
matching changes. This directly exploits existing work and may remove more
cross interactions than Bell-pair distillation can repair.

### 1.6 Avoid cross-chip quantum operations entirely (classical stitching)

When the number of cut gates is small, don't do them at all:

- **Circuit cutting / knitting (quasi-probability).** Independently cutting k
CZ/CNOT gates with an optimal local-operations decomposition has sampling
overhead `9^k`. Jointly cutting k CNOT/CZ gates with LOCC can reduce this to
`(2^(k+1)-1)^2 = O(4^k)`, at the cost of ancillary quantum memory and
feedforward. Wire cutting is a separate result: `16^k` with local operations
and `4^k` with LOCC. Parameterized rotations can be cheaper: their overhead
is angle-dependent and tends to one for a small angle. The crossover cannot
be fixed at “k ≤ 3–4”; it depends on target precision, observable variance,
hardware noise, ancillas, and available parallelism.
- **Entanglement forging (Eddins et al.).** Exactly the VQE use case: a 2N-qubit
state is decomposed into Schmidt-basis products of two N-qubit halves; each
half runs on one chip; cross-correlation is reconstructed classically.
Works best when the state is weakly entangled across the cut — see Part 2.
- **Divide-and-conquer / DMET-style embedding** (see Part 2) — the chemistry
itself is fragmented before it ever becomes a circuit.

Rule of thumb hierarchy as cut-weight grows:
few cut gates → cut classically; moderate → telegate + distillation;
heavy cut traffic → the problem is mis-partitioned or mis-matched to the
machine, repartition or accept direct gates on the best links.

### 1.7 Cross-link-targeted error mitigation

Because the noisy locations are *known and few*, mitigation can be surgical:

- **Selective ZNE / PEC on cross gates only.** Amplify (or quasi-probabilistically
cancel) noise only at tagged cross-gate locations. This can reduce the PEC
exponent, but the shot overhead remains the product of the cross gates'
inverse-channel norms and can still be exponential in accumulated fault rate.
It also leaves intra-chip noise untreated.
- **CDR with cut-aware training circuits** (extends `gradient_cdr_plan.md`):
build Clifford training circuits that preserve the number and location of
cross-chip gates, so the learned correction specifically models the boundary noise.
- **Chemistry symmetry verification.** At terminal measurement, verify particle
number, spin parity, tapered Z2 symmetries, or paired-occupation constraints
and reject violating shots. This requires no Bell factory or mid-circuit
feedforward and should be tested before boundary error-correcting codes.
- **Dynamical decoupling on data qubits** while they idle waiting for Bell-pair
generation/distillation on comm qubits — the async protocol in 1.1 makes
idling windows long and predictable, ideal for DD.
- **Randomized compiling on cross links** can tailor the *averaged* noise toward
a stochastic Pauli channel. It does not make a Bell pair Werner, remove
correlations, or guarantee tailoring under strongly gate-dependent noise.



### 1.8 Longer-horizon: boundary-only error correction

Asymmetric protection could encode only communication qubits or use an
error-detecting encoded Bell pair, while data qubits stay bare. A concrete
small-code proposal cannot be chosen from error rate alone: it depends on
which local stabilizer measurements are available, correlated cross-link
errors, communication-qubit count, and storage time. This is distillation's
cousin (detection + postselection) and is probably a paper-scale idea rather
than a quick experiment.

---



## Part 2 — Problems whose structure matches the architecture

The architecture wants problems whose interaction graph is **clustered:
dense blobs, weak/sparse coupling between blobs** — so the min-cut in 1.3 is
genuinely small — or problems where cross-chip interaction is needed only in
a **single shallow layer**. Candidates, roughly ordered by fit to this repo:

### 2.1 Fragmented quantum chemistry (strongest fit, continuous with current work)

Molecules/materials with weakly coupled fragments:

- van der Waals / hydrogen-bonded dimers and clusters (water clusters,
benzene dimer), molecule–surface physisorption;
- H4 viewed as two H2 units (already have `H4_6q_3doubles_disjoint_rzx` —
the "disjoint" ansätze in `exploring/` are literally chip-shaped);
- active-space-per-fragment methods: DMET, divide-and-conquer VQE,
cluster-VQE (`clusering_VQE/`), localized-orbital UCCSD where doubles
amplitudes across fragments are small.

Why it can fit: for well-separated fragments and an appropriate localized
active space, many important amplitudes may become fragment-local or small.
This is not automatic—Coulomb terms, interfragment correlation,
fermion-to-qubit strings, and basis choice can retain long-range interactions.
Locality must be verified from the integrals and compiled Pauli terms. When it
holds, the remaining small-angle cross terms are favorable for ebit packing or
angle-dependent circuit cutting.

Concrete project: **"chip-aware orbital localization"** — choose the orbital
localization/ordering to minimize the cut weight of the UCCSD interaction
graph across chips, then compare direct-CZ vs telegate+distillation vs
forging on the existing H4/HF/Cl2 pipelines with the location-aware noise model.

### 2.2 Two-copy algorithms: a promising but link-noise-sensitive fit

A whole family of algorithms needs **two copies of a state + one transversal
layer of cross-copy 2Q gates**. On this machine: copy 1 on chip A, copy 2 on
chip B, and — because the matching is *flexible* — choose cross links so qubit
i(A) pairs with qubit i(B). If all corresponding pairs are simultaneously
available in one matching, the cross-chip cost can be one transversal layer at
the end (or a destructive Bell measurement implemented by CX/CZ plus local
measurement):

- **Virtual distillation / ESD error mitigation**: estimate Tr[ρ²O]/Tr[ρ²] to
suppress components outside the dominant eigenvector of ρ. It does not
remove the coherent “eigenvector floor,” and noise in the distillation
circuit can erase the gain. For two copies, destructive SWAP/Bell-basis
measurements can replace controlled-SWAPs for suitable estimators, giving a
transversal final layer; observable insertion and measurement grouping still
require care. This is promising here, but it is a hypothesis to test rather
than an automatic double win.
- **Overlap / fidelity estimation** |⟨ψ|φ⟩|²: prepare ψ on A, φ on B,
transversal Bell measurement. Feeds non-orthogonal VQE, quantum subspace
expansion, eigenstate witnessing — adjacent to the OGM work in `state_transfer/`.
- **Higher moments / entanglement spectroscopy, Tr[ρ^n]**: for `n>2`, a cyclic
moment requires a controlled cyclic permutation or a specialized multi-copy
protocol with additional depth/ancillas. A literal four-cycle also conflicts
with a one-cross-link-per-qubit matching, so `Tr[ρ⁴]` is not generally one
transversal layer on this layout.
- **SWAP-test-based kernels** for QML: kernel entry = overlap of two feature
states, one per chip.

This has a clean topology match, but the final cross-copy layer is directly in
the estimator and can be the dominant error. It should not yet be called the
“killer app”; the right experiment is to find when the mitigation gained from
the second copy exceeds the extra cross-link and measurement noise.

### 2.3 Clustered optimization (QAOA/annealing-inspired)

MaxCut / Ising on graphs with community structure (social-network-like,
modular). Partition communities onto chips; only inter-community edges cross.
Cost-layer cross terms are diagonal (`e^{-iγZZ}`) → ideal for ebit packing
(one ebit can cover an uninterrupted star packet whose targets are on the same
remote chip; multiple destination chips require more resources) and for
LOCC-assisted cutting. Benchmark: random modular graphs, sweep
inter/intra edge ratio, find where cross-chip machine + telegate beats a
monolithic noisy device of the same size.

### 2.4 Weakly coupled lattice dynamics (Trotter)

Coupled chains/ladders/bilayers (bilayer Hubbard, coupled Heisenberg chains,
spin ladders with weak rungs): put each chain on a chip. Two structural gifts:

- inter-chain terms are geometrically sparse (a matching — exactly what the
hardware provides natively!);
- if inter-chain coupling J' << J, use **multi-rate Trotterization**: combine m
weak-coupling steps into a less frequent cross-chip rotation with angle
approximately `m J' dt`. This can reduce cross-gate count, but changes the
product-formula error and must be bounded or benchmarked; simply dropping
`m-1` cross layers would simulate the wrong Hamiltonian.

Same trick applies to chemistry Trotter dynamics with weak inter-fragment terms.

### 2.5 Distributed/parallel primitives

- **Parallel VQE ensembles**: different chips run different ansatz candidates /
different Hamiltonian terms / different CDR training circuits concurrently —
zero cross-chip gates, pure throughput. (Mundane but real.)
- **Distributed amplitude estimation / metrology with shared GHZ** across chips —
needs a one-time GHZ preparation followed by local phase accumulation. The
preparation can take one cross layer only if the selected chip-level links
support the needed spanning tree in parallel.



### 2.6 Anti-patterns (what NOT to run here)

Useful to state explicitly: expander-like interaction graphs, QFT-heavy
algorithms, and random dense circuits cut badly because every static partition
has a large boundary. A star is more nuanced: migrating its hub once per chip
can cover many leaf interactions, so central-spin/impurity circuits may favor
TeleData rather than being automatic anti-patterns. If even dynamic remapping
leaves cross traffic far above link capacity, this machine is the wrong tool.

---



## Part 3 — Suggested next experiments in this repo

Ordered by effort/insight ratio:

1. **QST migration versus direct-RZX crossover.** Use the existing
  `state_transfer/rewrite_hf_tapered_with_qst.py` HF pipeline. Compare direct
   cross-chip RZX with ping-pong QST and a version that leaves the state on the
   destination chip to cover a whole interaction epoch. Include QST noise and
   destination-capacity constraints.
2. **Cross-chip-aware shared Clifford-frame synthesis.** Extend
  `H4_circuits/build_h4_circuit.py` so its gauge/frame search minimizes
   cross-chip parity rank and legal-matching/routing cost, not only total
   compiled gates. Compare shared-frame synthesis with independent Pauli
   gadgets.
3. **Terminal chemistry symmetry verification.** Add particle-number, spin/Z2,
  and paired-occupation postselection to the current shot pipeline; measure
   accepted-shot rate and energy bias under elevated cross-chip noise.
4. **Define and benchmark the communication primitives.** Extend the current
  model, which applies noise after a direct tagged CZ, to explicitly represent
   Bell-pair generation, local CNOTs, mid-circuit measurement/feedforward,
   failed-pair discard, and memory noise. Compare direct gate, raw TeleGate,
   one-round distillation, and TeleData under at least (a) dephasing-biased and
   (b) Werner/depolarizing link noise. Report process fidelity, raw pairs per
   accepted operation, latency, and qubit overhead—not only `p_x/p_i`.
5. **TeleGate versus TeleData crossover.** On actual UCCSD circuits, identify
  time-ordered bursts of remote interactions and measure when moving a qubit
   once beats teleporting several gates. This is a stronger compiler baseline
   than static min-cut alone.
6. **Cross-budgeted VQE/ADAPT.** Build fragment-local operators freely and
  penalize inter-chip operators by calibrated expected error. Plot energy
   error against the number of cross-chip layers/ebits. Compare with compiling
   an unconstrained UCCSD circuit afterward.
7. **H4 = 2×H2 across two chips.** Take the existing disjoint-RZX H4 ansatz,
  place fragments on chips, compare energies: (a) direct cross CZ,
   (b) telegate, (c) telegate+distilled, (d) circuit-cut / forged (no cross
   gates). Uses existing Hamiltonians and shot-estimation code.
8. **Virtual distillation demo on 2 chips.** VQE state duplicated on A and B,
  transversal cross-chip Bell measurement, estimate mitigated energy. Compare
   error suppression gained vs cross-link error paid — find the `p_x` threshold
   where it's net-positive.
9. **Matching-aware compiler prototype.** Given an interaction graph + the 72
  candidate edges, jointly pick placement + matching (ILP or greedy +
   local search); evaluate on UCCSD circuits from `UCCSD_Mole/`. Measures how
   much the *flexibility* itself is worth (vs a fixed matching).
10. **Ebit-packing/Pauli-gadget pass.** Jointly choose gadget order, CNOT-tree
  topology, and allocation; count ebits versus naive remote-gate count on the
   UCCSD circuits.



## Key references to pull

- Eisert, Jacobs, Papadopoulos, Plenio — optimal local implementation of
nonlocal gates (EJPP), PRA 62, 052317 (2000).
- Bennett et al. (BBPSSW) & Deutsch et al. (DEJMPS) — entanglement distillation.
- Andrés-Martínez & Heunen — distributing circuits over heterogeneous networks
via hypergraph partitioning.
- Wu, Ahn, et al. — entanglement-efficient distributed compilation (ebit packing).
- Peng, Harrow, Ozols, Wu — simulating large circuits on small devices (wire cutting);
Piveteau & Sutter — circuit knitting with classical communication.
- Eddins et al. — entanglement forging (IBM, H2O VQE).
- Huggins et al. / Koczor — virtual distillation / ESD.
- Ferrari et al. — modular compilation combining TeleData and TeleGate.
- Saleem et al. — VQE ansätze with a limited number of inter-module operations.
- COSMA — joint fermion mapping, Pauli-gadget scheduling, allocation, and routing
for modular quantum chemistry.
- Monroe et al. — modular ion-trap architecture with probabilistic photonic
interconnects and fault-tolerant use of distributed entanglement.

