"""Candidate compilation methods for the flexible-connectivity search."""
from __future__ import annotations

import itertools
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from constraints import bridges_disjoint, satisfies_rules
from freeze import all_keep_masks, fully_freeze, freeze_z_pairs
from flexible_compile import (
    candidate_bridges,
    compile_flexible,
    compile_strings,
    gen,
    prog_to_gates,
)
from error_budget import score_gates, spin_split_cross_pairs, cross_from_list


# ----------------------------------------------------------------------
# Method A: flexible hub schedule search (spin-block layout)
# ----------------------------------------------------------------------
def search_flexible_hubs(strings, signs=None, cross_pairs=None, max_blocks=5,
                         require_disjoint_rzx=False, require_all_qubits=False,
                         order="auto"):
    """Brute-force per-block α–β bridge choices (tiny pools).

    If require_disjoint_rzx, only vertex-disjoint hub schedules are kept
    (no two RZX may share a qubit).  If require_all_qubits, every wire must
    appear in the fused gate list.
    """
    n = len(strings[0])
    if signs is None:
        signs = [1] * len(strings)
    if cross_pairs is None:
        cross_pairs = spin_split_cross_pairs(n)

    order_idx = gen._auto_order(strings) if order == "auto" else list(range(len(strings)))
    ordered = [strings[i] for i in order_idx]
    ordered_signs = [signs[i] for i in order_idx]
    cands = [candidate_bridges(s, n) for s in ordered]
    if any(len(c) == 0 for c in cands):
        return []
    # cap product size
    sizes = [len(c) for c in cands]
    prod = 1
    for s in sizes:
        prod *= s
    results = []
    if prod > 8000:
        # greedy: prefer disjoint vertical bridges when required
        schedule = []
        used: set[int] = set()
        half = n // 2
        for c in cands:
            ranked = sorted(
                c,
                key=lambda p: (
                    0 if (p[0] not in used and p[1] not in used) else 1,
                    0 if p[1] == p[0] + half else 1,
                ),
            )
            choice = None
            for p in ranked:
                if not require_disjoint_rzx or (
                    p[0] not in used and p[1] not in used
                ):
                    choice = p
                    break
            if choice is None:
                return []
            schedule.append(choice)
            used.update(choice)
        schedules = [schedule]
    else:
        schedules = list(itertools.product(*cands))

    best = None
    for sched in schedules:
        if require_disjoint_rzx and not bridges_disjoint(sched):
            continue
        try:
            out = compile_strings(
                ordered, signs=ordered_signs, order="given",
                hub_schedule=sched, fuse=True,
            )
        except Exception:
            continue
        ok, _why = satisfies_rules(out["gates"], n)
        if require_disjoint_rzx and require_all_qubits and not ok:
            continue
        if require_disjoint_rzx and not require_all_qubits and _why == "rzx_overlap":
            continue
        if require_all_qubits and not require_disjoint_rzx and _why.startswith("unused"):
            continue
        bud = score_gates(out["gates"], cross_pairs)
        rec = {
            "name": "flexible_hubs",
            "schedule": sched,
            "order": order_idx,
            "gates": out["gates"],
            "budget": bud,
            "strings": ordered,
            "strings_orig_order": strings,
            "rules_ok": ok,
            "rules_why": _why,
        }
        results.append(rec)
        if best is None or bud.error < best["budget"].error:
            best = rec
    if not results:
        return []
    ranked = sorted(results, key=lambda r: (not r["rules_ok"], r["budget"].error))
    return ranked[:6]


# ----------------------------------------------------------------------
# Method B: full Z-pair freeze + flexible hubs
# ----------------------------------------------------------------------
def method_full_freeze_flexible(strings, signs=None, cross_pairs=None):
    frozen = fully_freeze(strings)
    hits = search_flexible_hubs(frozen, signs=signs, cross_pairs=cross_pairs)
    for h in hits:
        h["name"] = "full_freeze_flexible"
        h["frozen_strings"] = frozen
    return hits


# ----------------------------------------------------------------------
# Method C: disjoint weight-2 blocks (frozen paired doubles)
# Each Y_k X_{k+half}-style pair compiles to one RZX; no shared ladder.
# ----------------------------------------------------------------------
def _compile_yx_pair(q_y: int, q_x: int, theta_index: int, sign: int = 1):
    """Gates for exp(-i * sign * t/2 * Y⊗X) via CZ·RX·CZ -> RZX fusible form."""
    # Frame: basis so ladder letters match Y on q_y and X on q_x.
    # pivot = q_y, ladder = H(q_x) · CZ(q_y, q_x)  gives X_pivot -> Z_y? let's use gen.
    n = max(q_y, q_x) + 1
    # Build a string on a register large enough; caller uses full n.
    return q_y, q_x, theta_index, sign


def method_independent_pairs(strings, n_qubits, signs=None, cross_pairs=None):
    """Assume fully frozen strings are weight-2 Y_a X_b; emit independent blocks."""
    if signs is None:
        signs = [1] * len(strings)
    if cross_pairs is None:
        cross_pairs = spin_split_cross_pairs(n_qubits)

    frozen = fully_freeze(strings)
    gates = []
    prog = []
    expected = []
    for d, s in enumerate(frozen):
        sup = [q for q, p in enumerate(s) if p != "I"]
        if len(sup) != 2:
            return []  # not applicable
        a, b = sup
        letters = s[a] + s[b]
        # compile with flexible on the 2-support string padded to n_qubits
        prefix, pivot, ph, _ = compile_flexible(s, n_qubits, hub_a=a if a < n_qubits // 2 else None,
                                                hub_b=b if b >= n_qubits // 2 else None)
        prog += prefix
        prog.append(("ROT", pivot, d, signs[d] * ph))
        prog += gen._invert(prefix)
        expected.append((s, ph))
    prog = gen._peephole(prog)
    gen._verify_program(prog, n_qubits, expected)
    gates = gen.fuse_cz_rot_cz_to_rzx(prog_to_gates(prog))
    bud = score_gates(gates, cross_pairs)
    return [{
        "name": "independent_pairs_freeze",
        "frozen_strings": frozen,
        "gates": gates,
        "budget": bud,
        "strings": frozen,
    }]


# ----------------------------------------------------------------------
# Method D: spatial co-location + local α⊗β pre-merge (Method 2)
# Logical spin-block indices -> physical wires grouped by spatial orbital.
# Chip A holds high spatials (incl. LUMO), chip B the rest; one bridge.
# ----------------------------------------------------------------------
def spatial_index(q: int, n_qubits: int) -> int:
    half = n_qubits // 2
    return q if q < half else q - half


def method_spatial_premerge(strings, n_qubits, shared_spatial: int,
                            signs=None, cross_pairs_physical=None):
    """Compile frozen paired doubles with local pre-merge + 1 RZX on the bridge.

    Physical layout (n even):
      wire 2*s     = spatial s, alpha
      wire 2*s + 1 = spatial s, beta
    Bridge between the LUMO spatial's alpha wire and a designated chip-B hub
    is NOT required; we put an abstract cross link between the two chip
    parity qubits.  Scoring: any 2q gate whose two qubits lie in different
    chips counts as cross-chip.

    Chip A = spatials {shared_spatial} union optionally neighbours
    Chip B = everything else.
    For the simple win we put ONLY the shared LUMO spatial on chip A
    (2 qubits) and the rest on chip B.
    """
    if signs is None:
        signs = [1] * len(strings)
    half = n_qubits // 2
    frozen = fully_freeze(strings)

    # Map spin-block qubit -> physical wire
    # physical: [Lα, Lβ, ... other spatials as (α,β) pairs on chip B]
    # Actually use: phys_of_spinblock[q]
    def spinblock_to_phys(q: int) -> int:
        s = spatial_index(q, n_qubits)
        spin = 0 if q < half else 1
        return 2 * s + spin

    chip_A_spatials = {shared_spatial}
    chip_of_spatial = {
        s: ("A" if s in chip_A_spatials else "B") for s in range(half)
    }

    def is_cross(q1: int, q2: int) -> bool:
        s1, s2 = q1 // 2, q2 // 2  # physical wires
        return chip_of_spatial[s1] != chip_of_spatial[s2]

    # Build circuit in PHYSICAL wire order by rewriting strings
    phys_strings = []
    for s in frozen:
        letters = ["I"] * n_qubits
        for q, p in enumerate(s):
            if p != "I":
                letters[spinblock_to_phys(q)] = p
        phys_strings.append("".join(letters))

    # Custom compile: for each weight-4 string on two spatials s0 (chip B) and
    # L (chip A), local-merge each spatial's (α,β) then RZX across.
    gates = []
    # We still use flexible compile on physical strings but score with chip cut.
    try:
        out = compile_strings(phys_strings, signs=signs, order="auto",
                              hub_schedule=None, fuse=True)
    except Exception:
        return []

    # Determine cross pairs = all physical pairs that cross chips
    cross = set()
    for a in range(n_qubits):
        for b in range(a + 1, n_qubits):
            if is_cross(a, b):
                cross.add(frozenset((a, b)))
    if cross_pairs_physical is not None:
        cross = cross_pairs_physical

    bud = score_gates(out["gates"], cross)
    return [{
        "name": "spatial_premerge",
        "frozen_strings": frozen,
        "phys_strings": phys_strings,
        "gates": out["gates"],
        "budget": bud,
        "cross_pairs": [tuple(sorted(x)) for x in cross],
        "chip_A_spatials": sorted(chip_A_spatials),
    }]


def method_spatial_premerge_explicit(strings, n_qubits, shared_spatial: int,
                                     signs=None):
    """Hand-built local-premerge circuit: exactly 1 cross RZX per double.

    For a frozen string supported on spatials {k, L} with letters
    Y/X on the four spin-orbitals, emit:
      on chip B (spatial k): basis + CZ(kα,kβ) fan-in to kα
      on chip A (spatial L): basis + CZ(Lα,Lβ) fan-in to Lα
      RZX(θ) on (kα, Lα)   <-- the only cross-chip gate
      undo local fans
    """
    if signs is None:
        signs = [1] * len(strings)
    half = n_qubits // 2
    frozen = fully_freeze(strings)
    L = shared_spatial

    def phys(s, spin):  # spin 0=α, 1=β
        return 2 * s + spin

    gates = []
    for d, s in enumerate(frozen):
        sup = [q for q, p in enumerate(s) if p != "I"]
        spatials = sorted({spatial_index(q, n_qubits) for q in sup})
        if len(spatials) != 2 or L not in spatials:
            return []  # pattern mismatch
        k = spatials[0] if spatials[1] == L else spatials[1]
        if k == L:
            return []

        # letters on (kα,kβ,Lα,Lβ) in spin-block indexing
        def letter_sb(s_orb, spin):
            q = s_orb if spin == 0 else s_orb + half
            return s[q]

        # Physical qubits
        ka, kb = phys(k, 0), phys(k, 1)
        La, Lb = phys(L, 0), phys(L, 1)

        # Basis changes so that after local CZ merges we rotate Y/X correctly.
        # Use the standard generator on the 4-qubit rewritten string in phys order
        # by calling compile_flexible on a phys string with forced hubs (ka, La).
        phys_str_list = ["I"] * n_qubits
        for s_orb, spin in ((k, 0), (k, 1), (L, 0), (L, 1)):
            phys_str_list[phys(s_orb, spin)] = letter_sb(s_orb, spin)
        phys_str = "".join(phys_str_list)

        # Force bridge = (ka, La): both are α wires of the two spatials.
        # compile_flexible expects α = q < n/2 in its OWN indexing — but our
        # physical layout is interleaved, so the built-in alpha/beta split is
        # wrong.  Build the ladder manually.
        prefix, pivot, ph = _compile_spatial_pair(
            phys_str, n_qubits, k, L, half_spinblock=half
        )
        gates.extend(prog_to_gates(
            prefix + [("ROT", pivot, d, signs[d] * ph)] + gen._invert(prefix)
        ))

    # Peephole across the concatenated program
    # Re-parse gates into prog tuples for peephole — simpler: rebuild prog
    prog = []
    expected = []
    for d, s in enumerate(frozen):
        spatials = sorted({spatial_index(q, n_qubits) for q in range(n_qubits) if s[q] != "I"})
        k = spatials[0] if spatials[1] == L else spatials[1]
        phys_str_list = ["I"] * n_qubits
        for s_orb, spin in ((k, 0), (k, 1), (L, 0), (L, 1)):
            q_sb = s_orb if spin == 0 else s_orb + half
            phys_str_list[phys(s_orb, spin)] = s[q_sb]
        phys_str = "".join(phys_str_list)
        prefix, pivot, ph = _compile_spatial_pair(phys_str, n_qubits, k, L, half)
        prog += prefix
        prog.append(("ROT", pivot, d, signs[d] * ph))
        prog += gen._invert(prefix)
        expected.append((phys_str, ph))
    prog = gen._peephole(prog)
    # verify each ROT frame matches phys string
    gen._verify_program(prog, n_qubits, expected)
    gates = gen.fuse_cz_rot_cz_to_rzx(prog_to_gates(prog))

    # Cross = any 2q with endpoints in different chips (A={L}, B=rest)
    cross = set()
    for a in range(n_qubits):
        for b in range(a + 1, n_qubits):
            sa, sb = a // 2, b // 2
            if (sa == L) != (sb == L):
                cross.add(frozenset((a, b)))
    bud = score_gates(gates, cross)
    return [{
        "name": "spatial_premerge_explicit",
        "frozen_strings": frozen,
        "gates": gates,
        "budget": bud,
        "n_cross_target": len(frozen),
        "cross_pairs": [tuple(sorted(x)) for x in cross if any(
            frozenset(g.get("qubits", [])) == x for g in gates if len(g.get("qubits", [])) == 2
        )],
    }]


def _compile_spatial_pair(phys_string: str, n: int, k: int, L: int, half_spinblock: int):
    """Compile a 4-support phys string on spatials k,L with bridge (kα, Lα).

    Physical wire: 2*s = α, 2*s+1 = β.  Ladder:
      CZ(kα,kβ) + H(kβ);  CZ(Lα,Lβ) + H(Lβ);  H(Lα); CZ(kα, Lα)
    pivot = kα.
    """
    ka, kb = 2 * k, 2 * k + 1
    La, Lb = 2 * L, 2 * L + 1
    pivot = ka
    ladder = []
    # local merge on k (only if both in support)
    if phys_string[ka] != "I" and phys_string[kb] != "I":
        ladder += [("CZ", ka, kb), ("H", kb)]
    elif phys_string[kb] != "I":
        # only beta — swap roles
        pivot = kb
        ladder += [("CZ", kb, ka), ("H", ka)] if phys_string[ka] != "I" else []
    if phys_string[La] != "I" and phys_string[Lb] != "I":
        ladder += [("CZ", La, Lb), ("H", Lb)]
    # open L hub and cross link
    hub_L = La if phys_string[La] != "I" else Lb
    ladder += [("H", hub_L), ("CZ", pivot, hub_L)]

    _, lad = gen._frame(ladder, pivot, n)
    basis = []
    for q in range(n):
        if phys_string[q] != "I":
            basis += gen._basis_for(lad[q], phys_string[q], q)
    prefix = basis + ladder
    ph, letters = gen._frame(prefix, pivot, n)
    assert letters == list(phys_string), ("".join(letters), phys_string)
    assert ph in (1, -1)
    return prefix, pivot, ph


# ----------------------------------------------------------------------
# Method E: commuting-set / single Clifford (Method 4 lite)
# For mutually commuting strings, keep one shared ladder to the common
# bridge and only rewrite the differing leaves between RZs.
# This is what hub-continuity + peephole already approximate; we force the
# bridge to stay fixed and drop full uncompute between blocks manually.
# ----------------------------------------------------------------------
def method_shared_clifford(strings, signs=None, bridge=None, cross_pairs=None):
    """Compile commuting Paulis with a pinned bridge and no forced uncompute.

    Construction: C · Π_k RZ_pivot(θ_k) · C† is WRONG when Paulis differ.
    Instead: use standard sandwich but pin hubs so peephole maximises cancel.
    Plus: between rotations, only emit the difference of prefixes.
    """
    n = len(strings[0])
    if signs is None:
        signs = [1] * len(strings)
    if cross_pairs is None:
        cross_pairs = spin_split_cross_pairs(n)
    frozen = fully_freeze(strings)

    # pick bridge
    if bridge is None:
        half = n // 2
        common = set(range(n))
        for s in frozen:
            common &= {q for q, p in enumerate(s) if p != "I"}
        alphas = [q for q in common if q < half]
        betas = [q for q in common if q >= half]
        if alphas and betas:
            # prefer vertical
            vert = [(a, a + half) for a in alphas if a + half in betas]
            bridge = vert[0] if vert else (alphas[0], betas[0])
        else:
            return []

    ha, hb = bridge
    # Build prefixes for each string with pinned hubs
    prefixes = []
    pivots = []
    phs = []
    for s in frozen:
        prefix, pivot, ph, _ = compile_flexible(s, n, hub_a=ha, hub_b=hb)
        prefixes.append(prefix)
        pivots.append(pivot)
        phs.append(ph)

    # Difference compilation: C0, ROT0, C0†C1, ROT1, C1†C2, ROT2, C2†
    prog = []
    expected = []
    prog += prefixes[0]
    prog.append(("ROT", pivots[0], 0, signs[0] * phs[0]))
    expected.append((frozen[0], phs[0]))
    for k in range(1, len(frozen)):
        # interface = prefixes[k-1]† · prefixes[k]
        interface = gen._invert(prefixes[k - 1]) + prefixes[k]
        interface = gen._peephole(interface)
        prog += interface
        prog.append(("ROT", pivots[k], k, signs[k] * phs[k]))
        expected.append((frozen[k], phs[k]))
    prog += gen._invert(prefixes[-1])
    prog = gen._peephole(prog)
    try:
        gen._verify_program(prog, n, expected)
    except AssertionError:
        # fall back to full sandwiches with pinned hubs
        return search_flexible_hubs(
            frozen, signs=signs, cross_pairs=cross_pairs
        )

    gates = gen.fuse_cz_rot_cz_to_rzx(prog_to_gates(prog))
    bud = score_gates(gates, cross_pairs)
    return [{
        "name": "shared_clifford_pinned",
        "bridge": bridge,
        "frozen_strings": frozen,
        "gates": gates,
        "budget": bud,
    }]


# ----------------------------------------------------------------------
# Method F: disjoint RZX + all qubits used (general rule for Cl2+)
# ----------------------------------------------------------------------
def method_disjoint_rzx_all_qubits(strings, signs=None, cross_pairs=None,
                                   max_mask_opts=6):
    """Selective Z-freeze + vertex-disjoint α–β RZX, using every qubit.

    Enumerates keep-masks per string (courier Z-pairs) and hub schedules whose
    bridges are pairwise qubit-disjoint, then keeps circuits that touch every
    wire.  Ranked by error-budget proxy.
    """
    n = len(strings[0])
    if signs is None:
        signs = [1] * len(strings)
    if cross_pairs is None:
        cross_pairs = spin_split_cross_pairs(n)

    mask_lists = []
    for s in strings:
        opts = list(all_keep_masks(s))
        # prefer fewer couriers first (cheaper), but keep enough variety
        opts.sort(key=lambda x: len(x[0]))
        mask_lists.append(opts[:max_mask_opts])

    winners = []
    best = None
    for combo in itertools.product(*mask_lists):
        frozen = [c[1] for c in combo]
        keeps = [sorted(c[0]) for c in combo]
        # union support must cover all qubits OR couriers will — still search
        hits = search_flexible_hubs(
            frozen,
            signs=signs,
            cross_pairs=cross_pairs,
            require_disjoint_rzx=True,
            require_all_qubits=True,
            order="given",  # keep t_k ↔ doubles[k]
        )
        for h in hits:
            if not h.get("rules_ok"):
                continue
            rec = dict(h)
            rec["name"] = "disjoint_rzx_all_qubits"
            rec["frozen_strings"] = frozen
            rec["keep_pairs"] = keeps
            winners.append(rec)
            if best is None or rec["budget"].error < best["budget"].error:
                best = rec
    if not winners:
        return []
    winners.sort(key=lambda r: r["budget"].error)
    # unique by (frozen, schedule)
    uniq, seen = [], set()
    for w in winners:
        key = (tuple(w["frozen_strings"]), tuple(w["schedule"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(w)
    return uniq[:8]
