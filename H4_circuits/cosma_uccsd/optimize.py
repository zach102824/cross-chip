"""End-to-end COSMA-faithful optimization for full-linked H4 UCCSD."""

from __future__ import annotations

from dataclasses import dataclass, field

from .emit import CircuitScore, best_score_over_seeds
from .majorana import excitation_generator_majorana, majorana_to_pauli
from .mapping import MappingBasis, build_mapping, run_tree_ga
from .schedule import (
    PauliFactor,
    expand_operator_to_factors,
    factors_all_commute,
    support_delta,
    topological_gray_schedule,
)
from .taper_path import (
    build_mapping_taper_data,
    jw_export_taper,
    relabel_op_tencirchem_to_export,
    taper_pauli_operator,
)
from .workload import H4Workload, load_h4_workload


@dataclass
class PathResult:
    path: str
    mapping_name: str
    score: CircuitScore
    factors: list[PauliFactor]
    seed: int
    mapping: dict
    notes: list[str] = field(default_factory=list)
    exact_per_excitation: bool = True

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "mapping_name": self.mapping_name,
            "score": self.score.to_dict(),
            "seed": self.seed,
            "mapping": self.mapping,
            "notes": self.notes,
            "exact_per_excitation": self.exact_per_excitation,
            "n_factors": len(self.factors),
            "support_delta": support_delta(self.factors),
            "factor_labels": [f.label for f in self.factors],
            "paulis": [f.pauli for f in self.factors],
            "pids": [f.pid for f in self.factors],
            "angle_signs": [f.angle_sign for f in self.factors],
        }


def map_all_excitations(
    workload: H4Workload, basis: MappingBasis
) -> tuple[list[PauliFactor], list[str], bool]:
    notes: list[str] = []
    exact = True
    raw_factors: list[PauliFactor] = []
    for exc in workload.linked_excitations():
        maj = excitation_generator_majorana(exc.ex_op)
        op = majorana_to_pauli(maj, basis)
        facs = expand_operator_to_factors(
            label=f"pid{exc.pid}_op{exc.ex_op_index}",
            pid=exc.pid,
            ex_op_index=exc.ex_op_index,
            theta_name=exc.theta_name,
            pauli_op=op,
        )
        if not facs:
            notes.append(f"empty mapping for pid={exc.pid} op={exc.ex_op_index}")
            exact = False
            continue
        if not factors_all_commute(facs):
            notes.append(
                f"noncommuting Pauli sum for pid={exc.pid} op={exc.ex_op_index}; Trotter product"
            )
            exact = False
        raw_factors.extend(facs)
    return topological_gray_schedule(raw_factors), notes, exact


def _taper_factors(
    factors: list[PauliFactor],
    basis: MappingBasis,
    notes: list[str],
) -> tuple[list[PauliFactor], bool]:
    """Return tapered factors on 6 qubits and exactness flag."""
    exact = True
    # Prefer mapping-dependent taper; for JW use export-layout taper_lib.
    taper = build_mapping_taper_data(basis)
    use_export = False
    if taper is None:
        if basis.name != "JW":
            notes.append("mapping-dependent taper unavailable")
            return [], False
        taper = jw_export_taper()
        use_export = True
        notes.append("6q JW via export-layout taper_lib")
    else:
        notes.append(f"6q mapping-dependent taper removed={taper.removed_qubits}")

    tapered: list[PauliFactor] = []
    for fac in factors:
        op = {fac.pauli: fac.coeff}
        if use_export:
            op = relabel_op_tencirchem_to_export(op)
        try:
            top = taper_pauli_operator(op, taper)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"taper failed for {fac.label}: {exc}")
            exact = False
            continue
        new_facs = expand_operator_to_factors(
            fac.label, fac.pid, fac.ex_op_index, fac.theta_name, top
        )
        if not new_facs:
            notes.append(f"taper dropped {fac.label}")
            exact = False
            continue
        if not factors_all_commute(new_facs):
            exact = False
        tapered.extend(new_facs)
    return topological_gray_schedule(tapered), exact


def compile_path(
    workload: H4Workload,
    basis: MappingBasis,
    n_qubits: int,
    path_name: str,
    seeds=None,
) -> PathResult:
    factors, notes, exact = map_all_excitations(workload, basis)
    if n_qubits == 6:
        factors, exact_t = _taper_factors(factors, basis, notes)
        exact = exact and exact_t
        if not factors:
            bad = CircuitScore(10**9, 10**9, 10**9, 6, 0, 10**9)
            return PathResult(path_name, basis.name, bad, [], -1, basis.to_dict(), notes, False)

    score, _logical, _routed, seed = best_score_over_seeds(factors, n_qubits, seeds=seeds)
    return PathResult(path_name, basis.name, score, factors, seed, basis.to_dict(), notes, exact)


def optimize_h4(
    encodings: list[str] | None = None,
    ga_population: int = 16,
    ga_generations: int = 8,
    ga_seed: int = 42,
    transpile_seeds: range | None = None,
) -> dict:
    if encodings is None:
        encodings = ["JW", "PE", "JKMN", "TREE_GA"]
    if transpile_seeds is None:
        transpile_seeds = range(6)

    workload = load_h4_workload(1.0)
    results_8: list[PathResult] = []
    results_6: list[PathResult] = []
    bases: list[MappingBasis] = []

    for enc in encodings:
        enc_u = enc.upper()
        print(f"[optimize_h4] building basis {enc_u}...", flush=True)
        if enc_u == "TREE_GA":

            def fitness(basis: MappingBasis) -> float:
                facs, _, _ = map_all_excitations(workload, basis)
                w = sum(p.count("X") + p.count("Y") + p.count("Z") for p in (f.pauli for f in facs))
                return -float(support_delta(facs)) - 0.01 * w

            basis, _fit = run_tree_ga(
                workload.n_modes,
                fitness,
                population=ga_population,
                generations=ga_generations,
                seed=ga_seed,
            )
            bases.append(basis)
            print(f"[optimize_h4] TREE_GA done (fit={_fit:.2f})", flush=True)
        else:
            bases.append(build_mapping(enc, workload.n_modes))

    for basis in bases:
        print(f"[optimize_h4] compiling 8q path for {basis.name}...", flush=True)
        results_8.append(
            compile_path(workload, basis, n_qubits=8, path_name="8q", seeds=transpile_seeds)
        )
        print(f"[optimize_h4] compiling 6q path for {basis.name}...", flush=True)
        results_6.append(
            compile_path(workload, basis, n_qubits=6, path_name="6q", seeds=transpile_seeds)
        )
        sc8 = results_8[-1].score
        sc6 = results_6[-1].score
        print(
            f"[optimize_h4] {basis.name}: 8q cross={sc8.cross_chip_2q} "
            f"total2q={sc8.total_2q} | 6q cross={sc6.cross_chip_2q} total2q={sc6.total_2q}",
            flush=True,
        )

    def best_of(rows: list[PathResult]) -> PathResult:
        return min(rows, key=lambda r: r.score.key())

    valid6 = [r for r in results_6 if r.score.cross_chip_2q < 10**8]
    return {
        "workload": {
            "pids": workload.pids,
            "n_linked_excitations": len(workload.linked_excitations()),
            "n_modes": workload.n_modes,
            "bond_length_A": workload.bond_length_A,
        },
        "results_8q": [r.to_dict() for r in results_8],
        "results_6q": [r.to_dict() for r in results_6],
        "best_8q": best_of(results_8).to_dict(),
        "best_6q": best_of(valid6 or results_6).to_dict(),
        "approx_8gadget_baseline": _load_approx_baseline(),
        "objective": "lexicographic (cross_chip_2q, total_2q, depth)",
        "implementation_note": (
            "Python port of COSMA PPTT mapping / Gray-legal schedule / core-local "
            "parity trees; native cosma C++ failed to build (missing C++ SDK headers)."
        ),
    }


def _load_approx_baseline() -> dict:
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "H4_8doubles_diagframe.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {
        "note": "first-linked-op-only approx (NOT full TenCirChem linked generators)",
        "logical_2q": data.get("logical_budget", {}).get("two_qubit_gates"),
        "routed_2q": data.get("routed_budget", {}).get("two_qubit_gates"),
        "cross_chip_2q": data.get("routed_budget", {}).get("cross_chip_2q"),
        "depth": data.get("routed_budget", {}).get("depth"),
    }
