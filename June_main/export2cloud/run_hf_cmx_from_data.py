#!/usr/bin/env python3
"""Run HF CMX / CME(k=3) from saved VQE data (standalone, OGM measurement).

This is the standalone version of the CMX cell in ``June_main/main_HF.ipynb``.
It does NOT run VQE. It loads ``params_final`` from a saved run folder
(``data/HF_bond_<R>/vqe_results.json``), rebuilds the same HF prep + ansatz
circuit, measures <H>, <H^2>, <H^3> with the SAME CDR+REM pipeline using the
OGM measurement scheme, runs the connected-moment expansion, and saves the CMX
results back into the run folder.

All Hamiltonian and OGM basis files are read from THIS directory
(``June_main/export2cloud``):
  - Hamiltonians : ``Pauli_Ham/HF_bond_<R>.txt`` / ``HF_square_bond_<R>.txt`` / ``HF_triple_bond_<R>.txt``
  - OGM bases    : ``June_main/OGM_measurement_basis/OGM_HF_bond_<R>.txt`` / ``..._square_...`` / ``..._triple_...``

Run, e.g.:
    python run_hf_cmx_from_data.py
    python run_hf_cmx_from_data.py --run-dir data/HF_bond_2.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import cirq
import numpy as np
import sympy
from cirq.ops import GlobalPhaseGate

setattr(cirq, "GlobalPhaseGate", GlobalPhaseGate)

# Everything (Pauli_Ham, OGM bases, circuits, shared library) is resolved
# relative to this file's directory, exactly like the cloud export main_HF.py.
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

from cloud_results import save_checkpoint  # noqa: E402

for _p in (str(BASE_DIR / "June_main"), str(BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from main_cursor_lib import (  # noqa: E402
    CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    ONE_QUBIT_GATE_DEPOL_PROB,
    TWO_QUBIT_GATE_DEPOL_PROB,
    GateArityDepolarizingNoise,
    RZXGate,
    trace_energy,
)
from shot_measurement import run_mitigation, sanitize_density_matrix  # noqa: E402


MOLECULE = "HF"
CIRCUIT_NAME = "HF_8q_3doubles_rzx"
N_ACTIVE_ELECTRONS = 6
N_SPATIAL_ORBITALS = 4
N_QUBITS = 2 * N_SPATIAL_ORBITALS
ETA = N_ACTIVE_ELECTRONS // 2
CROSS_CHIP_QUBIT_PAIRS = {(2, 6)}
CZ_CROSS_CHIP_TAG = "cz_cross_chip"

# Local data directories (all under June_main/export2cloud).
PAULI_HAM_DIR = BASE_DIR / "Pauli_Ham"
OGM_DIR = BASE_DIR / "June_main" / "OGM_measurement_basis"
CIRCUITS_DIR = BASE_DIR / "June_main" / "circuits2read"

_READOUT_P0_TEMPLATE = np.array([0.97, 0.96, 0.93, 0.96, 0.92, 0.93, 0.94, 0.92], dtype=float)
_READOUT_P1_TEMPLATE = np.array([0.85, 0.90, 0.88, 0.90, 0.86, 0.89, 0.87, 0.85], dtype=float)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def infer_bond_length(run_dir: Path, explicit_bond: float | None) -> float:
    if explicit_bond is not None:
        return float(explicit_bond)

    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        if "bond_length" in summary:
            return float(summary["bond_length"])

    marker = "_bond_"
    if marker in run_dir.name:
        return float(run_dir.name.split(marker, 1)[1])

    raise ValueError(
        "Could not infer bond length. Pass --bond-length or use a run directory "
        "named like HF_bond_2.2."
    )


def saved_metadata(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return {}
    return dict(load_json(summary_path).get("metadata", {}))


def is_cross_chip_pair(qubit_indices: list[int]) -> bool:
    return frozenset(qubit_indices) in {frozenset(pair) for pair in CROSS_CHIP_QUBIT_PAIRS}


def load_circuit_from_json(
    path: Path,
) -> tuple[cirq.Circuit, list[cirq.LineQubit], list[sympy.Symbol], dict[str, Any]]:
    data = load_json(path)
    qubits = cirq.LineQubit.range(int(data["num_qubits"]))
    param_names = list(data["param_names"])
    if not param_names:
        for gate in data["gates"]:
            name = gate.get("param")
            if name is not None and name not in param_names:
                param_names.append(name)

    symbol_by_name = {name: sympy.Symbol(f"th_{i}") for i, name in enumerate(param_names)}
    symbols = [symbol_by_name[name] for name in param_names]

    circuit = cirq.Circuit()
    for gate in data["gates"]:
        op_name = gate["op"]
        qs = [qubits[i] for i in gate["qubits"]]
        if op_name == "h":
            op = cirq.H(qs[0])
        elif op_name == "x":
            op = cirq.X(qs[0])
        elif op_name == "ry":
            op = cirq.ry(float(gate["value"])).on(qs[0])
        elif op_name == "cx":
            op = cirq.CNOT(qs[0], qs[1])
        elif op_name == "cz":
            op = cirq.CZ(qs[0], qs[1])
        elif op_name == "rzx":
            if "param" in gate:
                angle = float(gate.get("coeff", 1.0)) * symbol_by_name[gate["param"]]
            else:
                angle = float(gate.get("angle", gate.get("value", 0.0)))
            op = RZXGate(angle).on(qs[0], qs[1])
        elif op_name in ("rx", "rz"):
            angle = (
                float(gate["coeff"]) * symbol_by_name[gate["param"]]
                if "param" in gate
                else float(gate["value"])
            )
            op = (cirq.rx if op_name == "rx" else cirq.rz)(angle).on(qs[0])
        else:
            raise ValueError(f"Unhandled op in circuit JSON: {op_name!r}")

        if len(op.qubits) == 2 and is_cross_chip_pair(gate["qubits"]):
            op = op.with_tags(CZ_CROSS_CHIP_TAG)
        circuit.append(op)

    return circuit, list(qubits), symbols, data


def load_pauli_sum_from_numbered_file(path: Path, qubits: list[cirq.Qid]) -> cirq.PauliSum:
    idx_to_pauli = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    out = cirq.PauliSum()

    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            coeff = float(parts[0])
            pauli_codes = [int(x) for x in parts[1:]]
            if len(pauli_codes) != len(qubits):
                raise ValueError(
                    f"{path}:{lineno} has {len(pauli_codes)} Pauli indices, expected {len(qubits)}."
                )

            pauli_string = cirq.PauliString()
            for qubit, code in zip(qubits, pauli_codes):
                if code == 0:
                    continue
                if code not in idx_to_pauli:
                    raise ValueError(f"{path}:{lineno} has invalid Pauli code {code}; expected 0/1/2/3.")
                pauli_string *= idx_to_pauli[code](qubit)
            out += coeff * pauli_string

    return out


def hf_prep_circuit(qubits: list[cirq.Qid]) -> cirq.Circuit:
    """HF determinant: X on each occupied spin-orbital."""
    occupied = list(range(ETA)) + list(range(N_SPATIAL_ORBITALS, N_SPATIAL_ORBITALS + ETA))
    return cirq.Circuit([cirq.X(qubits[k]) for k in occupied])


def cme_k3(h1: float, h2: float, h3: float) -> tuple[float, float, float, float]:
    c1 = h1
    c2 = h2 - h1**2
    c3 = h3 - 3.0 * h1 * h2 + 2.0 * h1**3
    energy = float("nan") if abs(c3) < 1e-12 else c1 - (c2**2) / c3
    return energy, c1, c2, c3


def parse_multiplier_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one multiplier, e.g. 1,5,10,15")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=BASE_DIR / "data" / "HF_bond_2.2",
        help="Saved run directory containing vqe_results.json (relative paths are resolved under this script's dir).",
    )
    parser.add_argument("--bond-length", type=float, default=None, help="Override bond length inferred from run data.")
    parser.add_argument("--h-shots", type=int, default=None, help="Shot budget for <H>; defaults to saved num_shots.")
    parser.add_argument(
        "--multipliers",
        type=parse_multiplier_list,
        default=parse_multiplier_list("1,5,10,15"),
        help="Comma-separated <H^2>/<H^3> shot multipliers, e.g. 1,5,10,15.",
    )
    parser.add_argument("--moment-source", default="cdr_rem", choices=["cdr_rem", "cdr_unmit", "rem", "unmit"])
    parser.add_argument("--cdr-training-circuits", type=int, default=None)
    parser.add_argument("--cdr-t-max", type=int, default=None)
    parser.add_argument("--cdr-seed", type=int, default=int(os.environ.get("CDR_BASE_SEED", "42")))
    parser.add_argument("--simulator-seed", type=int, default=int(os.environ.get("GLOBAL_RANDOM_SEED", "1234")))
    parser.add_argument("--sampling-seed", type=int, default=int(os.environ.get("GLOBAL_SAMPLING_SEED", "1234")))
    parser.add_argument(
        "--shadowgrouping-root",
        type=Path,
        default=Path(os.environ.get("SHADOWGROUPING_ROOT", "/Users/zacharyhe/shadowgrouping")),
        help="Path to the shadowgrouping package (required by the OGM measurement scheme).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    raw_run_dir = args.run_dir.expanduser()
    run_dir = raw_run_dir if raw_run_dir.is_absolute() else (BASE_DIR / raw_run_dir)
    run_dir = run_dir.resolve()
    vqe_path = run_dir / "vqe_results.json"
    require_file(vqe_path, "saved VQE results")

    bond_length = infer_bond_length(run_dir, args.bond_length)
    metadata = saved_metadata(run_dir)

    h_num_shots = int(args.h_shots or metadata.get("num_shots", os.environ.get("GLOBAL_NUM_SHOTS", 8192)))
    cdr_training_circuits = int(
        args.cdr_training_circuits
        if args.cdr_training_circuits is not None
        else metadata.get("cdr_training_circuits", os.environ.get("CDR_NUM_TRAINING_CIRCUITS", 30))
    )
    cdr_t_max = int(
        args.cdr_t_max
        if args.cdr_t_max is not None
        else os.environ.get("CDR_T_MAX_VQE", os.environ.get("CDR_T_MAX_GRADIENT", 2))
    )

    # OGM requires the shadowgrouping package (same as the notebook).
    shadowgrouping_root = args.shadowgrouping_root.expanduser()
    if not shadowgrouping_root.is_dir():
        raise RuntimeError(
            f"SHADOWGROUPING_ROOT does not exist: {shadowgrouping_root}. "
            "The OGM measurement scheme needs it; pass --shadowgrouping-root."
        )

    vqe_results = load_json(vqe_path)
    circuit_json = CIRCUITS_DIR / f"{CIRCUIT_NAME}.json"
    require_file(circuit_json, "HF circuit JSON")
    ansatz_circuit, qubits, symbols, circuit_meta = load_circuit_from_json(circuit_json)
    if len(qubits) != N_QUBITS:
        raise ValueError(f"Circuit has {len(qubits)} qubits, expected {N_QUBITS}.")

    circuit = hf_prep_circuit(qubits) + ansatz_circuit
    n_params = len(symbols)
    params_final = np.asarray(vqe_results["params_final"], dtype=float).reshape(n_params)

    def resolver_from_params(params_vec: np.ndarray) -> dict[sympy.Symbol, float]:
        params = np.asarray(params_vec, dtype=float).reshape(n_params)
        return {symbols[i]: float(params[i]) for i in range(n_params)}

    # Hamiltonian (H / H^2 / H^3) + matching OGM basis files, both from this dir.
    ham_paths = {
        "H": PAULI_HAM_DIR / f"{MOLECULE}_bond_{bond_length:.1f}.txt",
        "H2": PAULI_HAM_DIR / f"{MOLECULE}_square_bond_{bond_length:.1f}.txt",
        "H3": PAULI_HAM_DIR / f"{MOLECULE}_triple_bond_{bond_length:.1f}.txt",
    }
    ogm_paths = {
        "H": OGM_DIR / f"OGM_{MOLECULE}_bond_{bond_length:.1f}.txt",
        "H2": OGM_DIR / f"OGM_{MOLECULE}_square_bond_{bond_length:.1f}.txt",
        "H3": OGM_DIR / f"OGM_{MOLECULE}_triple_bond_{bond_length:.1f}.txt",
    }
    for key in ("H", "H2", "H3"):
        require_file(ham_paths[key], f"{key} Hamiltonian")
        require_file(ogm_paths[key], f"{key} OGM basis")

    pauli_sums = {key: load_pauli_sum_from_numbered_file(path, list(qubits)) for key, path in ham_paths.items()}
    e_gs = float(np.linalg.eigvalsh(pauli_sums["H"].matrix(qubits=qubits))[0].real)

    base_noise_cfg = {
        "two_qubit_depol_prob": float(TWO_QUBIT_GATE_DEPOL_PROB),
        "one_qubit_depol_prob": float(ONE_QUBIT_GATE_DEPOL_PROB),
        "cross_chip_two_qubit_depol_prob": float(CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB),
    }
    readout_cal = {
        "p_0_success": np.resize(_READOUT_P0_TEMPLATE, N_QUBITS).astype(float),
        "p_1_success": np.resize(_READOUT_P1_TEMPLATE, N_QUBITS).astype(float),
    }
    cdr_cfg = {
        "num_circuits": cdr_training_circuits,
        "t_max": cdr_t_max,
        "seed": int(args.cdr_seed),
    }
    base_shot_cfg = {
        "measurement_scheme": "ogm",
        "apply_readout_noise": True,
        "sampling_seed": int(args.sampling_seed),
        "shadowgrouping_root": str(shadowgrouping_root),
    }

    final_resolver = cirq.ParamResolver(resolver_from_params(params_final))

    # Rebuild the FINAL noisy density matrix (kept only for the exact Tr[H^k rho]
    # reference), and the exact noiseless statevector at the same final params.
    gate_noise = GateArityDepolarizingNoise(**base_noise_cfg)
    resolved_noisy = cirq.resolve_parameters(circuit.with_noise(gate_noise), final_resolver)
    rho_final = np.asarray(
        cirq.DensityMatrixSimulator(seed=int(args.simulator_seed))
        .simulate(resolved_noisy, qubit_order=qubits)
        .final_density_matrix,
        dtype=np.complex128,
    )
    rho_final = sanitize_density_matrix(rho_final)

    psi_noiseless = np.asarray(
        cirq.Simulator(dtype=np.complex128)
        .simulate(cirq.resolve_parameters(circuit, final_resolver), qubit_order=qubits)
        .final_state_vector,
        dtype=np.complex128,
    )
    qubit_map = {q: i for i, q in enumerate(qubits)}

    def measure_moment(key: str, num_shots: int) -> dict[str, float]:
        shot_cfg = dict(base_shot_cfg)
        shot_cfg["num_shots"] = int(num_shots)
        shot_cfg["ogm_file"] = ogm_paths[key]

        mit = run_mitigation(
            "cdr",
            ansatz_circuit=circuit,
            observable_h=pauli_sums[key],
            qubits=qubits,
            target_resolver=final_resolver,
            target_params=final_resolver,
            symbols=symbols,
            base_noise_cfg=base_noise_cfg,
            shot_cfg=shot_cfg,
            readout_cal=readout_cal,
            cdr_cfg=cdr_cfg,
            simulator_seed=int(args.simulator_seed),
        )
        result = {
            "unmit": float(mit["unmit_target"]),
            "rem": float(mit["rem_target"]),
            "cdr_unmit": float(mit["cdr_unmit_corrected"]),
            "cdr_rem": float(mit["cdr_rem_corrected"]),
            "exact": float(trace_energy(pauli_sums[key].matrix(qubits=qubits), rho_final)),
            "noiseless": float(
                np.real(pauli_sums[key].expectation_from_state_vector(psi_noiseless, qubit_map=qubit_map))
            ),
        }
        print(
            f"<{key:<2}>  shots={int(num_shots):>8d}  "
            f"cdr+rem={result['cdr_rem']:+.8f}  rem={result['rem']:+.8f}  "
            f"unmit={result['unmit']:+.8f}  noiseless={result['noiseless']:+.8f}  "
            f"noisy_exact={result['exact']:+.8f}  "
            f"|cdr+rem - noiseless|={abs(result['cdr_rem'] - result['noiseless']):.3e}"
        )
        return result

    print(f"Loaded saved HF VQE data : {vqe_path}")
    print(f"bond_length={bond_length:.1f}  params_final={params_final.tolist()}")
    print(f"measurement_scheme=ogm  <H> shots={h_num_shots}  multipliers={args.multipliers}")
    print(f"true ground-state energy e_gs = {e_gs:.10f} Eh")

    # Measure <H> ONCE (its shot budget is fixed across the multiplier sweep).
    moments = {"H": measure_moment("H", h_num_shots)}

    cme_results_by_multiplier: dict[int, dict[str, Any]] = {}
    for mult in args.multipliers:
        h2_h3_shots = int(mult * h_num_shots)
        print(f"\n--- <H^2>, <H^3> shots = {mult}x <H> shots = {h2_h3_shots} ---")
        moments["H2"] = measure_moment("H2", h2_h3_shots)
        moments["H3"] = measure_moment("H3", h2_h3_shots)

        src = {key: moments[key][args.moment_source] for key in ("H", "H2", "H3")}
        e_cme, c1, c2, c3 = cme_k3(src["H"], src["H2"], src["H3"])
        e_cme_exact, _, _, _ = cme_k3(moments["H"]["exact"], moments["H2"]["exact"], moments["H3"]["exact"])
        e_cme_noiseless, _, _, _ = cme_k3(
            moments["H"]["noiseless"], moments["H2"]["noiseless"], moments["H3"]["noiseless"]
        )

        print(f"=== Connected Moment Expansion (k=3), <H^2>/<H^3> shot multiplier = {mult}x ===")
        print(f"moment source for formula : {args.moment_source}")
        print(f"<H>={src['H']:+.8f}  <H^2>={src['H2']:+.8f}  <H^3>={src['H3']:+.8f}")
        print(f"connected moments         : c1={c1:+.6e}  c2={c2:+.6e}  c3={c3:+.6e}")
        print(f"E_CME(k=3) [shots/{args.moment_source}]      = {e_cme:.10f} Eh")
        print(f"E_CME(k=3) [noiseless moments] = {e_cme_noiseless:.10f} Eh")
        print(f"E_CME(k=3) [exact Tr moments]  = {e_cme_exact:.10f} Eh")
        print(f"|E_CME - e_gs|       (shots)    = {abs(e_cme - e_gs):.6e} Eh")

        cme_results_by_multiplier[mult] = {
            "params_final": params_final.copy(),
            "shots": {"H": h_num_shots, "H2": h2_h3_shots, "H3": h2_h3_shots},
            "moments": {key: dict(value) for key, value in moments.items()},
            "connected_moments": {"c1": float(c1), "c2": float(c2), "c3": float(c3)},
            "E_cme_shots": float(e_cme),
            "E_cme_noiseless": float(e_cme_noiseless),
            "E_cme_exact": float(e_cme_exact),
            "e_gs": float(e_gs),
            "moment_source": args.moment_source,
            "h2_h3_shot_multiplier": int(mult),
        }

    print("\n=== CME(k=3) summary across <H^2>/<H^3> shot multipliers ===")
    for mult in args.multipliers:
        r = cme_results_by_multiplier[mult]
        print(
            f"{mult:>3d}x (shots={r['shots']['H2']:>8d})  "
            f"E_CME={r['E_cme_shots']:.10f} Eh  |E_CME - e_gs|={abs(r['E_cme_shots'] - e_gs):.6e} Eh"
        )

    cme_results = cme_results_by_multiplier[args.multipliers[-1]]
    written = save_checkpoint(
        data_dir=run_dir.parent,
        molecule=MOLECULE,
        bond_length=float(bond_length),
        stage="cmx_from_saved_vqe",
        vqe_results=vqe_results,
        cme_results=cme_results,
        cme_results_by_multiplier=cme_results_by_multiplier,
        metadata={
            **metadata,
            "circuit_name": CIRCUIT_NAME,
            "measurement_scheme": "ogm",
            "num_shots": int(h_num_shots),
            "cdr_training_circuits": int(cdr_training_circuits),
            "cdr_t_max": int(cdr_t_max),
            "source_run_dir": str(run_dir),
            "source_vqe_results": str(vqe_path),
            "script": Path(__file__).name,
            "circuit_num_qubits": len(qubits),
            "circuit_num_params": n_params,
            "circuit_json": str(circuit_json),
            "circuit_meta_num_gates": len(circuit_meta.get("gates", [])),
        },
    )
    print("\nSaved CMX result files:")
    for name, path in sorted(written.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
