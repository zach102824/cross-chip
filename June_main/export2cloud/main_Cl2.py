#!/usr/bin/env python3
"""Cloud-runnable export generated from June_main/main_HF.ipynb.

Run from this directory, for example: python main_Cl2.py
Results are written under data/Cl2_bond_<R>/. Override bond length with
CL2_BOND_LENGTH (e.g. CL2_BOND_LENGTH=1.2 python main_Cl2.py).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.chdir(Path(__file__).resolve().parent)

from cloud_results import save_checkpoint


# This cell is to read the circuit from the file 
# and set up some basic parameters and mole
# The experimental equilibrium bond length ($R_e$) of the HF molecule is approximately 0.917 Å. 
import cirq
import numpy as np
from cirq.ops import GlobalPhaseGate

# Needed for text/SVG diagrams: top-level ``cirq.GlobalPhaseGate`` can be missing in some kernels.
setattr(cirq, "GlobalPhaseGate", GlobalPhaseGate)
import sympy


# Shared sampling/CDR settings used across later notebook cells.
GLOBAL_NUM_SHOTS = int(os.environ.get("GLOBAL_NUM_SHOTS", "8192"))
CDR_NUM_TRAINING_CIRCUITS = int(os.environ.get("CDR_NUM_TRAINING_CIRCUITS", "30"))
CDR_T_MAX_GRADIENT = int(os.environ.get("CDR_T_MAX_GRADIENT", "2"))
CDR_T_MAX_VQE = int(os.environ.get("CDR_T_MAX_VQE", str(CDR_T_MAX_GRADIENT)))
CDR_BASE_SEED = int(os.environ.get("CDR_BASE_SEED", "42"))
GLOBAL_RANDOM_SEED = int(os.environ.get("GLOBAL_RANDOM_SEED", "1234"))
GLOBAL_SAMPLING_SEED = int(os.environ.get("GLOBAL_SAMPLING_SEED", "1234"))
GLOBAL_MEASUREMENT_SCHEME = os.environ.get("MEASUREMENT_SCHEME", "ogm")
GLOBAL_APPLY_READOUT_NOISE = True
GLOBAL_READOUT_P0_SUCCESS = np.array([0.97, 0.96, 0.93, 0.96, 0.92, 0.93, 0.94, 0.92])
GLOBAL_READOUT_P1_SUCCESS = np.array([0.85, 0.90, 0.88, 0.90, 0.86, 0.89, 0.87, 0.85])


import json
from pathlib import Path

# Which molecule / bond length to load. The circuit JSON is produced by the
# molecule notebook (e.g. UCCSD_Mole/HF.ipynb) via uccsd_circuit_io.build_and_save.
MOLECULE = "Cl2"
BOND_LENGTH = float(os.environ.get("CL2_BOND_LENGTH", "2.2"))
bond_length = BOND_LENGTH  # alias used by later cells (OGM / Hamiltonian paths)

# Active space for the loaded molecule. These MUST match the molecule notebook
# that produced the circuit / Hamiltonian (e.g. UCCSD_Mole/HF.ipynb uses 6
# electrons in 4 spatial orbitals for HF). They drive the HF-determinant
# reference and the qubit-count sanity checks below.
N_ACTIVE_ELECTRONS = 8              # active electrons
N_SPATIAL_ORBITALS = 5             # active spatial orbitals
N_QUBITS = 2 * N_SPATIAL_ORBITALS  # spin-orbitals == qubits
ETA = N_ACTIVE_ELECTRONS // 2      # occupied spatial orbitals per spin sector

_READOUT_P0_TEMPLATE = np.array([0.97, 0.96, 0.93, 0.96, 0.92, 0.93, 0.94, 0.92], dtype=float)
_READOUT_P1_TEMPLATE = np.array([0.85, 0.90, 0.88, 0.90, 0.86, 0.89, 0.87, 0.85], dtype=float)
GLOBAL_READOUT_P0_SUCCESS = np.resize(_READOUT_P0_TEMPLATE, N_QUBITS).astype(float)
GLOBAL_READOUT_P1_SUCCESS = np.resize(_READOUT_P1_TEMPLATE, N_QUBITS).astype(float)
print(f"Cloud measurement scheme: {GLOBAL_MEASUREMENT_SCHEME}")

_repo = Path.cwd().resolve()
while not (_repo / "June_main").is_dir() and _repo != _repo.parent:
    _repo = _repo.parent
CIRCUITS_DIR = _repo / "June_main" / "circuits2read"

# Make the shared library importable from this first cell (later cells repeat this).
import sys as _sys
for _p in (str(_repo / "June_main"), str(_repo)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# Parameterized native cross-chip RZX gate (supports sympy angles + resolution and
# is recognized by the CDR Clifford-snapping in main_cursor_lib).
from main_cursor_lib import RZXGate
# Bare ansatz (NO initial-state prep), with cross-chip CZ already decomposed to
# the native RZX form. The reference state (HF / multireference) is prepended in
# the next cell, so this cell just reads + draws the ansatz to confirm it loaded.
CIRCUIT_NAME = "Cl2_10q_3doubles_rzx"
CIRCUIT_JSON = CIRCUITS_DIR / f"{CIRCUIT_NAME}.json"

# Cross-chip two-qubit gates carry this tag so the noise model can apply a higher
# (cross-chip) depolarizing probability.
CZ_CROSS_CHIP_TAG = "cz_cross_chip"

# Which qubit pairs physically span two chips. The circuit JSON no longer carries
# a per-gate "cross_chip" flag; instead, ANY two-qubit gate acting on one of these
# (qubit_i, qubit_j) line-qubit index pairs is treated as a cross-chip link and
# gets the higher CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB noise. Order does not
# matter: (3, 4) and (4, 3) are the same link. Edit this to match the chip layout.
# In HF_8q_3doubles_rzx.json the native RZX entanglers act on qubits 2 and 6, so
# that is the cross-chip link here.
CROSS_CHIP_QUBIT_PAIRS = {(3, 8), (0, 1), (5, 6)}
_CROSS_CHIP_PAIR_SET = {frozenset(pair) for pair in CROSS_CHIP_QUBIT_PAIRS}


def _is_cross_chip_pair(qubit_indices):
    """True if the (unordered) qubit-index pair is a configured cross-chip link."""
    return frozenset(qubit_indices) in _CROSS_CHIP_PAIR_SET


def load_circuit_from_json(path):
    """Build a cirq.Circuit (LineQubit layout) from a saved UCCSD circuit JSON.

    Returns (circuit, qubits, symbols, meta). ANY two-qubit gate (CZ, RZX, CX, ...)
    whose qubit pair is listed in ``CROSS_CHIP_QUBIT_PAIRS`` is tagged with
    ``CZ_CROSS_CHIP_TAG`` so it gets the higher cross-chip depolarizing noise.
    Parameterized RX/RZ/RZX angles use sympy symbols named ``th_<k>`` so the CDR
    Clifford-snapping in ``main_cursor_lib`` treats them as theta-like.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    n = int(data["num_qubits"])
    q = cirq.LineQubit.range(n)
    param_names = list(data["param_names"])
    # Some circuit JSONs (e.g. the RZX-native variant) leave ``param_names`` empty
    # but still reference params (``"param": "t0"``) on their gates. Recover the
    # parameter list from gate references, in order of first appearance.
    if not param_names:
        for g in data["gates"]:
            name = g.get("param")
            if name is not None and name not in param_names:
                param_names.append(name)
    sym = {name: sympy.Symbol(f"th_{i}") for i, name in enumerate(param_names)}
    syms = [sym[name] for name in param_names]

    c = cirq.Circuit()
    for g in data["gates"]:
        op = g["op"]
        qs = [q[i] for i in g["qubits"]]
        if op == "h":
            new_op = cirq.H(qs[0])
        elif op == "x":
            new_op = cirq.X(qs[0])
        elif op == "ry":
            new_op = cirq.ry(float(g["value"])).on(qs[0])
        elif op == "cx":
            new_op = cirq.CNOT(qs[0], qs[1])
        elif op == "cz":
            new_op = cirq.CZ(qs[0], qs[1])
        elif op == "rzx":
            # Native cross-chip gate kept as ONE 2-qubit op. The angle can be a
            # sympy symbol (parameterized ansatz) and is resolved later for VQE,
            # gradients and CDR -- just like the single-qubit rx/rz rotations.
            if "param" in g:
                rzx_angle = float(g.get("coeff", 1.0)) * sym[g["param"]]
            else:
                rzx_angle = float(g.get("angle", g.get("value", 0.0)))
            new_op = RZXGate(rzx_angle).on(qs[0], qs[1])
        elif op in ("rx", "rz"):
            angle = float(g["coeff"]) * sym[g["param"]] if "param" in g else float(g["value"])
            gate = cirq.rx if op == "rx" else cirq.rz
            new_op = gate(angle).on(qs[0])
        else:
            raise ValueError(f"Unhandled op in circuit JSON: {op!r}")

        # Gate-agnostic cross-chip tagging: ANY two-qubit gate (CZ, RZX, CX, ...)
        # acting on a cross-chip pair gets the tag, so the noise model applies the
        # higher CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB. The noise model only cares
        # that the op is 2-qubit and carries the tag -- not which gate it is.
        if len(new_op.qubits) == 2 and _is_cross_chip_pair(g["qubits"]):
            new_op = new_op.with_tags(CZ_CROSS_CHIP_TAG)
        c.append(new_op)
    return c, list(q), syms, data


circuit, qubits, symbols, circuit_meta = load_circuit_from_json(CIRCUIT_JSON)
# Bare ansatz handle: the next cell prepends the reference-state prep and rebinds
# ``circuit`` to (prep + ansatz). Keeping this separate makes that step idempotent.
ansatz_circuit = circuit
num_qubits = len(qubits)
n_params = len(symbols)
qubit_map = {qq: i for i, qq in enumerate(qubits)}

# Sanity-check the loaded circuit against the declared active space.
assert num_qubits == N_QUBITS, (
    f"Circuit has {num_qubits} qubits but active space implies "
    f"N_QUBITS={N_QUBITS} (= 2 * N_SPATIAL_ORBITALS={N_SPATIAL_ORBITALS})."
)
_meta_ne = circuit_meta.get("n_electrons")
if _meta_ne is not None:
    assert int(_meta_ne) == N_ACTIVE_ELECTRONS, (
        f"Circuit JSON n_electrons={int(_meta_ne)} != "
        f"N_ACTIVE_ELECTRONS={N_ACTIVE_ELECTRONS}."
    )


def resolver_from_params(params_vec):
    """Map a length-``n_params`` vector to a {symbol: value} resolver dict."""
    p = np.asarray(params_vec, dtype=float).reshape(n_params)
    return {symbols[i]: float(p[i]) for i in range(n_params)}


# Backward-compatible aliases for cells originally written for the 3-parameter LiH case.
symbols_li_h = symbols
if n_params >= 1:
    theta1 = symbols[0]
if n_params >= 2:
    theta2 = symbols[1]
if n_params >= 3:
    theta3 = symbols[2]

_n_cross = sum(
    1 for g in circuit_meta["gates"]
    if len(g["qubits"]) == 2 and _is_cross_chip_pair(g["qubits"])
)
_n_rzx = sum(1 for g in circuit_meta["gates"] if g["op"] == "rzx")
print(
    f"Loaded bare ansatz: {num_qubits} qubits, {n_params} params, "
    f"{_n_rzx} RZX, {_n_cross} cross-chip 2Q gates  ({CIRCUIT_JSON.name})"
)

# %% Notebook cell 1

import sys
from pathlib import Path

# Repo root resolution for local file loading (root holds Pauli_Ham/ and June_main/).
_repo = Path.cwd().resolve()
while not (_repo / "Pauli_Ham").is_dir() and _repo != _repo.parent:
    _repo = _repo.parent
for _p in (str(_repo / "June_main"), str(_repo)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Read numbered-Pauli Hamiltonian from Pauli_Ham/<MOLECULE>_bond_<bond>.txt.
def load_pauli_sum_from_numbered_file(path: Path, qubits: list[cirq.Qid]) -> cirq.PauliSum:
    idx_to_pauli = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    out = cirq.PauliSum()

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
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
            for q, code in zip(qubits, pauli_codes):
                if code == 0:
                    continue
                if code not in idx_to_pauli:
                    raise ValueError(f"{path}:{lineno} has invalid Pauli code {code}; expected 0/1/2/3.")
                pauli_string *= idx_to_pauli[code](q)

            out += coeff * pauli_string

    return out


ham_path = _repo / "Pauli_Ham" / f"{MOLECULE}_bond_{bond_length:.1f}.txt"

# Hamiltonian + exact ground-state energy. The bare ansatz (no prep) is not a
# meaningful state by itself, so the reference energy of the FULL circuit is
# reported in the "Initial state preparation" cell once the prep is prepended.
pauli_sum = load_pauli_sum_from_numbered_file(ham_path, list(qubits))
qubit_map = {q: i for i, q in enumerate(qubits)}
e_gs = float(np.linalg.eigvalsh(pauli_sum.matrix(qubits=qubits))[0].real)

# HF determinant reference (X on the occupied spin-orbitals of this active
# space). Occupation comes from the explicit active-space inputs above
# (N_ACTIVE_ELECTRONS / N_SPATIAL_ORBITALS), not the circuit metadata.
_half = N_SPATIAL_ORBITALS
_eta = ETA
_occ = list(range(_eta)) + list(range(_half, _half + _eta))
psi_hf = cirq.Simulator().simulate(
    cirq.Circuit([cirq.X(qubits[k]) for k in _occ] or [cirq.I(q) for q in qubits]),
    qubit_order=qubits,
).final_state_vector
e_hf_ref = float(
    np.real(pauli_sum.expectation_from_state_vector(np.asarray(psi_hf, dtype=np.complex128), qubit_map=qubit_map))
)

print(
    f"\n{MOLECULE} bond {bond_length} Å  "
    f"(active space: {N_ACTIVE_ELECTRONS}e, {N_SPATIAL_ORBITALS}o -> {N_QUBITS} qubits)"
)
print(f"Hamiltonian source: {ham_path}")
print(f"⟨H⟩ HF determinant reference: {e_hf_ref:.10f} Eh")
print(f"Ground-state energy e_gs (exact): {e_gs:.10f} Eh")

# %% Notebook cell 2

# === Initial state preparation: HF (default) or multireference ======
# Prepend the reference state to the bare ansatz loaded above, then rebind
# ``circuit`` to the full (prep + ansatz) circuit used by every later cell.
#   "hf"       -> Hartree-Fock determinant (X on the occupied spin-orbitals)
#   "multiref" -> paper Eq.(6) state (|HF> - beta|exc>)/sqrt(1+beta^2). The prep
#                 rotation Ry(alpha = -2*arctan(beta)) is optimized HERE by
#                 minimizing the noiseless <H> (no separate set-then-optimize).
INIT_STATE_METHOD = "hf"  # "hf" or "multiref"

from scipy.optimize import minimize_scalar

_half = N_SPATIAL_ORBITALS
_eta = ETA
_occupied = list(range(_eta)) + list(range(_half, _half + _eta))
_sim_prep = cirq.Simulator()


def hf_prep_circuit() -> cirq.Circuit:
    """HF determinant: X on each occupied spin-orbital."""
    return cirq.Circuit([cirq.X(qubits[k]) for k in _occupied])


def multiref_prep_circuit(alpha: float) -> cirq.Circuit:
    """Paper Eqs.(7)-(8): Ry(alpha) on q_{eta-1}, a 3-CNOT chain, then the HF X layer."""
    c = cirq.Circuit()
    c.append(cirq.ry(float(alpha)).on(qubits[_eta - 1]))
    c.append(cirq.CNOT(qubits[_eta - 1], qubits[_eta]))
    c.append(cirq.CNOT(qubits[_eta], qubits[_half + _eta - 1]))
    c.append(cirq.CNOT(qubits[_half + _eta - 1], qubits[_half + _eta]))
    c.append([cirq.X(qubits[k]) for k in _occupied])
    return c


def _prep_state_energy(prep: cirq.Circuit) -> float:
    psi = np.asarray(
        _sim_prep.simulate(prep, qubit_order=qubits).final_state_vector, dtype=np.complex128
    )
    return float(np.real(pauli_sum.expectation_from_state_vector(psi, qubit_map=qubit_map)))


if INIT_STATE_METHOD == "hf":
    prep_circuit = hf_prep_circuit()
    multiref_alpha = None
    multiref_beta = None
    print("Initial state: Hartree-Fock determinant")
    print(f"  occupied spin-orbitals : {_occupied}")
elif INIT_STATE_METHOD == "multiref":
    # Combined choice + optimization: minimize <H> over the prep rotation alpha.
    _opt = minimize_scalar(
        lambda a: _prep_state_energy(multiref_prep_circuit(a)),
        bounds=(-np.pi, np.pi),
        method="bounded",
    )
    multiref_alpha = float(_opt.x)
    multiref_beta = float(-np.tan(multiref_alpha / 2.0))
    prep_circuit = multiref_prep_circuit(multiref_alpha)
    print("Initial state: multireference (paper Eq.6), beta optimized in this cell")
    print(f"  optimal alpha = {multiref_alpha:.6f} rad  ->  beta = {multiref_beta:.6f}")
else:
    raise ValueError(f"INIT_STATE_METHOD must be 'hf' or 'multiref', got {INIT_STATE_METHOD!r}")

# Full circuit = prep + bare ansatz. Rebuilt from ``ansatz_circuit`` every run so
# re-executing this cell is idempotent; rebind ``circuit`` for downstream cells.
circuit = prep_circuit + ansatz_circuit
n_params = len(symbols)  # prep adds no free parameters

# Reference energy of the full circuit at theta = 0 (= the chosen prep state).
_res0 = cirq.resolve_parameters(circuit, cirq.ParamResolver({s: 0.0 for s in symbols}))
_psi0 = np.asarray(
    cirq.Simulator().simulate(_res0, qubit_order=qubits).final_state_vector, dtype=np.complex128
)
e_prep_ref = float(np.real(pauli_sum.expectation_from_state_vector(_psi0, qubit_map=qubit_map)))

print(f"  ⟨H⟩ prep state (all θ=0)  : {e_prep_ref:.10f} Eh")
print(f"  ⟨H⟩ HF determinant ref    : {e_hf_ref:.10f} Eh")
print(f"  ground-state e_gs (exact) : {e_gs:.10f} Eh")
print(
    f"Full circuit ready: {len(list(circuit.all_operations()))} ops, "
    f"{n_params} params (downstream cells use `circuit`)."
)

# Cloud export uses a molecule-agnostic zero-vector probe before VQE.
params = np.zeros(n_params, dtype=float)
resolver_test = cirq.ParamResolver(resolver_from_params(params))
resolved_test = cirq.resolve_parameters(circuit, resolver_test)
psi_test = np.asarray(
    cirq.Simulator(dtype=np.complex128).simulate(resolved_test, qubit_order=qubits).final_state_vector,
    dtype=np.complex128,
)
e_test = float(np.real(pauli_sum.expectation_from_state_vector(psi_test, qubit_map=qubit_map)))
print(f"params = {params.tolist()}")
print(f"E(params) = {e_test:.10f} Eh")

# %% Notebook cell 4

# No measurement error here (density matrix + Tr[H ρ]).
# just noisy energy using trace of H * rho 
import sys
from pathlib import Path

_repo = Path.cwd().resolve()
while not (_repo / "June_main" / "main_cursor_lib.py").is_file() and _repo != _repo.parent:
    _repo = _repo.parent
for _p in (str(_repo / "June_main"), str(_repo)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from main_cursor_lib import (
    GateArityDepolarizingNoise,
    CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    ONE_QUBIT_GATE_DEPOL_PROB,
    TWO_QUBIT_GATE_DEPOL_PROB,
    trace_energy,
)

random_seed = int(globals()["GLOBAL_RANDOM_SEED"])

gate_noise = GateArityDepolarizingNoise(
    two_qubit_depol_prob=TWO_QUBIT_GATE_DEPOL_PROB,
    one_qubit_depol_prob=ONE_QUBIT_GATE_DEPOL_PROB,
    cross_chip_two_qubit_depol_prob=CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
)
noisy_ansatz = circuit.with_noise(gate_noise)
resolver_noisy = cirq.ParamResolver(resolver_from_params(params))
resolved_noisy = cirq.resolve_parameters(noisy_ansatz, resolver_noisy)

rho_noisy = np.asarray(
    cirq.DensityMatrixSimulator(seed=random_seed)
    .simulate(resolved_noisy, qubit_order=qubits)
    .final_density_matrix,
    dtype=np.complex128,
)

hamiltonian_matrix = pauli_sum.matrix(qubits=qubits)
trace_noisy_energy = trace_energy(hamiltonian_matrix, rho_noisy)

print(
    f"noise: two_qubit_depol={gate_noise.two_qubit_depol_prob} "
    f"one_qubit_depol={gate_noise.one_qubit_depol_prob} "
    f"cross_chip_two_qubit_depol={gate_noise.cross_chip_two_qubit_depol_prob}"
)
print(f"params = {params.tolist()}")
print(f"Tr[H ρ_noisy] (gate noise only): {trace_noisy_energy:.10f} Eh")

# %% Notebook cell 5

# Finite-shot energy from the same ``rho_noisy`` as above: OGM measurement layout + asymmetric
# readout (``p_0_success`` / ``p_1_success`` on LineQubit 0…5) and optional REM in post-processing.
# the OGM basis file is read from the local OGM_measurement_basis folder, not from shadowgrouping.
import sys
from pathlib import Path

_repo = Path.cwd().resolve()
while not (_repo / "June_main" / "main_cursor_lib.py").is_file() and _repo != _repo.parent:
    _repo = _repo.parent
for _p in (str(_repo / "June_main"), str(_repo)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shot_measurement import estimate_energy_from_noisy_rho_shots

num_shots = int(globals()["GLOBAL_NUM_SHOTS"])
measurement_scheme = str(globals()["GLOBAL_MEASUREMENT_SCHEME"])
sampling_seed = int(globals()["GLOBAL_SAMPLING_SEED"])
epsilon = 0.1

p_0_success = np.array(globals()["GLOBAL_READOUT_P0_SUCCESS"], dtype=float)
p_1_success = np.array(globals()["GLOBAL_READOUT_P1_SUCCESS"], dtype=float)

apply_readout_noise = bool(globals()["GLOBAL_APPLY_READOUT_NOISE"])
apply_rem = True

ogm_file = globals().get(
    "ogm_file",
    _repo / "June_main" / "OGM_measurement_basis" / f"OGM_{MOLECULE}_bond_{bond_length:.1f}.txt",
)

print(f"OGM file: {ogm_file}  exists={ogm_file.is_file()}")

if not ogm_file.is_file():
    print(
        "Skip OGM shot estimate: OGM basis file missing in June_main/OGM_measurement_basis."
    )
else:
    try:
        shot_est = estimate_energy_from_noisy_rho_shots(
            rho_noisy,
            pauli_sum,
            qubits,
            num_shots=num_shots,
            measurement_scheme=measurement_scheme,
            p_0_success=p_0_success,
            p_1_success=p_1_success,
            apply_rem=apply_rem,
            apply_readout_noise=apply_readout_noise,
            sampling_seed=sampling_seed,
            epsilon=epsilon,
            ogm_file=ogm_file,
        )
        eu = float(shot_est["energy_unmitigated"])
        er = float(shot_est["energy_rem"])

        print(f"Finite-shot energy (readout noise, no REM correction): {eu:.12f} Eh")
        print(f"Finite-shot energy (REM readout mitigation):          {er:.12f} Eh")
        print(f"REM delta (REM - raw shots):                            {er - eu:.12f} Eh")
        print(
            f"\nReference Tr[H ρ] (same ρ, exact Pauli from DM; no shot noise): {trace_noisy_energy:.12f} Eh"
        )
    except Exception:
        raise

# %% Notebook cell 7

# default per-Pauli CDR to mitigate error from measurement and gate noise
import sys
from pathlib import Path

import cirq
import numpy as np

_repo = Path.cwd().resolve()
while not (_repo / "June_main" / "main_cursor_lib.py").is_file() and _repo != _repo.parent:
    _repo = _repo.parent
for _p in (str(_repo / "June_main"), str(_repo)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from main_cursor_lib import (
    CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    ONE_QUBIT_GATE_DEPOL_PROB,
    TWO_QUBIT_GATE_DEPOL_PROB,
    run_cdr_with_per_pauli_coeff_print,
)
from shot_measurement import run_mitigation

symbols_li_h = symbols
target_resolver = resolver_from_params(params)

base_noise_cfg = {
    "two_qubit_depol_prob": TWO_QUBIT_GATE_DEPOL_PROB,
    "one_qubit_depol_prob": ONE_QUBIT_GATE_DEPOL_PROB,
    "cross_chip_two_qubit_depol_prob": CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
}

shot_cfg = {
    "num_shots": int(globals().get("num_shots", globals()["GLOBAL_NUM_SHOTS"])),
    "measurement_scheme": str(globals().get("measurement_scheme", globals()["GLOBAL_MEASUREMENT_SCHEME"])),
    "apply_readout_noise": bool(globals().get("apply_readout_noise", globals()["GLOBAL_APPLY_READOUT_NOISE"])),
    "sampling_seed": int(globals().get("sampling_seed", globals()["GLOBAL_SAMPLING_SEED"])),
    "ogm_file": globals().get(
        "ogm_file",
        _repo / "June_main" / "OGM_measurement_basis" / f"OGM_{MOLECULE}_bond_{bond_length:.1f}.txt",
    ),
}
readout_cal = {
    "p_0_success": np.array(globals().get("p_0_success", globals()["GLOBAL_READOUT_P0_SUCCESS"]), dtype=float),
    "p_1_success": np.array(globals().get("p_1_success", globals()["GLOBAL_READOUT_P1_SUCCESS"]), dtype=float),
}

cdr_cfg_base = dict(
    num_circuits=int(globals().get("CDR_NUM_TRAINING_CIRCUITS", globals()["CDR_NUM_TRAINING_CIRCUITS"])),
    t_max=int(globals().get("CDR_T_MAX_VQE", globals()["CDR_T_MAX_VQE"])),
    seed=int(globals().get("CDR_BASE_SEED", globals()["CDR_BASE_SEED"])),
)

if not Path(shot_cfg["ogm_file"]).is_file():
    print("Skip CDR demo: OGM basis file missing in June_main/OGM_measurement_basis.")
else:
    mit_pp = run_cdr_with_per_pauli_coeff_print(
        ansatz_circuit=circuit,
        observable_h=pauli_sum,
        qubits=qubits,
        target_resolver=target_resolver,
        target_params=target_resolver,
        symbols=symbols_li_h,
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
        readout_cal=readout_cal,
        cdr_cfg={**cdr_cfg_base},
        simulator_seed=int(globals().get("random_seed", globals()["GLOBAL_RANDOM_SEED"])),
    )
    if "e_test" in globals():
        eref = float(e_test)
    else:
        resolved_target = cirq.resolve_parameters(circuit, cirq.ParamResolver(target_resolver))
        psi_target = cirq.Simulator(
            seed=int(globals().get("random_seed", globals()["GLOBAL_RANDOM_SEED"]))
        ).simulate(resolved_target, qubit_order=qubits).final_state_vector
        h_mat = pauli_sum.matrix(qubits=qubits)
        eref = float(np.vdot(psi_target, h_mat @ psi_target).real)

    print("CDR (per-pauli)")
    print(
        f"raw finite-shot (unmit / REM): {float(mit_pp['unmit_target']):.12f} / {float(mit_pp['rem_target']):.12f} Eh"
    )
    print(
        "cdr corrected (unmit / REM): "
        f"{float(mit_pp['cdr_unmit_corrected']):.12f} / {float(mit_pp['cdr_rem_corrected']):.12f} Eh"
    )
    print(f"reference exact noiseless: {eref:.12f} Eh")
    print(f"Energy error: {eref - float(mit_pp['cdr_rem_corrected']):.12f} Eh")

# %% Notebook cell 12

# Gradients for the mitigated objective: CDR-corrected REM energy by default.
import numpy as np
import cirq

from main_cursor_lib import (
    CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    ONE_QUBIT_GATE_DEPOL_PROB,
    TWO_QUBIT_GATE_DEPOL_PROB,
)

_grad_sim = cirq.Simulator(
    seed=int(globals().get("random_seed", globals()["GLOBAL_RANDOM_SEED"])),
    dtype=np.complex128,
)
_PARAM_SHIFT = np.pi / 2.0
_DEFAULT_ENERGY_MODE = "cdr_rem_corrected"

# Independent shot-noise control. When True, every shot-energy evaluation draws a
# fresh sampling seed, so the noisy gradient sees genuine, uncorrelated shot noise
# (no common-random-number cancellation between the +/- parameter-shift terms).
# Set to False to recover the old deterministic-seed behaviour.
_VQE_INDEPENDENT_SAMPLING = True
_VQE_SEED_RNG = np.random.default_rng(int(globals().get("GLOBAL_SAMPLING_SEED", 1234)))


def _next_sampling_seed() -> int:
    if not _VQE_INDEPENDENT_SAMPLING:
        return int(globals().get("sampling_seed", globals()["GLOBAL_SAMPLING_SEED"]))
    return int(_VQE_SEED_RNG.integers(1, 2**31 - 1))


def _resolver_from_params(params_vec: np.ndarray) -> dict:
    return resolver_from_params(params_vec)


def _noiseless_energy(params_vec: np.ndarray) -> float:
    """Exact noiseless statevector expectation ⟨H⟩."""
    resolver = cirq.ParamResolver(_resolver_from_params(params_vec))
    resolved = cirq.resolve_parameters(circuit, resolver)
    psi = np.asarray(
        _grad_sim.simulate(resolved, qubit_order=qubits).final_state_vector,
        dtype=np.complex128,
    )
    return float(np.real(pauli_sum.expectation_from_state_vector(psi, qubit_map=qubit_map)))


def _cdr_rem_corrected_energy(params_vec: np.ndarray) -> float:
    """CDR-corrected REM target energy from ``run_mitigation(mode='cdr')``."""
    if "run_mitigation" not in globals():
        raise RuntimeError("run_mitigation not found. Run the CDR setup cell first.")

    resolver = _resolver_from_params(params_vec)
    symbols = globals().get("symbols_li_h", globals()["symbols"])

    base_noise = dict(
        globals().get(
            "base_noise_cfg",
            {
                "two_qubit_depol_prob": float(TWO_QUBIT_GATE_DEPOL_PROB),
                "one_qubit_depol_prob": float(ONE_QUBIT_GATE_DEPOL_PROB),
                "cross_chip_two_qubit_depol_prob": float(CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB),
            },
        )
    )
    shot_cfg_local = dict(globals().get("shot_cfg", {}))
    shot_cfg_local.setdefault("num_shots", int(globals().get("num_shots", globals()["GLOBAL_NUM_SHOTS"])))
    shot_cfg_local.setdefault("measurement_scheme", str(globals()["GLOBAL_MEASUREMENT_SCHEME"]))
    shot_cfg_local.setdefault("apply_readout_noise", bool(globals()["GLOBAL_APPLY_READOUT_NOISE"]))
    shot_cfg_local["sampling_seed"] = _next_sampling_seed()

    readout_cal_local = dict(globals().get("readout_cal", {}))
    readout_cal_local.setdefault("p_0_success", np.array(globals()["GLOBAL_READOUT_P0_SUCCESS"], dtype=float))
    readout_cal_local.setdefault("p_1_success", np.array(globals()["GLOBAL_READOUT_P1_SUCCESS"], dtype=float))

    cdr_cfg_local = dict(
        globals().get(
            "cdr_cfg_base",
            {
                "num_circuits": int(globals()["CDR_NUM_TRAINING_CIRCUITS"]),
                "t_max": int(globals()["CDR_T_MAX_GRADIENT"]),
                "seed": int(globals()["CDR_BASE_SEED"]),
            },
        )
    )

    mit = run_mitigation(
        "cdr",
        ansatz_circuit=circuit,
        observable_h=pauli_sum,
        qubits=qubits,
        target_resolver=resolver,
        target_params=resolver,
        symbols=symbols,
        base_noise_cfg=base_noise,
        shot_cfg=shot_cfg_local,
        readout_cal=readout_cal_local,
        cdr_cfg=cdr_cfg_local,
        simulator_seed=int(globals().get("random_seed", globals()["GLOBAL_RANDOM_SEED"])),
    )
    return float(mit["cdr_rem_corrected"])


def energy_from_params(params_vec: np.ndarray, energy_mode: str = _DEFAULT_ENERGY_MODE) -> float:
    """Objective energy at params.

    Supported modes:
      - ``cdr_rem_corrected`` (default): mitigated target from CDR + REM
      - ``noiseless``: exact statevector expectation
    """
    mode = str(energy_mode).strip().lower()
    if mode in {"cdr", "cdr_rem", "cdr_rem_corrected"}:
        return _cdr_rem_corrected_energy(params_vec)
    if mode in {"noiseless", "exact", "statevector"}:
        return _noiseless_energy(params_vec)
    raise ValueError(f"Unsupported energy_mode={energy_mode!r}.")


def parameter_shift_gradient(params_vec: np.ndarray, energy_mode: str = _DEFAULT_ENERGY_MODE) -> np.ndarray:
    """Gradient estimator via ±π/2 shifts (exact only for pure expectation objectives)."""
    g = np.zeros(n_params, dtype=float)
    p = np.asarray(params_vec, dtype=float).reshape(n_params)
    for i in range(n_params):
        plus = p.copy()
        minus = p.copy()
        plus[i] += _PARAM_SHIFT
        minus[i] -= _PARAM_SHIFT
        g[i] = 0.5 * (
            energy_from_params(plus, energy_mode=energy_mode)
            - energy_from_params(minus, energy_mode=energy_mode)
        )
    return g


def finite_difference_gradient(
    params_vec: np.ndarray,
    eps: float = 2e-2,
    energy_mode: str = _DEFAULT_ENERGY_MODE,
) -> np.ndarray:
    """Central finite-difference gradient for the selected objective."""
    g = np.zeros(n_params, dtype=float)
    p = np.asarray(params_vec, dtype=float).reshape(n_params)
    for i in range(n_params):
        plus = p.copy()
        minus = p.copy()
        plus[i] += eps
        minus[i] -= eps
        g[i] = (
            energy_from_params(plus, energy_mode=energy_mode)
            - energy_from_params(minus, energy_mode=energy_mode)
        ) / (2.0 * eps)
    return g


# %% Notebook cell 15

# --- VQE configuration (CDR+REM + parameter-shift + fixed LR) ---

# Optimisation (fixed length)
VQE_ITERS = int(os.environ.get("VQE_ITERS", "15"))

# Fixed learning rate (gradient descent on parameter-shift gradients)
VQE_LR = 0.5
NOISELESS_REF_LR = VQE_LR

# Require upstream CDR globals (run the CDR setup cell first)
_req = ["run_mitigation", "base_noise_cfg", "shot_cfg", "readout_cal", "symbols", "circuit", "pauli_sum", "qubits", "resolver_from_params"]
_miss = [n for n in _req if n not in globals()]
if _miss:
    raise RuntimeError("Missing notebook globals: " + ", ".join(_miss))


# %% Notebook cell 16

# --- VQE loop: fixed-LR gradient descent + full 3-parameter updates ---
import numpy as np

from main_cursor_lib import (
    CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    ONE_QUBIT_GATE_DEPOL_PROB,
    TWO_QUBIT_GATE_DEPOL_PROB,
)

_ENERGY_MODE = "cdr_rem_corrected"

# Count every outer ``cdr_rem_corrected`` objective evaluation (``energy_from_params`` / ``run_mitigation``).
_n_energy_total = 0
_n_init = 0
_n_warmup = 0
_n_main_grad = 0
_n_main_post = 0
_n_final = 0
_n_noiseless_track = 0


def _vqe_energy(p, bucket: str) -> float:
    global _n_energy_total
    _n_energy_total += 1
    if bucket == "init":
        global _n_init
        _n_init += 1
    elif bucket == "warmup":
        global _n_warmup
        _n_warmup += 1
    elif bucket == "main_grad":
        global _n_main_grad
        _n_main_grad += 1
    elif bucket == "main_post":
        global _n_main_post
        _n_main_post += 1
    elif bucket == "final":
        global _n_final
        _n_final += 1
    else:
        raise ValueError(bucket)
    return float(energy_from_params(np.asarray(p, dtype=float).reshape(n_params), energy_mode=_ENERGY_MODE))


def _vqe_run_mitigation_triple(p, bucket: str) -> tuple[float, float, float]:
    """One ``run_mitigation("cdr", ...)`` call -> (raw finite-shot, REM, REM+CF)."""
    global _n_energy_total
    _n_energy_total += 1
    if bucket == "init":
        global _n_init
        _n_init += 1
    elif bucket == "warmup":
        global _n_warmup
        _n_warmup += 1
    elif bucket == "main_post":
        global _n_main_post
        _n_main_post += 1
    elif bucket == "final":
        global _n_final
        _n_final += 1
    else:
        raise ValueError(bucket)

    pv = np.asarray(p, dtype=float).reshape(n_params)
    resolver = resolver_from_params(pv)
    symbols = globals().get("symbols_li_h", globals()["symbols"])
    base_noise = dict(
        globals().get(
            "base_noise_cfg",
            {
                "two_qubit_depol_prob": float(TWO_QUBIT_GATE_DEPOL_PROB),
                "one_qubit_depol_prob": float(ONE_QUBIT_GATE_DEPOL_PROB),
                "cross_chip_two_qubit_depol_prob": float(CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB),
            },
        )
    )
    shot_cfg_local = dict(globals().get("shot_cfg", {}))
    shot_cfg_local.setdefault("num_shots", int(globals().get("num_shots", globals()["GLOBAL_NUM_SHOTS"])))
    shot_cfg_local.setdefault("measurement_scheme", str(globals()["GLOBAL_MEASUREMENT_SCHEME"]))
    shot_cfg_local.setdefault("apply_readout_noise", bool(globals()["GLOBAL_APPLY_READOUT_NOISE"]))
    shot_cfg_local["sampling_seed"] = _next_sampling_seed()
    readout_cal_local = dict(globals().get("readout_cal", {}))
    readout_cal_local.setdefault("p_0_success", np.array(globals()["GLOBAL_READOUT_P0_SUCCESS"], dtype=float))
    readout_cal_local.setdefault("p_1_success", np.array(globals()["GLOBAL_READOUT_P1_SUCCESS"], dtype=float))
    cdr_cfg_local = dict(
        globals().get(
            "cdr_cfg_base",
            {
                "num_circuits": int(globals()["CDR_NUM_TRAINING_CIRCUITS"]),
                "t_max": int(globals()["CDR_T_MAX_VQE"]),
                "seed": int(globals()["CDR_BASE_SEED"]),
            },
        )
    )

    mit = run_mitigation(
        "cdr",
        ansatz_circuit=circuit,
        observable_h=pauli_sum,
        qubits=qubits,
        target_resolver=resolver,
        target_params=resolver,
        symbols=symbols,
        base_noise_cfg=base_noise,
        shot_cfg=shot_cfg_local,
        readout_cal=readout_cal_local,
        cdr_cfg=cdr_cfg_local,
        simulator_seed=int(globals().get("random_seed", globals()["GLOBAL_RANDOM_SEED"])),
    )
    return float(mit["unmit_target"]), float(mit["rem_target"]), float(mit["cdr_rem_corrected"])


def _vqe_noiseless(p) -> float:
    """Exact statevector ⟨H⟩ at the same parameters (not counted in ``_n_energy_total``)."""
    global _n_noiseless_track
    _n_noiseless_track += 1
    _pv = np.asarray(p, dtype=float).reshape(n_params)
    return float(energy_from_params(_pv, energy_mode="noiseless"))


def _vqe_param_shift_grad(p, bucket: str) -> np.ndarray:
    g = np.zeros(n_params, dtype=float)
    pv = np.asarray(p, dtype=float).reshape(n_params)
    h = float(_PARAM_SHIFT)
    for i in range(n_params):
        pp = pv.copy()
        pm = pv.copy()
        pp[i] += h
        pm[i] -= h
        g[i] = 0.5 * (_vqe_energy(pp, bucket) - _vqe_energy(pm, bucket))
    return g


# Initial guess (all-zero parameters = the reference state)
_params0 = np.zeros(n_params, dtype=float)
print(f"[VQE] initial params ({n_params} angles) = {_params0.tolist()}")

P = n_params

# Real-machine cost model for one CDR objective call.
# In this notebook's CDR path (per-pauli fit), one run_mitigation('cdr') evaluates shot energies
# approximately (num_circuits + 2) times:
#   - num_circuits training circuits, plus
#   - 2 target baseline evaluations (one early baseline + one per-term baseline).
_shot_cfg_local = dict(globals().get("shot_cfg", {}))
_cdr_cfg_local = dict(globals().get("cdr_cfg_base", {}))
_num_shots = int(_shot_cfg_local.get("num_shots", globals()["num_shots"]))
_num_circuits = int(_cdr_cfg_local.get("num_circuits", globals()["CDR_NUM_TRAINING_CIRCUITS"]))
_energy_evals_per_cdr_call = int(_num_circuits + 1)
_shots_per_cdr_call = int(_energy_evals_per_cdr_call * _num_shots)

print(
    f"[VQE] Cost model per CDR objective call: energy-evals={_energy_evals_per_cdr_call} "
    f"(num_circuits={_num_circuits} + 1), shots/eval={_num_shots}, "
    f"shots/call={_shots_per_cdr_call}"
)

# Baseline (iteration 0): one mitigation triple + exact noiseless ⟨H⟩ at θ₀
energy_curves: list[dict] = []
raw0, rem0, cdr0 = _vqe_run_mitigation_triple(_params0, "init")
E0 = float(cdr0)
nls0 = float(_vqe_noiseless(_params0))
energy_curves.append(
    {
        "iter": 0,
        "raw_eh": float(raw0),
        "rem_eh": float(rem0),
        "rem_cf_eh": float(cdr0),
        "noiseless_eh": float(nls0),
    }
)

VQE_LR = float(VQE_LR)
_NOISELESS_REF_LR = float(NOISELESS_REF_LR)
print(f"[VQE] fixed VQE_LR={VQE_LR:g} (CDR+REM main loop)")
print(f"[VQE] noiseless reference LR={_NOISELESS_REF_LR:g} (decoupled exact reference curve)")

# Decoupled exact reference: noiseless-only fixed-LR + parameter-shift.


def _build_noiseless_reference_curve(params_start: np.ndarray, iters: int):
    th = np.asarray(params_start, dtype=float).reshape(n_params).copy()
    out = [float(_vqe_noiseless(th))]
    thetas = [th.copy()]
    for _ in range(int(iters)):
        g_ref = parameter_shift_gradient(th, energy_mode="noiseless")
        th = th - float(_NOISELESS_REF_LR) * g_ref
        out.append(float(_vqe_noiseless(th)))
        thetas.append(th.copy())
    return out, thetas


noiseless_ref_curve, noiseless_ref_thetas = _build_noiseless_reference_curve(_params0, int(VQE_ITERS))

# --- Main optimisation (fixed exactly VQE_ITERS iterations) ---
theta = _params0.copy()

trace = []
best_E = float(E0)
best_theta = _params0.copy()
prev_E = float(E0)

for it in range(1, int(VQE_ITERS) + 1):
    _count_before = int(_n_energy_total)
    theta_before = theta.copy()
    active = np.arange(P, dtype=int)
    g = _vqe_param_shift_grad(theta, "main_grad")
    theta = theta - float(VQE_LR) * g
    raw_e, rem_e, E = _vqe_run_mitigation_triple(theta, "main_post")
    nls_e = float(noiseless_ref_curve[int(it)])
    dE = float(E) - float(prev_E)

    # --- Diagnostics: isolate the effect of noise on the gradient vs the params ---
    # g was computed on the noisy (CDR+REM) objective at theta_before; compare it to
    # the *exact* parameter-shift gradient at the SAME point. A tiny ||g_noisy - g_noiseless||
    # alongside a ~1e-2 absolute REM+CF energy bias confirms the bias cancels in the
    # +/- shift difference. theta_diff compares the two (decoupled) GD trajectories.
    g_noiseless_same = parameter_shift_gradient(theta_before, energy_mode="noiseless")
    grad_diff = float(np.linalg.norm(g - g_noiseless_same))
    grad_noisy_norm = float(np.linalg.norm(g))
    theta_diff = float(np.linalg.norm(theta - noiseless_ref_thetas[int(it)]))
    nls_at_theta = float(_vqe_noiseless(theta))
    e_bias = abs(float(E) - nls_at_theta)

    energy_curves.append(
        {
            "iter": int(it),
            "raw_eh": float(raw_e),
            "rem_eh": float(rem_e),
            "rem_cf_eh": float(E),
            "noiseless_eh": float(nls_e),
        }
    )

    trace.append(
        {
            "iter": int(it),
            "energy": float(E),
            "raw_eh": float(raw_e),
            "rem_eh": float(rem_e),
            "noiseless_eh": float(nls_e),
            "theta": theta.copy(),
            "grad": g.copy(),
            "active": active.copy(),
            "dE": float(dE),
            "grad_diff": grad_diff,
            "grad_noisy_norm": grad_noisy_norm,
            "theta_diff": theta_diff,
            "e_bias": e_bias,
        }
    )

    if E < best_E:
        best_E = float(E)
        best_theta = theta.copy()

    step_vec = theta_before - theta
    step_l2 = float(np.linalg.norm(step_vec)) if step_vec.size else 0.0
    step_max = float(np.max(np.abs(step_vec))) if step_vec.size else 0.0

    cdr_calls_this_iter = int(_n_energy_total - _count_before)
    cum_cdr_calls = int(_n_energy_total)
    cum_energy_evals = int(cum_cdr_calls * _energy_evals_per_cdr_call)
    cum_shots = int(cum_cdr_calls * _shots_per_cdr_call)
    grad_str = np.array2string(g, precision=6, separator=", ")
    theta_updated = np.array2string(theta, precision=6, separator=", ")

    print(
        f"[VQE] iter={it:02d}  lr={float(VQE_LR):.5g}  E={E:.8f}  dE={dE:+.3e}  "
        f"grad={grad_str}  step_max={step_max:.3e}  step_l2={step_l2:.3e}  active={active.tolist()}  "
        f"theta={theta_updated}  cdr_calls_iter={cdr_calls_this_iter}  "
        f"cdr_calls_cum={cum_cdr_calls}  energy_evals_cum≈{cum_energy_evals}  shots_cum≈{cum_shots}"
    )
    print(
        f"[VQE-diag] iter={it:02d}  "
        f"||g_noisy - g_noiseless||={grad_diff:.3e}  "
        f"||g_noisy||={grad_noisy_norm:.3e}  "
        f"||theta_noisy - theta_noiseless||={theta_diff:.3e}  "
        f"|REM+CF - noiseless|@theta={e_bias:.3e}"
    )
    prev_E = float(E)

_, _, E_final = _vqe_run_mitigation_triple(theta, "final")

_total_cdr_calls = int(_n_energy_total)
_total_energy_evals = int(_total_cdr_calls * _energy_evals_per_cdr_call)
_total_shots = int(_total_cdr_calls * _shots_per_cdr_call)
print(
    f"[VQE] TOTAL measurement cost: cdr_calls={_total_cdr_calls}, "
    f"energy_evals≈{_total_energy_evals}, total_shots≈{_total_shots}"
)

vqe_results = {
    "params_init": _params0.copy(),
    "params_final": theta.copy(),
    "params_best": best_theta.copy(),
    "E_init": float(E0),
    "E_final": float(E_final),
    "E_best": float(best_E),
    "trace": trace,
    "energy_curves": energy_curves,
    "lr": float(VQE_LR),
    "noiseless_ref_lr": float(_NOISELESS_REF_LR),
    "max_iters": int(VQE_ITERS),
    "executed_iters": int(trace[-1]["iter"]) if trace else 0,
    "counts": {
        "total": int(_n_energy_total),
        "init": int(_n_init),
        "warmup": int(_n_warmup),
        "main_grad": int(_n_main_grad),
        "main_post": int(_n_main_post),
        "final": int(_n_final),
        "noiseless_track": int(_n_noiseless_track),
    },
    "cost_model": {
        "num_circuits": int(_num_circuits),
        "num_shots_per_eval": int(_num_shots),
        "energy_evals_per_cdr_call": int(_energy_evals_per_cdr_call),
        "shots_per_cdr_call": int(_shots_per_cdr_call),
        "total_cdr_calls": int(_total_cdr_calls),
        "total_energy_evals_est": int(_total_energy_evals),
        "total_shots_est": int(_total_shots),
    },
}



# Save VQE results immediately so a long cloud job still leaves data if CME is skipped/fails.
_saved_vqe_paths = save_checkpoint(
    data_dir=Path("data"),
    molecule=MOLECULE,
    bond_length=float(bond_length),
    stage="vqe",
    vqe_results=vqe_results,
    metadata={
        "circuit_name": CIRCUIT_NAME,
        "measurement_scheme": GLOBAL_MEASUREMENT_SCHEME,
        "num_shots": int(GLOBAL_NUM_SHOTS),
        "cdr_training_circuits": int(CDR_NUM_TRAINING_CIRCUITS),
        "vqe_iters": int(VQE_ITERS),
    },
)


# %% Notebook cell 19 (CME/CMX)

cme_results = None
cme_results_by_multiplier = None

def _run_cmx():
    global cme_results, cme_results_by_multiplier

    try:

        # --- Connected Moment Expansion (CME), order k=3 ---
        # E_ground = <H> - (<H^2> - <H>^2)^2 / (<H^3> - 3<H><H^2> + 2<H>^3)
        #
        # Quantum state: the BEST noisy state found during VQE (gate noise + the
        # parameters with the lowest measured CDR+REM energy). <H>, <H^2> and <H^3> are each measured with the
        # SAME CDR+REM pipeline used for the VQE energy objective (run_mitigation("cdr"))
        # -- one call per observable, each with its own Hamiltonian file, OGM basis file
        # and shot budget. CDR rebuilds the noisy state internally from `circuit` + the
        # best-parameter resolver + the same noise model and simulator seed, i.e. the
        # identical selected noisy state as `rho_cmx` (kept here only for the exact
        # Tr[H^k rho] reference).
        #
        # Accuracy is judged against the EXACT NOISELESS expectation <psi|H^k|psi> at the
        # best params (mirrors the VQE loop's |REM+CF - noiseless| diagnostic), because
        # CDR+REM recovers the noiseless value -- not the noisy-state Tr[H^k rho_cmx].
        import sys
        from pathlib import Path

        import cirq
        import numpy as np

        _repo = Path.cwd().resolve()
        while not (_repo / "June_main" / "main_cursor_lib.py").is_file() and _repo != _repo.parent:
            _repo = _repo.parent
        for _p in (str(_repo / "June_main"), str(_repo)):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from main_cursor_lib import (
            GateArityDepolarizingNoise,
            CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
            ONE_QUBIT_GATE_DEPOL_PROB,
            TWO_QUBIT_GATE_DEPOL_PROB,
            trace_energy,
        )
        from shot_measurement import run_mitigation, sanitize_density_matrix

        # ----------------------------------------------------------------------------
        # Per-observable shot budgets. <H^2> and <H^3> are much harder to estimate than
        # <H> (larger coefficients + many more Pauli terms), so they get their OWN, much
        # bigger shot counts here instead of the single GLOBAL_NUM_SHOTS knob (cell :18).
        # Tune these freely.
        # ----------------------------------------------------------------------------
        CME_H_NUM_SHOTS = int(globals().get("GLOBAL_NUM_SHOTS", 8192))   # shots for <H>
        # <H^2> and <H^3> are swept over several multiples of CME_H_NUM_SHOTS (instead of
        # a single fixed shot count) so we can see how their shot budget affects the
        # CME(k=3) accuracy. Edit this list freely.
        CME_SHOT_MULTIPLIERS = [1, 5, 10, 15]
        CME_VARIANCE_REPEATS = int(os.environ.get("CME_VARIANCE_REPEATS", "10"))
        if CME_VARIANCE_REPEATS < 2:
            raise ValueError("CME_VARIANCE_REPEATS must be at least 2 to estimate a variance.")

        # Which measured value feeds the CME formula, one of:
        #   "cdr_rem"   -> CDR + REM corrected (matches the VQE energy objective)
        #   "cdr_unmit" -> CDR corrected on the raw (non-REM) finite-shot energy
        #   "rem"       -> REM only (no CDR)
        #   "unmit"     -> raw finite-shot (no mitigation)
        CME_MOMENT_SOURCE = "cdr_rem"

        if "vqe_results" not in globals():
            raise RuntimeError("Run the VQE loop cell first (defines vqe_results).")

        # 1) Rebuild the noisy density matrix from the lowest-energy VQE parameters
        #    (same noisy-ansatz + density-matrix recipe as cell ``21b206da``).
        random_seed = int(globals().get("random_seed", globals()["GLOBAL_RANDOM_SEED"]))
        params_cmx = np.asarray(vqe_results["params_best"], dtype=float).reshape(n_params)
        print(f"[CMX] using VQE best params (E_best={float(vqe_results['E_best']):.10f} Eh)")

        gate_noise = GateArityDepolarizingNoise(
            two_qubit_depol_prob=TWO_QUBIT_GATE_DEPOL_PROB,
            one_qubit_depol_prob=ONE_QUBIT_GATE_DEPOL_PROB,
            cross_chip_two_qubit_depol_prob=CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
        )
        _noisy_ansatz = circuit.with_noise(gate_noise)
        _resolved_cmx = cirq.resolve_parameters(
            _noisy_ansatz, cirq.ParamResolver(resolver_from_params(params_cmx))
        )
        rho_cmx = np.asarray(
            cirq.DensityMatrixSimulator(seed=random_seed)
            .simulate(_resolved_cmx, qubit_order=qubits)
            .final_density_matrix,
            dtype=np.complex128,
        )
        rho_cmx = sanitize_density_matrix(rho_cmx)

        # 1b) Exact NOISELESS statevector at the SAME best params. This is the reference
        #     the VQE loop compares CDR+REM against (|REM+CF - noiseless|): CDR is built to
        #     recover the noiseless expectation, so each moment's cdr+rem should be compared
        #     to <psi|H^k|psi> here -- NOT to the noisy-state Tr[H^k rho_cmx] below.
        _psi_noiseless = np.asarray(
            cirq.Simulator(dtype=np.complex128)
            .simulate(
                cirq.resolve_parameters(circuit, cirq.ParamResolver(resolver_from_params(params_cmx))),
                qubit_order=qubits,
            )
            .final_state_vector,
            dtype=np.complex128,
        )
        _qubit_map = {q: i for i, q in enumerate(qubits)}

        # 2) Hamiltonian (H / H^2 / H^3) + matching OGM measurement-basis files.
        ham_paths = {
            "H": _repo / "Pauli_Ham" / f"{MOLECULE}_bond_{bond_length:.1f}.txt",
            "H2": _repo / "Pauli_Ham" / f"{MOLECULE}_square_bond_{bond_length:.1f}.txt",
            "H3": _repo / "Pauli_Ham" / f"{MOLECULE}_triple_bond_{bond_length:.1f}.txt",
        }
        ogm_paths = {
            "H": _repo / "June_main" / "OGM_measurement_basis" / f"OGM_{MOLECULE}_bond_{bond_length:.1f}.txt",
            "H2": _repo / "June_main" / "OGM_measurement_basis" / f"OGM_{MOLECULE}_square_bond_{bond_length:.1f}.txt",
            "H3": _repo / "June_main" / "OGM_measurement_basis" / f"OGM_{MOLECULE}_triple_bond_{bond_length:.1f}.txt",
        }
        for key in ("H", "H2", "H3"):
            if not ham_paths[key].is_file():
                raise FileNotFoundError(f"Missing Hamiltonian file for {key}: {ham_paths[key]}")
            if str(globals().get("GLOBAL_MEASUREMENT_SCHEME", "")).lower() != "direct_pauli" and not ogm_paths[key].is_file():
                raise FileNotFoundError(f"Missing OGM basis file for {key}: {ogm_paths[key]}")

        # Reuse the already-loaded H PauliSum; load H^2 and H^3 from their numbered files.
        pauli_sums = {
            "H": pauli_sum,
            "H2": load_pauli_sum_from_numbered_file(ham_paths["H2"], list(qubits)),
            "H3": load_pauli_sum_from_numbered_file(ham_paths["H3"], list(qubits)),
        }

        # 3) Mitigation config (CDR + REM), reused from the VQE objective so the moments
        #    are measured exactly like the 15th VQE energy. Falls back to GLOBAL_* knobs
        #    if the CDR-setup cell's globals are not present.
        cmx_resolver = cirq.ParamResolver(resolver_from_params(params_cmx))
        symbols_cme = globals().get("symbols_li_h", globals()["symbols"])

        base_noise_cme = dict(
            globals().get(
                "base_noise_cfg",
                {
                    "two_qubit_depol_prob": float(TWO_QUBIT_GATE_DEPOL_PROB),
                    "one_qubit_depol_prob": float(ONE_QUBIT_GATE_DEPOL_PROB),
                    "cross_chip_two_qubit_depol_prob": float(CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB),
                },
            )
        )
        readout_cal_cme = dict(globals().get("readout_cal", {}))
        readout_cal_cme.setdefault("p_0_success", np.array(globals()["GLOBAL_READOUT_P0_SUCCESS"], dtype=float))
        readout_cal_cme.setdefault("p_1_success", np.array(globals()["GLOBAL_READOUT_P1_SUCCESS"], dtype=float))
        cdr_cfg_cme = dict(
            globals().get(
                "cdr_cfg_base",
                {
                    "num_circuits": int(globals()["CDR_NUM_TRAINING_CIRCUITS"]),
                    "t_max": int(globals()["CDR_T_MAX_VQE"]),
                    "seed": int(globals()["CDR_BASE_SEED"]),
                },
            )
        )
        _base_shot_cfg = dict(globals().get("shot_cfg", {}))
        _base_shot_cfg.setdefault("measurement_scheme", str(globals()["GLOBAL_MEASUREMENT_SCHEME"]))
        _base_shot_cfg.setdefault("apply_readout_noise", bool(globals()["GLOBAL_APPLY_READOUT_NOISE"]))
        _base_shot_cfg.setdefault("sampling_seed", int(globals()["GLOBAL_SAMPLING_SEED"]))

        def _fresh_cmx_seed() -> int:
            """Return a fresh non-deterministic seed from OS entropy."""
            return int.from_bytes(os.urandom(8), byteorder="big") % (2**31 - 1) or 1

        def _measure_moment(key: str, num_shots: int) -> dict:
            """Measure <H^k> (key in {"H","H2","H3"}) with the CDR+REM pipeline at the
            given shot budget using independent sampling, CDR and simulator seeds."""
            sampling_seed_key = _fresh_cmx_seed()
            cdr_seed_key = _fresh_cmx_seed()
            simulator_seed_key = _fresh_cmx_seed()
            shot_cfg_key = dict(_base_shot_cfg)
            shot_cfg_key["num_shots"] = int(num_shots)
            shot_cfg_key["ogm_file"] = ogm_paths[key]
            shot_cfg_key["sampling_seed"] = sampling_seed_key
            cdr_cfg_key = dict(cdr_cfg_cme)
            cdr_cfg_key["seed"] = cdr_seed_key
            mit = run_mitigation(
                "cdr",
                ansatz_circuit=circuit,
                observable_h=pauli_sums[key],
                qubits=qubits,
                target_resolver=cmx_resolver,
                target_params=cmx_resolver,
                symbols=symbols_cme,
                base_noise_cfg=base_noise_cme,
                shot_cfg=shot_cfg_key,
                readout_cal=readout_cal_cme,
                cdr_cfg=cdr_cfg_key,
                simulator_seed=simulator_seed_key,
            )
            result = {
                "unmit": float(mit["unmit_target"]),
                "rem": float(mit["rem_target"]),
                "cdr_unmit": float(mit["cdr_unmit_corrected"]),
                "cdr_rem": float(mit["cdr_rem_corrected"]),
                "exact": float(trace_energy(pauli_sums[key].matrix(qubits=qubits), rho_cmx)),
                "noiseless": float(
                    np.real(pauli_sums[key].expectation_from_state_vector(_psi_noiseless, qubit_map=_qubit_map))
                ),
                "sampling_seed": int(sampling_seed_key),
                "cdr_seed": int(cdr_seed_key),
                "simulator_seed": int(simulator_seed_key),
            }
            print(
                f"<{key:<2}>  shots={int(num_shots):>8d}  "
                f"cdr+rem={result['cdr_rem']:+.8f}  rem={result['rem']:+.8f}  "
                f"unmit={result['unmit']:+.8f}  noiseless={result['noiseless']:+.8f}  "
                f"noisy_exact={result['exact']:+.8f}  "
                f"|cdr+rem - noiseless|={abs(result['cdr_rem'] - result['noiseless']):.3e}"
            )
            return result


        # 5) Connected moment expansion (k=3).
        def _cme_k3(h1: float, h2: float, h3: float):
            c1 = h1
            c2 = h2 - h1 ** 2
            c3 = h3 - 3.0 * h1 * h2 + 2.0 * h1 ** 3
            e = float("nan") if abs(c3) < 1e-12 else c1 - (c2 ** 2) / c3
            return e, c1, c2, c3


        def _cme_k3_gradient(a: float, b: float, c: float) -> np.ndarray:
            """Analytic gradient of E=a-(b-a^2)^2/(c-3ab+2a^3)."""
            s21 = b - a ** 2
            s31 = c - 3.0 * a * b + 2.0 * a ** 3
            if abs(s31) < 1e-12:
                return np.full(3, np.nan, dtype=float)
            return np.array(
                [
                    1.0 + 4.0 * a * s21 / s31
                    + s21 ** 2 * (-3.0 * b + 6.0 * a ** 2) / s31 ** 2,
                    -2.0 * s21 / s31 - 3.0 * a * s21 ** 2 / s31 ** 2,
                    s21 ** 2 / s31 ** 2,
                ],
                dtype=float,
            )


        def _validate_cme_k3_gradient_numerically() -> None:
            x = np.array([0.4, 1.2, 2.0], dtype=float)
            analytic = _cme_k3_gradient(*x)
            numeric = np.empty(3, dtype=float)
            for i in range(3):
                step = 1e-6 * max(1.0, abs(float(x[i])))
                xp, xm = x.copy(), x.copy()
                xp[i] += step
                xm[i] -= step
                numeric[i] = (_cme_k3(*xp)[0] - _cme_k3(*xm)[0]) / (2.0 * step)
            if not np.allclose(analytic, numeric, rtol=2e-6, atol=2e-7):
                raise RuntimeError(
                    f"CMX gradient check failed: analytic={analytic}, finite_difference={numeric}"
                )
            print(
                "[CMX] analytic derivatives passed the finite-difference check "
                f"(max |delta|={np.max(np.abs(analytic - numeric)):.3e})."
            )


        def _cme_k3_with_variance(a, b, c, var_a, var_b, var_c):
            energy, c1, c2, c3 = _cme_k3(a, b, c)
            gradient = _cme_k3_gradient(a, b, c)
            variance = max(
                float(np.sum(gradient ** 2 * np.array([var_a, var_b, var_c], dtype=float))),
                0.0,
            )
            return energy, c1, c2, c3, variance, float(np.sqrt(variance)), gradient


        _validate_cme_k3_gradient_numerically()
        e_ref = float(globals().get("e_gs", np.linalg.eigvalsh(pauli_sum.matrix(qubits=qubits))[0].real))

        # 6) Sweep the <H^2>/<H^3> shot budget over CME_SHOT_MULTIPLIERS x CME_H_NUM_SHOTS,
        #    independently remeasuring all moments and rerunning CME(k=3). <H>
        #    keeps its fixed base shot budget while <H^2>/<H^3> use the multiplier.
        cme_results_by_multiplier = {}
        for mult in CME_SHOT_MULTIPLIERS:
            h2_h3_shots = int(mult * CME_H_NUM_SHOTS)
            print(
                f"\n--- H shots/repeat={CME_H_NUM_SHOTS}, H2/H3 shots/repeat={h2_h3_shots}; "
                f"independent repeats={CME_VARIANCE_REPEATS} ---"
            )
            moment_replicates = {key: [] for key in ("H", "H2", "H3")}
            for rep in range(CME_VARIANCE_REPEATS):
                print(f"[CMX] variance repeat {rep + 1}/{CME_VARIANCE_REPEATS}")
                moment_replicates["H"].append(_measure_moment("H", CME_H_NUM_SHOTS))
                moment_replicates["H2"].append(_measure_moment("H2", h2_h3_shots))
                moment_replicates["H3"].append(_measure_moment("H3", h2_h3_shots))
            moment_arrays = {
                key: np.asarray(
                    [result[CME_MOMENT_SOURCE] for result in moment_replicates[key]], dtype=float
                )
                for key in ("H", "H2", "H3")
            }
            moments = {
                key: {
                    value_key: float(np.mean([r[value_key] for r in moment_replicates[key]]))
                    for value_key in ("unmit", "rem", "cdr_unmit", "cdr_rem", "exact", "noiseless")
                }
                for key in ("H", "H2", "H3")
            }
            for key in ("H", "H2", "H3"):
                moments[key]["replicates"] = [dict(r) for r in moment_replicates[key]]
                moments[key]["sample_variance"] = float(np.var(moment_arrays[key], ddof=1))
                moments[key]["mean_variance"] = float(
                    moments[key]["sample_variance"] / CME_VARIANCE_REPEATS
                )
            shots = {"H": CME_H_NUM_SHOTS, "H2": h2_h3_shots, "H3": h2_h3_shots}

            src = {k: moments[k][CME_MOMENT_SOURCE] for k in ("H", "H2", "H3")}
            E_cme, C1, C2, C3, E_cme_variance, E_cme_sigma, E_cme_gradient = _cme_k3_with_variance(
                src["H"], src["H2"], src["H3"],
                moments["H"]["mean_variance"],
                moments["H2"]["mean_variance"],
                moments["H3"]["mean_variance"],
            )
            E_cme_minus_1sigma = float(E_cme - E_cme_sigma)
            E_cme_minus_2sigma = float(E_cme - 2.0 * E_cme_sigma)
            E_cme_replicates = np.asarray(
                [
                    _cme_k3(moment_arrays["H"][i], moment_arrays["H2"][i], moment_arrays["H3"][i])[0]
                    for i in range(CME_VARIANCE_REPEATS)
                ],
                dtype=float,
            )
            finite_replicates = E_cme_replicates[np.isfinite(E_cme_replicates)]
            E_cme_empirical_variance = (
                float(np.var(finite_replicates, ddof=1) / finite_replicates.size)
                if finite_replicates.size >= 2 else float("nan")
            )
            E_cme_empirical_sigma = float(np.sqrt(E_cme_empirical_variance))
            E_cme_exact, _, _, _ = _cme_k3(moments["H"]["exact"], moments["H2"]["exact"], moments["H3"]["exact"])
            E_cme_noiseless, _, _, _ = _cme_k3(
                moments["H"]["noiseless"], moments["H2"]["noiseless"], moments["H3"]["noiseless"]
            )

            print(f"=== Connected Moment Expansion (k=3), <H^2>/<H^3> shot multiplier = {mult}x ===")
            print(f"moment source for formula : {CME_MOMENT_SOURCE}")
            print(f"<H>={src['H']:+.8f}  <H^2>={src['H2']:+.8f}  <H^3>={src['H3']:+.8f}")
            print(f"connected moments         : c1={C1:+.6e}  c2={C2:+.6e}  c3={C3:+.6e}")
            # Per-moment bias vs the NOISELESS reference, exactly like the VQE loop's
            # |REM+CF - noiseless| diagnostic (this is the meaningful CDR+REM error).
            print(
                "moment vs NOISELESS       : "
                + "  ".join(
                    f"|<{k}>{CME_MOMENT_SOURCE} - <{k}>nls|={abs(moments[k][CME_MOMENT_SOURCE] - moments[k]['noiseless']):.3e}"
                    for k in ("H", "H2", "H3")
                )
            )
            print(f"E_CME(k=3) [shots/{CME_MOMENT_SOURCE}]      = {E_cme:.10f} Eh")
            print(f"Var[E_CME] (Eq. 73 propagation)             = {E_cme_variance:.6e} Eh^2")
            print(f"sigma[E_CME]                                = {E_cme_sigma:.6e} Eh")
            print(f"E_CME - 1 sigma                             = {E_cme_minus_1sigma:.10f} Eh")
            print(f"E_CME - 2 sigma                             = {E_cme_minus_2sigma:.10f} Eh")
            print(f"empirical sigma of replicated CMX mean      = {E_cme_empirical_sigma:.6e} Eh")
            print(f"E_CME(k=3) [noiseless moments] = {E_cme_noiseless:.10f} Eh")
            print(f"E_CME(k=3) [exact Tr moments]  = {E_cme_exact:.10f} Eh")
            print(f"true ground-state energy e_gs  = {e_ref:.10f} Eh")
            print(f"|E_CME - e_gs|       (shots)    = {abs(E_cme - e_ref):.6e} Eh")
            print(f"|<H>   - e_gs|       (shots)    = {abs(src['H'] - e_ref):.6e} Eh")
            print(f"|<H>cdr+rem - <H>noiseless|     = {abs(src['H'] - moments['H']['noiseless']):.6e} Eh  (cf. VQE loop ~1e-2)")
            print(f"|E_CME(shots) - E_CME(noiseless)| = {abs(E_cme - E_cme_noiseless):.6e} Eh")

            cme_results_by_multiplier[mult] = {
                "params_best": params_cmx.copy(),
                "shots": dict(shots),
                "moments": {k: dict(v) for k, v in moments.items()},
                "connected_moments": {"c1": float(C1), "c2": float(C2), "c3": float(C3)},
                "E_cme_shots": float(E_cme),
                "E_cme_variance": float(E_cme_variance),
                "E_cme_sigma": float(E_cme_sigma),
                "E_cme_minus_1sigma": float(E_cme_minus_1sigma),
                "E_cme_minus_2sigma": float(E_cme_minus_2sigma),
                "E_cme_gradient": np.asarray(E_cme_gradient, dtype=float),
                "E_cme_replicates": E_cme_replicates,
                "E_cme_empirical_variance": float(E_cme_empirical_variance),
                "E_cme_empirical_sigma": float(E_cme_empirical_sigma),
                "E_cme_noiseless": float(E_cme_noiseless),
                "E_cme_exact": float(E_cme_exact),
                "e_gs": float(e_ref),
                "moment_source": CME_MOMENT_SOURCE,
                "h2_h3_shot_multiplier": mult,
                "variance_repeats": int(CME_VARIANCE_REPEATS),
            }

        print("\n=== CME(k=3) summary across <H^2>/<H^3> shot multipliers ===")
        for mult in CME_SHOT_MULTIPLIERS:
            r = cme_results_by_multiplier[mult]
            print(
                f"{mult:>3d}x (shots={r['shots']['H2']:>8d})  "
                f"E_CME={r['E_cme_shots']:.10f} +/- {r['E_cme_sigma']:.3e} Eh  "
                f"E-1sigma={r['E_cme_minus_1sigma']:.10f}  "
                f"E-2sigma={r['E_cme_minus_2sigma']:.10f}  "
                f"|E_CME - e_gs|={abs(r['E_cme_shots'] - e_ref):.6e} Eh"
            )

        # Backward-compatible alias for cells expecting a single ``cme_results`` dict:
        # the largest-multiplier sweep result (closest to the shot budget this cell used
        # before the sweep was added).
        cme_results = cme_results_by_multiplier[CME_SHOT_MULTIPLIERS[-1]]
    except FileNotFoundError as _cme_missing_file:

        print(f"[cloud-results] Skipping CME/CMX because an input file is missing: {_cme_missing_file}")

        cme_results_by_multiplier = {}

        cme_results = None




_run_cmx()


# Save final VQE + CME payloads for local plot recreation.
_saved_final_paths = save_checkpoint(
    data_dir=Path("data"),
    molecule=MOLECULE,
    bond_length=float(bond_length),
    stage="final",
    vqe_results=globals().get("vqe_results"),
    cme_results=globals().get("cme_results"),
    cme_results_by_multiplier=globals().get("cme_results_by_multiplier"),
    metadata={
        "circuit_name": CIRCUIT_NAME,
        "measurement_scheme": GLOBAL_MEASUREMENT_SCHEME,
        "num_shots": int(GLOBAL_NUM_SHOTS),
        "cdr_training_circuits": int(CDR_NUM_TRAINING_CIRCUITS),
        "vqe_iters": int(VQE_ITERS),
    },
)
print("Saved cloud result files:")
for _name, _path in sorted(_saved_final_paths.items()):
    print(f"  {_name}: {_path}")
