from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import cirq
import numpy as np
import sympy

PAULI_CHAR_TO_GATE = {"I": None, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

# LiH simplified gate-only noise: depolarizing strength depends only on gate arity (2Q vs 1Q).
TWO_QUBIT_GATE_DEPOL_PROB = 0.018
ONE_QUBIT_GATE_DEPOL_PROB = 0.0018

# Legacy notebooks referred to a single “depol_prob”; that matched the two-qubit channel strength.
DEFAULT_DEPOL_PROB = TWO_QUBIT_GATE_DEPOL_PROB


class GateArityDepolarizingNoise(cirq.NoiseModel):
    """Single-qubit depolarizing noise applied after each gate, keyed by arity.

    - Two-qubit gates: ``two_qubit_depol_prob`` on each qubit (after the gate).
    - One-qubit gates: ``one_qubit_depol_prob`` on that qubit.

    Measurements are unchanged (readout error belongs in shot estimation).

    ``depol_prob`` is accepted for backward compatibility but ignored; use the explicit
    ``two_qubit_depol_prob`` / ``one_qubit_depol_prob`` kwargs or class defaults.
    ``self.depol_prob`` mirrors ``two_qubit_depol_prob`` for legacy introspection.
    """

    def __init__(
        self,
        *,
        two_qubit_depol_prob: float | None = None,
        one_qubit_depol_prob: float | None = None,
        depol_prob: float | None = None,
    ):
        _ = depol_prob
        self.two_qubit_depol_prob = float(
            two_qubit_depol_prob
            if two_qubit_depol_prob is not None
            else TWO_QUBIT_GATE_DEPOL_PROB
        )
        self.one_qubit_depol_prob = float(
            one_qubit_depol_prob
            if one_qubit_depol_prob is not None
            else ONE_QUBIT_GATE_DEPOL_PROB
        )
        self.depol_prob = self.two_qubit_depol_prob

    def noisy_operation(self, operation: cirq.Operation):
        if isinstance(operation.gate, cirq.MeasurementGate):
            yield operation
            return

        n = len(operation.qubits)
        if n == 2:
            yield operation
            p2 = min(1.0, max(0.0, self.two_qubit_depol_prob))
            for q in operation.qubits:
                yield cirq.depolarize(p2).on(q)
            return
        if n == 1:
            yield operation
            p1 = min(1.0, max(0.0, self.one_qubit_depol_prob))
            yield cirq.depolarize(p1).on(operation.qubits[0])
            return

        yield operation


def load_hamiltonian_paths(
    workspace: Path,
    h_atom: int,
    bond_length: float | int,
    *,
    hamiltonian_basename: str | None = None,
) -> tuple[Path, Path, Path]:
    local_folder = workspace / "Pauli_Ham"
    colab_folder = Path("/content/drive/My Drive/Quantum_chemistry/pauli_Ham")
    save_folder = local_folder if local_folder.exists() else colab_folder
    bond_token = f"{bond_length}".rstrip("0").rstrip(".") if isinstance(bond_length, float) else str(bond_length)
    if hamiltonian_basename is not None:
        stem = f"{hamiltonian_basename}_bond_{bond_token}"
    else:
        stem = f"H{h_atom}_bond_{bond_token}"
    # Prefer OpenFermion pickle name …_of.pkl; fall back to legacy … .pkl
    pkl_path = save_folder / f"{stem}_of.pkl"
    if not pkl_path.is_file():
        legacy_pkl = save_folder / f"{stem}.pkl"
        if legacy_pkl.is_file():
            pkl_path = legacy_pkl
    text_path = save_folder / f"{stem}_pauli_convention.txt"
    return save_folder, pkl_path, text_path


def pauli_text_to_pauli_sum(path: Path, qubits: list[cirq.Qid]) -> cirq.PauliSum:
    """Each Pauli word's characters map left-to-right to ``qubits[0], qubits[1], ...``."""

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected alternating Pauli/coeff lines in {path}")

    pauli_sum = cirq.PauliSum()
    for pauli_word, coeff_text in zip(lines[0::2], lines[1::2]):
        if len(pauli_word) != len(qubits):
            raise ValueError(
                f"Pauli word length {len(pauli_word)} does not match {len(qubits)} qubits: {pauli_word}"
            )
        coefficient = complex(coeff_text)
        pauli_string = cirq.PauliString(coefficient)
        for idx, pauli_char in enumerate(pauli_word):
            gate = PAULI_CHAR_TO_GATE[pauli_char]
            if gate is not None:
                pauli_string *= gate(qubits[idx])
        pauli_sum += pauli_string
    return pauli_sum


def qubit_operator_to_pauli_sum(qubit_operator, qubits: list[cirq.Qid]) -> cirq.PauliSum:
    pauli_map = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
    pauli_sum = cirq.PauliSum()
    for term, coefficient in qubit_operator.terms.items():
        pauli_string = cirq.PauliString(coefficient)
        for qubit_idx, operator_str in term:
            pauli_string *= pauli_map[operator_str](qubits[qubit_idx])
        pauli_sum += pauli_string
    return pauli_sum


def load_observable_h(
    workspace: Path,
    ansatz_qubits: list[cirq.Qid],
    h_atom: int,
    bond_length: float | int,
    *,
    hamiltonian_basename: str | None = None,
) -> cirq.PauliSum:
    save_folder, pkl_path, text_path = load_hamiltonian_paths(
        workspace, h_atom, bond_length, hamiltonian_basename=hamiltonian_basename
    )
    if pkl_path.exists():
        try:
            import openfermion as of  # noqa: F401

            with pkl_path.open("rb") as f:
                h_qubit_loaded = pickle.load(f)
            return qubit_operator_to_pauli_sum(h_qubit_loaded, ansatz_qubits)
        except ModuleNotFoundError:
            pass

    if text_path.exists():
        return pauli_text_to_pauli_sum(text_path, ansatz_qubits)

    raise FileNotFoundError(
        f"Could not load Hamiltonian from {pkl_path} or {text_path} under {save_folder}."
    )


def trace_energy(hamiltonian: np.ndarray, rho: np.ndarray) -> float:
    return np.trace(hamiltonian @ rho).real


def stable_r2_from_sums(
    ss_res: float,
    ss_tot: float,
    *,
    ss_res_tol: float = 1e-4,
    ss_tot_tol: float = 1e-4,
) -> float:
    """Numerically stable R^2 from residual/total sum of squares.

    If residual error is already tiny, treat as a perfect fit to avoid noisy
    or NaN-like behavior in near-degenerate terms.
    """
    ss_res_f = float(ss_res)
    ss_tot_f = float(ss_tot)
    if ss_res_f <= float(ss_res_tol):
        return 1.0
    if ss_tot_f <= float(ss_tot_tol):
        return 1.0
    return float(1.0 - ss_res_f / ss_tot_f)


def scale_noise_params_for_zne(
    noise_scale: float,
    *,
    two_qubit_depol_prob: float = TWO_QUBIT_GATE_DEPOL_PROB,
    one_qubit_depol_prob: float = ONE_QUBIT_GATE_DEPOL_PROB,
) -> dict[str, float]:
    if noise_scale <= 0:
        raise ValueError(f"noise_scale must be > 0, got {noise_scale}.")

    def clip01(value: float) -> float:
        return float(min(1.0, max(0.0, value)))

    return {
        "two_qubit_depol_prob": clip01(two_qubit_depol_prob * noise_scale),
        "one_qubit_depol_prob": clip01(one_qubit_depol_prob * noise_scale),
    }


def trace_energy_at_noise_scale(
    ansatz_circuit: cirq.Circuit,
    resolver: cirq.ParamResolver,
    qubits: list[cirq.Qid],
    hamiltonian_matrix: np.ndarray,
    *,
    noise_scale: float,
    two_qubit_depol_prob: float = TWO_QUBIT_GATE_DEPOL_PROB,
    one_qubit_depol_prob: float = ONE_QUBIT_GATE_DEPOL_PROB,
    simulator_seed: int = 1234,
) -> float:
    scaled = scale_noise_params_for_zne(
        noise_scale,
        two_qubit_depol_prob=two_qubit_depol_prob,
        one_qubit_depol_prob=one_qubit_depol_prob,
    )
    noise_model = GateArityDepolarizingNoise(**scaled)
    noisy_circuit = ansatz_circuit.with_noise(noise_model)
    resolved_noisy_circuit = cirq.resolve_parameters(noisy_circuit, resolver)
    rho_noisy = cirq.DensityMatrixSimulator(seed=simulator_seed).simulate(
        resolved_noisy_circuit, qubit_order=qubits
    ).final_density_matrix
    return float(trace_energy(hamiltonian_matrix, rho_noisy))


def zne_extrapolate_energy(
    noise_scales: list[float], energies: list[float], fit_order: int = 1
) -> dict[str, object]:
    if len(noise_scales) != len(energies):
        raise ValueError(
            f"noise_scales and energies must have same length, got {len(noise_scales)} and {len(energies)}."
        )
    if len(noise_scales) < 2:
        raise ValueError("ZNE requires at least two scale points.")
    if fit_order < 1:
        raise ValueError(f"fit_order must be >= 1, got {fit_order}.")
    if fit_order >= len(noise_scales):
        raise ValueError(
            f"fit_order must be < number of points, got fit_order={fit_order}, points={len(noise_scales)}."
        )
    if any(scale <= 0 for scale in noise_scales):
        raise ValueError(f"All noise scales must be > 0, got {noise_scales}.")

    scales = np.asarray(noise_scales, dtype=float)
    values = np.asarray(energies, dtype=float)
    coeffs = np.polyfit(scales, values, deg=fit_order)
    energy_zne = float(np.polyval(coeffs, 0.0))
    return {
        "noise_scales": [float(x) for x in scales.tolist()],
        "energies": [float(x) for x in values.tolist()],
        "fit_order": int(fit_order),
        "fit_coefficients": [float(x) for x in coeffs.tolist()],
        "energy_zne": energy_zne,
    }


def run_trace_zne(
    ansatz_circuit: cirq.Circuit,
    resolver: cirq.ParamResolver,
    qubits: list[cirq.Qid],
    hamiltonian_matrix: np.ndarray,
    *,
    noise_scales: list[float] | tuple[float, ...] = (1.0, 2.0, 3.0),
    fit_order: int = 1,
    simulator_seed: int = 1234,
    two_qubit_depol_prob: float = TWO_QUBIT_GATE_DEPOL_PROB,
    one_qubit_depol_prob: float = ONE_QUBIT_GATE_DEPOL_PROB,
) -> dict[str, object]:
    scales = [float(s) for s in noise_scales]
    trace_energies = [
        trace_energy_at_noise_scale(
            ansatz_circuit,
            resolver,
            qubits,
            hamiltonian_matrix,
            noise_scale=scale,
            two_qubit_depol_prob=two_qubit_depol_prob,
            one_qubit_depol_prob=one_qubit_depol_prob,
            simulator_seed=simulator_seed,
        )
        for scale in scales
    ]
    zne = zne_extrapolate_energy(scales, trace_energies, fit_order=fit_order)
    smallest_idx = int(np.argmin(np.asarray(scales)))
    return {
        "noise_scales": zne["noise_scales"],
        "trace_energies": [float(x) for x in trace_energies],
        "energy_zne": float(zne["energy_zne"]),
        "fit_order": int(zne["fit_order"]),
        "fit_coefficients": [float(x) for x in zne["fit_coefficients"]],
        "baseline_noisy_energy": float(trace_energies[smallest_idx]),
    }


def states_equal_up_to_global_phase(lhs: np.ndarray, rhs: np.ndarray, atol: float = 1e-7) -> bool:
    return np.isclose(abs(np.vdot(lhs, rhs)), 1.0, atol=atol)


def is_cz_plus_single_qubit(operations: Iterable[cirq.Operation]) -> bool:
    for op in operations:
        if len(op.qubits) == 1:
            continue
        if len(op.qubits) == 2 and isinstance(op.gate, cirq.CZPowGate):
            if np.isclose(float(op.gate.exponent), 1.0):
                continue
        return False
    return True


# ---------------------------------------------------------------------------
# Clifford Data Regression (CDR) helpers
# ---------------------------------------------------------------------------

# Gate types that have a numeric `exponent` attribute and are Clifford only when
# 2 * exponent is (approximately) an integer.
_EXPONENT_GATE_TYPES: tuple[type, ...] = (
    cirq.PhasedXPowGate,
    cirq.ZPowGate,
    cirq.XPowGate,
    cirq.YPowGate,
)


def is_clifford_exponent(exponent: float, atol: float = 1e-9) -> bool:
    """A `*PowGate` is Clifford iff `2 * exponent` is (approximately) an integer.

    Examples (Clifford): exponent in {..., -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, ...}.
    Examples (non-Clifford): exponent = 0.25 (T), 0.7, etc.
    """
    val = 2.0 * float(exponent)
    return abs(val - round(val)) <= atol


def clifford_snap_theta(value: float) -> float:
    """Snap a `theta` parameter to its nearest Clifford-equivalent angle.

    The decomposition uses `theta / pi` as the exponent of two `PhasedXPowGate`s,
    so we want exponent in multiples of 0.5, i.e., `theta` in multiples of `pi/2`.
    """
    step = float(np.pi) / 2.0
    return float(round(float(value) / step) * step)


def clifford_snap_phi(value: float) -> float:
    """Snap a `phi` parameter to its nearest Clifford-equivalent angle.

    The decomposition uses `+/- phi / (2 pi)` as the exponent of three
    `ZPowGate`s. To force exponents to multiples of 0.5 we need
    `phi` in multiples of `pi`.
    """
    step = float(np.pi)
    return float(round(float(value) / step) * step)


def _is_symbol_theta(symbol: sympy.Symbol) -> bool:
    return str(symbol).startswith("th_")


def _is_symbol_phi(symbol: sympy.Symbol) -> bool:
    return str(symbol).startswith("ph_")


def clifford_snap_value_for_symbol(symbol: sympy.Symbol, value: float) -> float:
    """Apply the right Clifford snap rule based on whether the symbol is theta or phi."""
    if _is_symbol_theta(symbol):
        return clifford_snap_theta(value)
    if _is_symbol_phi(symbol):
        return clifford_snap_phi(value)
    raise ValueError(
        f"Unrecognized symbol naming convention for Clifford snap: {symbol}. "
        "Expected names starting with 'th_' or 'ph_'."
    )


def count_non_clifford_ops(
    circuit: cirq.Circuit, resolver: dict | cirq.ParamResolver
) -> int:
    """Count operations in `circuit` that become non-Clifford after substituting
    `resolver` into all symbolic parameters.

    Counts every `PhasedXPowGate`, `ZPowGate`, `XPowGate`, `YPowGate` whose
    `exponent` is NOT (approximately) a multiple of 0.5 after parameter
    resolution. `H`, `CZ`, `MeasurementGate`, and identity-like exponents
    (e.g. `XPowGate(exponent=1)`) are skipped.
    """
    resolved = cirq.resolve_parameters(circuit, resolver)
    count = 0
    for moment in resolved:
        for op in moment.operations:
            gate = op.gate
            if gate is None:
                continue
            if not isinstance(gate, _EXPONENT_GATE_TYPES):
                continue
            try:
                exponent = float(gate.exponent)
            except (TypeError, ValueError):
                continue
            if not is_clifford_exponent(exponent):
                count += len(op.qubits)
    return count


def generate_near_clifford_param_sets(
    target_params: dict[sympy.Symbol, float] | dict[str, float],
    symbols: list[sympy.Symbol],
    *,
    num_circuits: int,
    t_max: int,
    circuit: cirq.Circuit,
    min_snap_fraction: float = 0.0,
    seed: int = 0,
) -> list[dict[sympy.Symbol, float]]:
    """Generate `num_circuits` near-Clifford resolvers in two explicit steps.

    Step 1:
      Randomly choose which parameterized gates are replaced by nearest
      Clifford angles (snapped from target parameters), enforcing:
      ``(total_count - non_clifford_count) == t_max``.

    Step 2:
      For the remaining parameterized gates (the non-Clifford part), assign
      rotation angles independently and uniformly from ``[0, 2π]``.

    ``min_snap_fraction`` is retained only for API compatibility and ignored.
    """
    if num_circuits <= 0:
        raise ValueError(f"num_circuits must be > 0, got {num_circuits}.")
    if t_max < 0:
        raise ValueError(f"t_max must be >= 0, got {t_max}.")
    _ = float(min_snap_fraction)  # Backward-compatible no-op.

    target_by_symbol: dict[sympy.Symbol, float] = {}
    for sym in symbols:
        if sym in target_params:
            target_by_symbol[sym] = float(target_params[sym])
        elif str(sym) in target_params:
            target_by_symbol[sym] = float(target_params[str(sym)])
        else:
            raise KeyError(f"Target parameter missing for symbol {sym!s}.")

    # Estimate total count when all parameterized gates are non-Clifford.
    probe_rng = np.random.default_rng(int(seed) + 99991)
    total_count = 0
    for _ in range(12):
        probe_resolver = {
            sym: float(probe_rng.uniform(0.0, 2.0 * np.pi)) for sym in symbols
        }
        total_count = max(total_count, count_non_clifford_ops(circuit, probe_resolver))
    if t_max > total_count:
        raise ValueError(
            f"t_max={t_max} exceeds achievable Clifford count upper bound {total_count}."
        )

    resolvers: list[dict[sympy.Symbol, float]] = []
    for circ_idx in range(num_circuits):
        local_rng = np.random.default_rng(int(seed) + 1000 * (circ_idx + 1))
        found: dict[sympy.Symbol, float] | None = None

        for _attempt in range(96):
            # Step 1: start fully Clifford, then randomize selected symbols
            # until (total_count - non_clifford_count) == t_max.
            resolver: dict[sympy.Symbol, float] = {
                sym: clifford_snap_value_for_symbol(sym, target_by_symbol[sym])
                for sym in symbols
            }
            snapped: set[sympy.Symbol] = set(symbols)
            unsnapped: set[sympy.Symbol] = set()

            current_non_cliff = count_non_clifford_ops(circuit, resolver)
            current_cliff = total_count - current_non_cliff
            order = list(symbols)
            local_rng.shuffle(order)

            for sym in order:
                if current_cliff <= t_max:
                    break
                old_value = resolver[sym]
                changed = False
                for _ in range(24):
                    trial = float(local_rng.uniform(0.0, 2.0 * np.pi))
                    resolver[sym] = trial
                    trial_non_cliff = count_non_clifford_ops(circuit, resolver)
                    trial_cliff = total_count - trial_non_cliff
                    if t_max <= trial_cliff < current_cliff:
                        current_non_cliff = trial_non_cliff
                        current_cliff = trial_cliff
                        snapped.remove(sym)
                        unsnapped.add(sym)
                        changed = True
                        break
                if not changed:
                    resolver[sym] = old_value

            if current_cliff != t_max:
                continue

            # Step 2: assign remaining (non-Clifford) symbols independently
            # from Uniform[0, 2π], then keep exact Clifford count via repair.
            for sym in list(unsnapped):
                resolver[sym] = float(local_rng.uniform(0.0, 2.0 * np.pi))

            repair_guard = 0
            while repair_guard < 64:
                repair_guard += 1
                current_non_cliff = count_non_clifford_ops(circuit, resolver)
                current_cliff = total_count - current_non_cliff
                if current_cliff == t_max:
                    found = dict(resolver)
                    break
                if current_cliff > t_max and snapped:
                    pick = list(snapped)[int(local_rng.integers(0, len(snapped)))]
                    old = resolver[pick]
                    resolver[pick] = float(local_rng.uniform(0.0, 2.0 * np.pi))
                    trial_non_cliff = count_non_clifford_ops(circuit, resolver)
                    if (total_count - trial_non_cliff) <= current_cliff:
                        snapped.remove(pick)
                        unsnapped.add(pick)
                    else:
                        resolver[pick] = old
                elif current_cliff < t_max and unsnapped:
                    pick = list(unsnapped)[int(local_rng.integers(0, len(unsnapped)))]
                    resolver[pick] = clifford_snap_value_for_symbol(
                        pick, target_by_symbol[pick]
                    )
                    unsnapped.remove(pick)
                    snapped.add(pick)
                else:
                    break

            if found is not None:
                break

        if found is None:
            raise ValueError(
                f"Could not construct resolver with exact Clifford count t_max={t_max}."
            )
        resolvers.append(found)

    return resolvers


def random_clifford_theta(rng: np.random.Generator) -> float:
    """Sample θ uniformly from Clifford-equivalent angles {0, π/2, π, 3π/2}."""
    k = int(rng.integers(0, 4))
    return float(k * (np.pi / 2.0))


def random_clifford_phi(rng: np.random.Generator) -> float:
    """Sample φ uniformly from integer multiples of π (Clifford grid for φ blocks)."""
    m = int(rng.integers(-2, 3))
    return float(m * np.pi)


def random_clifford_value_for_symbol(symbol: sympy.Symbol, rng: np.random.Generator) -> float:
    """Random Clifford angle for a `th_*` or `ph_*` symbol (paper-style analogue circuits)."""
    if _is_symbol_theta(symbol):
        return random_clifford_theta(rng)
    if _is_symbol_phi(symbol):
        return random_clifford_phi(rng)
    raise ValueError(
        f"Unrecognized symbol naming convention for random Clifford: {symbol}. "
        "Expected names starting with 'th_' or 'ph_'."
    )


def generate_random_clifford_analogue_param_sets(
    target_params: dict[sympy.Symbol, float] | dict[str, float],
    symbols: list[sympy.Symbol],
    *,
    num_circuits: int,
    circuit: cirq.Circuit,
    seed: int = 0,
) -> list[dict[sympy.Symbol, float]]:
    """Generate training resolvers by assigning each symbol an independent random Clifford angle.

    Analogous to Guo et al.: Clifford-analogue circuits with random single-qubit Cliffords
    on the same circuit layout. Every resolved circuit is fully Clifford (``t_remaining`` = 0).

    Note: ``t_max`` does not apply to this strategy and is ignored by callers.
    """
    _ = circuit  # Unused; included for API symmetry with ``generate_near_clifford_param_sets``.
    if num_circuits <= 0:
        raise ValueError(f"num_circuits must be > 0, got {num_circuits}.")

    target_by_symbol: dict[sympy.Symbol, float] = {}
    for sym in symbols:
        if sym in target_params:
            target_by_symbol[sym] = float(target_params[sym])
        elif str(sym) in target_params:
            target_by_symbol[sym] = float(target_params[str(sym)])
        else:
            raise KeyError(f"Target parameter missing for symbol {sym!s}.")

    resolvers: list[dict[sympy.Symbol, float]] = []
    for circ_idx in range(num_circuits):
        local_rng = np.random.default_rng(int(seed) + 1000 * (circ_idx + 1))
        resolver: dict[sympy.Symbol, float] = {}
        for sym in symbols:
            resolver[sym] = random_clifford_value_for_symbol(sym, local_rng)
        resolvers.append(resolver)

    return resolvers


def run_cdr_with_per_pauli_coeff_print(
    *,
    ansatz_circuit: cirq.Circuit,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    target_resolver: dict,
    target_params: dict,
    symbols: list,
    base_noise_cfg: dict,
    shot_cfg: dict,
    readout_cal: dict,
    cdr_cfg: dict | None = None,
    simulator_seed: int = 1234,
) -> dict[str, object]:
    """Run CDR once and print per-Pauli affine coefficients (a_k, b_k).

    This helper is opt-in and keeps the default CDR path silent in other cells.
    """
    # Local import avoids introducing a module-level dependency cycle.
    from shot_measurement_test_LiH import run_mitigation

    cdr_cfg_local = dict(cdr_cfg or {})
    cdr_cfg_local.setdefault("cdr_fit_scope", "per_pauli")

    out = run_mitigation(
        "cdr",
        ansatz_circuit=ansatz_circuit,
        observable_h=observable_h,
        qubits=qubits,
        target_resolver=target_resolver,
        target_params=target_params,
        symbols=symbols,
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
        readout_cal=readout_cal,
        cdr_cfg=cdr_cfg_local,
        simulator_seed=int(simulator_seed),
    )

    models = out.get("cdr_models", {})
    coeffs_rem = np.asarray(models.get("coeffs_rem_to_exact_per_term", []), dtype=float)
    coeffs_unmit = np.asarray(models.get("coeffs_unmit_to_exact_per_term", []), dtype=float)
    weights = np.asarray(models.get("weights", []), dtype=float)

    qubit_to_idx = {q: i for i, q in enumerate(qubits)}
    term_labels: list[str] = []
    for pauli_term in observable_h:
        pauli_map = dict(pauli_term.items())
        if len(pauli_map) == 0:
            # Identity offset is not part of per-term CF coefficients.
            continue
        chars = ["I"] * len(qubits)
        for q, op in pauli_map.items():
            chars[qubit_to_idx[q]] = str(op)
        term_labels.append("".join(chars))

    print("CDR per-Pauli coefficients (exact_k ~= a_k * noisy_k + b_k)")
    if coeffs_rem.size == 0:
        print("No per-Pauli coefficients found. Ensure cdr_fit_scope='per_pauli'.")
        return out

    print(f"Number of Pauli terms: {len(coeffs_rem)}")
    print("REM branch coefficients:")
    for k, (a_k, b_k) in enumerate(coeffs_rem):
        label = term_labels[k] if k < len(term_labels) else "UNKNOWN"
        weight = float(weights[k]) if k < len(weights) else float("nan")
        print(
            f"term[{k:03d}] {label}  weight={weight: .12f}  "
            f"a={float(a_k): .12f}, b={float(b_k): .12f}"
        )

    if coeffs_unmit.size:
        print("UNMIT branch coefficients:")
        for k, (a_k, b_k) in enumerate(coeffs_unmit):
            label = term_labels[k] if k < len(term_labels) else "UNKNOWN"
            weight = float(weights[k]) if k < len(weights) else float("nan")
            print(
                f"term[{k:03d}] {label}  weight={weight: .12f}  "
                f"a={float(a_k): .12f}, b={float(b_k): .12f}"
            )

    return out

