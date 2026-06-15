from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import cirq
import numpy as np
import sympy

PAULI_CHAR_TO_GATE = {"I": None, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

# Gate-only noise: depolarizing strength depends on gate arity (2Q vs 1Q).
TWO_QUBIT_GATE_DEPOL_PROB = 0.001
ONE_QUBIT_GATE_DEPOL_PROB = 0.00005

# Legacy notebooks referred to a single “depol_prob”; that matched the two-qubit channel strength.
DEFAULT_DEPOL_PROB = TWO_QUBIT_GATE_DEPOL_PROB
DEFAULT_HIGH_CZ_MULTIPLIER = 1.0
DEFAULT_AMP_DAMP_GAMMA = 0.0
DEFAULT_PHASE_DAMP_GAMMA = 0.0
DEFAULT_LEAKAGE_APPROX_PROB = 0.0

CZ_NORMAL_TAG = "cz_normal"
CZ_HIGH_TAG = "cz_high"
CZ_ONSITE_TAG = "cz_onsite_normal"


def operation_has_tag(operation: cirq.Operation, tag: str) -> bool:
    return tag in getattr(operation, "tags", ())


def cz_tag_for_horizontal_pair(q0: cirq.Qid, q1: cirq.Qid) -> str:
    if not (isinstance(q0, cirq.GridQubit) and isinstance(q1, cirq.GridQubit)):
        raise ValueError("Horizontal CZ tagging requires GridQubit inputs.")
    if q0.row != q1.row or abs(q0.col - q1.col) != 1:
        raise ValueError(f"Qubits are not horizontal nearest neighbors: {q0}, {q1}.")
    min_col = min(q0.col, q1.col)
    return CZ_HIGH_TAG if (min_col % 2 == 1) else CZ_NORMAL_TAG


def cz_tag_for_qubit_pair(q0: cirq.Qid, q1: cirq.Qid) -> str:
    if isinstance(q0, cirq.GridQubit) and isinstance(q1, cirq.GridQubit):
        if q0.row == q1.row and abs(q0.col - q1.col) == 1:
            return cz_tag_for_horizontal_pair(q0, q1)
        if q0.col == q1.col and abs(q0.row - q1.row) == 1:
            return CZ_ONSITE_TAG
    return CZ_NORMAL_TAG


def ordered_parameter_symbols(
    num_spatial_orbitals: int, num_layers: int
) -> list[sympy.Symbol]:
    """Match the ansatz symbol naming convention used in notebooks/tests."""
    symbols: list[sympy.Symbol] = []
    p_idx = 0
    for layer in range(num_layers):
        for _ in range(0, num_spatial_orbitals - 1, 2):
            symbols.append(sympy.Symbol(f"th_{layer}_{p_idx}"))
            p_idx += 1
        for _ in range(1, num_spatial_orbitals - 1, 2):
            symbols.append(sympy.Symbol(f"th_{layer}_{p_idx}"))
            p_idx += 1
        for _ in range(num_spatial_orbitals):
            symbols.append(sympy.Symbol(f"ph_{layer}_{p_idx}"))
            p_idx += 1
    return symbols


def prepare_original_fsim_ansatz_cirq(
    num_spatial_orbitals: int, num_layers: int = 1
) -> tuple[cirq.Circuit, list[cirq.GridQubit]]:
    """Construct the original symbolic FSim ansatz."""
    qubits = [cirq.GridQubit(0, i) for i in range(num_spatial_orbitals)] + [
        cirq.GridQubit(1, i) for i in range(num_spatial_orbitals)
    ]
    circuit = cirq.Circuit()
    p_idx = 0
    circuit.append([cirq.X(qubits[i]) for i in range(1, len(qubits), 2)])

    for layer in range(num_layers):
        even_odd_moments = []
        for i in range(0, num_spatial_orbitals - 1, 2):
            theta = sympy.Symbol(f"th_{layer}_{p_idx}")
            even_odd_moments.append(cirq.FSimGate(theta, 0).on(qubits[i], qubits[i + 1]))
            even_odd_moments.append(
                cirq.FSimGate(theta, 0).on(
                    qubits[i + num_spatial_orbitals], qubits[i + 1 + num_spatial_orbitals]
                )
            )
            p_idx += 1
        circuit.append(even_odd_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        odd_even_moments = []
        for i in range(1, num_spatial_orbitals - 1, 2):
            theta = sympy.Symbol(f"th_{layer}_{p_idx}")
            odd_even_moments.append(cirq.FSimGate(theta, 0).on(qubits[i], qubits[i + 1]))
            odd_even_moments.append(
                cirq.FSimGate(theta, 0).on(
                    qubits[i + num_spatial_orbitals], qubits[i + 1 + num_spatial_orbitals]
                )
            )
            p_idx += 1
        circuit.append(odd_even_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        onsite_moments = []
        for i in range(num_spatial_orbitals):
            phi = sympy.Symbol(f"ph_{layer}_{p_idx}")
            onsite_moments.append(
                cirq.FSimGate(0, phi).on(qubits[i], qubits[i + num_spatial_orbitals])
            )
            p_idx += 1
        circuit.append(onsite_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circuit, qubits


def _is_zero_like(value: object, atol: float = 1e-12) -> bool:
    if isinstance(value, (int, float, np.floating)):
        return bool(np.isclose(float(value), 0.0, atol=atol))
    if isinstance(value, sympy.Basic):
        return bool(sympy.simplify(value) == 0)
    try:
        return bool(np.isclose(float(value), 0.0, atol=atol))
    except (TypeError, ValueError):
        return False


def _tag_cz_ops(ops: list[cirq.Operation], tag: str) -> list[cirq.Operation]:
    tagged: list[cirq.Operation] = []
    for op in ops:
        if isinstance(op.gate, cirq.CZPowGate) and np.isclose(float(op.gate.exponent), 1.0):
            tagged.append(op.with_tags(tag))
        else:
            tagged.append(op)
    return tagged


def decompose_ansatz_fsim_ops(fsim_circuit: cirq.Circuit) -> cirq.Circuit:
    """Replace each FSim(theta,0)/(0,phi) with CZ+1Q decomposition and CZ tags."""
    from decompose_fsim_gate import decompose_fsim_phi_only, decompose_fsim_theta_only

    decomposed = cirq.Circuit()
    for moment in fsim_circuit:
        moment_ops: list[cirq.Operation] = []
        for op in moment.operations:
            gate = op.gate
            if not isinstance(gate, cirq.FSimGate):
                moment_ops.append(op)
                continue

            theta = gate.theta
            phi = gate.phi
            q0, q1 = op.qubits

            if _is_zero_like(phi):
                theta_ops = decompose_fsim_theta_only(theta, q0, q1)
                theta_tag = cz_tag_for_horizontal_pair(q0, q1)
                moment_ops.extend(_tag_cz_ops(theta_ops, theta_tag))
            elif _is_zero_like(theta):
                phi_ops = decompose_fsim_phi_only(phi, q0, q1)
                moment_ops.extend(_tag_cz_ops(phi_ops, CZ_ONSITE_TAG))
            else:
                raise ValueError(
                    "Encountered general FSim(theta, phi); expected theta-only or phi-only."
                )
        decomposed.append(moment_ops)
    return decomposed


def prepare_decomposed_ansatz_cirq(
    num_spatial_orbitals: int, num_layers: int = 1
) -> tuple[cirq.Circuit, list[cirq.GridQubit]]:
    """Construct symbolic decomposed (CZ + 1Q) ansatz circuit."""
    fsim_circuit, qubits = prepare_original_fsim_ansatz_cirq(
        num_spatial_orbitals=num_spatial_orbitals, num_layers=num_layers
    )
    return decompose_ansatz_fsim_ops(fsim_circuit), qubits


def build_cdr_parametrized_decomposed_template(
    num_spatial_orbitals: int,
    num_layers: int,
) -> tuple[cirq.Circuit, list[cirq.GridQubit], list[sympy.Symbol], dict[str, object]]:
    """Build symbolic decomposed template and coupled-symbol metadata for CDR."""
    circuit, qubits = prepare_decomposed_ansatz_cirq(
        num_spatial_orbitals=num_spatial_orbitals, num_layers=num_layers
    )
    symbols = ordered_parameter_symbols(
        num_spatial_orbitals=num_spatial_orbitals, num_layers=num_layers
    )
    symbol_metadata: dict[str, dict[str, object]] = {}
    for sym in symbols:
        if _is_symbol_theta(sym):
            symbol_metadata[str(sym)] = {
                "group_type": "theta_pair",
                "coupled_ops": 2,
                "phase_type_noncliff": False,
            }
        elif _is_symbol_phi(sym):
            symbol_metadata[str(sym)] = {
                "group_type": "phi_onsite",
                "coupled_ops": 3,
                "phase_type_noncliff": True,
            }
        else:
            symbol_metadata[str(sym)] = {
                "group_type": "other",
                "coupled_ops": 1,
                "phase_type_noncliff": False,
            }
    metadata: dict[str, object] = {
        "symbol_metadata": symbol_metadata,
        "non_clifford_control_group": "parameter_symbol",
    }
    return circuit, qubits, symbols, metadata


class GateArityDepolarizingNoise(cirq.NoiseModel):
    """Single-qubit depolarizing noise applied after each gate, keyed by arity.

    - Two-qubit gates: ``two_qubit_depol_prob`` on each qubit (after the gate).
      For tagged full-CZ operations, ``CZ_HIGH_TAG`` scales this rate by
      ``high_cz_multiplier``.
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
        high_cz_multiplier: float = DEFAULT_HIGH_CZ_MULTIPLIER,
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
        self.high_cz_multiplier = float(high_cz_multiplier)
        self.depol_prob = self.two_qubit_depol_prob

    def noisy_operation(self, operation: cirq.Operation):
        if isinstance(operation.gate, cirq.MeasurementGate):
            yield operation
            return

        n = len(operation.qubits)
        if n == 2:
            yield operation
            p2 = float(self.two_qubit_depol_prob)
            if isinstance(operation.gate, cirq.CZPowGate) and np.isclose(
                float(operation.gate.exponent), 1.0
            ):
                if operation_has_tag(operation, CZ_HIGH_TAG):
                    p2 *= self.high_cz_multiplier
            p2 = min(1.0, max(0.0, p2))
            for q in operation.qubits:
                yield cirq.depolarize(p2).on(q)
            return
        if n == 1:
            yield operation
            p1 = min(1.0, max(0.0, self.one_qubit_depol_prob))
            yield cirq.depolarize(p1).on(operation.qubits[0])
            return

        yield operation


class LocationAwareDecomposedNoise(cirq.NoiseModel):
    """Location-aware channel model for decomposed CZ + 1Q circuits."""

    def __init__(
        self,
        amp_damp_gamma: float = DEFAULT_AMP_DAMP_GAMMA,
        phase_damp_gamma: float = DEFAULT_PHASE_DAMP_GAMMA,
        depol_prob: float = DEFAULT_DEPOL_PROB,
        one_qubit_depol_prob: float | None = None,
        high_cz_multiplier: float = DEFAULT_HIGH_CZ_MULTIPLIER,
        leakage_approx_prob: float = DEFAULT_LEAKAGE_APPROX_PROB,
    ):
        self.amp_damp_gamma = float(amp_damp_gamma)
        self.phase_damp_gamma = float(phase_damp_gamma)
        self.depol_prob = float(depol_prob)
        self.one_qubit_depol_prob = (
            float(one_qubit_depol_prob)
            if one_qubit_depol_prob is not None
            else float(depol_prob) / 10.0
        )
        self.high_cz_multiplier = float(high_cz_multiplier)
        self.leakage_approx_prob = float(leakage_approx_prob)

    def noisy_operation(self, operation: cirq.Operation):
        if isinstance(operation.gate, cirq.MeasurementGate):
            yield operation
            return

        if isinstance(operation.gate, cirq.CZPowGate):
            yield operation
            multiplier = self.high_cz_multiplier if operation_has_tag(operation, CZ_HIGH_TAG) else 1.0
            extra_depol = self.leakage_approx_prob if operation_has_tag(operation, CZ_HIGH_TAG) else 0.0
            for q in operation.qubits:
                yield cirq.amplitude_damp(min(1.0, self.amp_damp_gamma * multiplier)).on(q)
                yield cirq.phase_damp(min(1.0, self.phase_damp_gamma * multiplier)).on(q)
                total_depol = min(1.0, (self.depol_prob * multiplier) + extra_depol)
                yield cirq.depolarize(total_depol).on(q)
            return

        if len(operation.qubits) == 1:
            yield operation
            q = operation.qubits[0]
            yield cirq.amplitude_damp(min(1.0, self.amp_damp_gamma / 10.0)).on(q)
            yield cirq.phase_damp(min(1.0, self.phase_damp_gamma / 10.0)).on(q)
            yield cirq.depolarize(min(1.0, self.one_qubit_depol_prob)).on(q)
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


def is_symbol_value_clifford(
    symbol: sympy.Symbol, value: float, atol: float = 1e-9
) -> bool:
    """Return whether a symbol value lands on the Clifford grid."""
    v = float(value)
    if _is_symbol_theta(symbol):
        return is_clifford_exponent(v / np.pi, atol=atol)
    if _is_symbol_phi(symbol):
        return is_clifford_exponent(v / (2.0 * np.pi), atol=atol)
    raise ValueError(
        f"Unrecognized symbol naming convention for Clifford check: {symbol}. "
        "Expected names starting with 'th_' or 'ph_'."
    )


def summarize_non_clifford_param_distribution(
    resolvers: list[dict[sympy.Symbol, float]],
    symbols: list[sympy.Symbol],
) -> dict[str, float]:
    """Summarize achieved non-Clifford parameter budgets across resolvers."""
    n_total = len(symbols)
    phase_symbols = [s for s in symbols if _is_symbol_phi(s)]
    n_phase = len(phase_symbols)
    non_cliff_counts: list[int] = []
    phase_snap_fracs: list[float] = []

    for resolver in resolvers:
        non_cliff = 0
        phase_non_cliff = 0
        for sym in symbols:
            if not is_symbol_value_clifford(sym, float(resolver[sym])):
                non_cliff += 1
                if _is_symbol_phi(sym):
                    phase_non_cliff += 1
        non_cliff_counts.append(non_cliff)
        if n_phase == 0:
            phase_snap_fracs.append(1.0)
        else:
            phase_snap_fracs.append(1.0 - (float(phase_non_cliff) / float(n_phase)))

    if not non_cliff_counts:
        return {
            "num_resolvers": 0.0,
            "mean_non_clifford_params": 0.0,
            "max_non_clifford_params": 0.0,
            "min_non_clifford_params": 0.0,
            "mean_phase_noncliff_snap_fraction": 1.0,
        }
    return {
        "num_resolvers": float(len(non_cliff_counts)),
        "mean_non_clifford_params": float(np.mean(non_cliff_counts)),
        "max_non_clifford_params": float(np.max(non_cliff_counts)),
        "min_non_clifford_params": float(np.min(non_cliff_counts)),
        "mean_phase_noncliff_snap_fraction": float(np.mean(phase_snap_fracs)),
    }


def count_clifford_nonclifford_gates(
    circuit: cirq.Circuit, resolver: dict | cirq.ParamResolver
) -> dict[str, int]:
    """Count total, Clifford, and non-Clifford gates after parameter resolution."""
    resolved = cirq.resolve_parameters(circuit, resolver)
    total = 0
    clifford = 0
    non_clifford = 0
    for op in resolved.all_operations():
        gate = op.gate
        if gate is None or isinstance(gate, cirq.MeasurementGate):
            continue
        total += 1

        is_cliff = False
        if isinstance(gate, _EXPONENT_GATE_TYPES):
            try:
                is_cliff = is_clifford_exponent(float(gate.exponent))
            except (TypeError, ValueError):
                is_cliff = False
        elif isinstance(gate, cirq.CZPowGate):
            try:
                is_cliff = is_clifford_exponent(float(gate.exponent))
            except (TypeError, ValueError):
                is_cliff = False
        elif gate == cirq.H:
            is_cliff = True

        if is_cliff:
            clifford += 1
        else:
            non_clifford += 1

    return {
        "total_gates": int(total),
        "clifford_gates": int(clifford),
        "non_clifford_gates": int(non_clifford),
    }


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


def _normalize_target_params(
    target_params: dict[sympy.Symbol, float] | dict[str, float],
    symbols: list[sympy.Symbol],
) -> dict[sympy.Symbol, float]:
    target_by_symbol: dict[sympy.Symbol, float] = {}
    for sym in symbols:
        if sym in target_params:
            target_by_symbol[sym] = float(target_params[sym])
        elif str(sym) in target_params:
            target_by_symbol[sym] = float(target_params[str(sym)])
        else:
            raise KeyError(f"Target parameter missing for symbol {sym!s}.")
    return target_by_symbol


def _sample_non_clifford_value_for_symbol(
    symbol: sympy.Symbol, rng: np.random.Generator
) -> float:
    """Sample a value deliberately away from the Clifford grid."""
    for _ in range(64):
        candidate = float(rng.uniform(0.0, 2.0 * np.pi))
        if not is_symbol_value_clifford(symbol, candidate, atol=1e-5):
            return candidate
    # Fallback perturbation if random retries land on/near grid.
    if _is_symbol_theta(symbol):
        return float((np.pi / 7.0) % (2.0 * np.pi))
    if _is_symbol_phi(symbol):
        return float((np.pi / 3.0) % (2.0 * np.pi))
    return float(rng.uniform(0.0, 2.0 * np.pi))


def _generate_near_clifford_param_sets_symbol_budget(
    target_params: dict[sympy.Symbol, float] | dict[str, float],
    symbols: list[sympy.Symbol],
    *,
    num_circuits: int,
    non_clifford_params_max: int,
    target_phase_noncliff_snap_fraction: float | None,
    seed: int,
) -> list[dict[sympy.Symbol, float]]:
    if num_circuits <= 0:
        raise ValueError(f"num_circuits must be > 0, got {num_circuits}.")
    if non_clifford_params_max < 0:
        raise ValueError(
            "non_clifford_params_max must be >= 0, "
            f"got {non_clifford_params_max}."
        )
    if non_clifford_params_max > len(symbols):
        raise ValueError(
            "non_clifford_params_max cannot exceed number of symbols: "
            f"{non_clifford_params_max} > {len(symbols)}."
        )
    if target_phase_noncliff_snap_fraction is None:
        phase_snap_target = 0.0
    else:
        phase_snap_target = float(target_phase_noncliff_snap_fraction)
        if not (0.0 <= phase_snap_target <= 1.0):
            raise ValueError(
                "target_phase_noncliff_snap_fraction must be in [0, 1], "
                f"got {phase_snap_target}."
            )

    target_by_symbol = _normalize_target_params(target_params, symbols)
    phase_symbols = [s for s in symbols if _is_symbol_phi(s)]
    theta_symbols = [s for s in symbols if _is_symbol_theta(s)]
    n_phase = len(phase_symbols)
    desired_phase_snapped = int(np.ceil(phase_snap_target * n_phase))
    max_unsnapped_phase = max(0, n_phase - desired_phase_snapped)

    resolvers: list[dict[sympy.Symbol, float]] = []
    for circ_idx in range(num_circuits):
        local_rng = np.random.default_rng(int(seed) + 1000 * (circ_idx + 1))
        resolver: dict[sympy.Symbol, float] = {
            sym: clifford_snap_value_for_symbol(sym, target_by_symbol[sym])
            for sym in symbols
        }

        unsnapped_total = int(non_clifford_params_max)
        unsnapped_phase = min(
            int(max_unsnapped_phase),
            int(unsnapped_total),
            len(phase_symbols),
        )
        unsnapped_theta = min(
            int(unsnapped_total) - int(unsnapped_phase),
            len(theta_symbols),
        )

        phase_order = list(phase_symbols)
        theta_order = list(theta_symbols)
        local_rng.shuffle(phase_order)
        local_rng.shuffle(theta_order)
        unsnapped_symbols = phase_order[:unsnapped_phase] + theta_order[:unsnapped_theta]

        if len(unsnapped_symbols) < unsnapped_total:
            remaining = [
                s
                for s in symbols
                if s not in unsnapped_symbols
            ]
            local_rng.shuffle(remaining)
            need = unsnapped_total - len(unsnapped_symbols)
            unsnapped_symbols.extend(remaining[:need])

        for sym in unsnapped_symbols:
            resolver[sym] = _sample_non_clifford_value_for_symbol(sym, local_rng)
        resolvers.append(resolver)

    return resolvers


def _generate_near_clifford_param_sets_legacy_op_budget(
    target_params: dict[sympy.Symbol, float] | dict[str, float],
    symbols: list[sympy.Symbol],
    *,
    num_circuits: int,
    t_max: int,
    circuit: cirq.Circuit,
    min_snap_fraction: float = 0.0,
    seed: int = 0,
) -> list[dict[sympy.Symbol, float]]:
    if num_circuits <= 0:
        raise ValueError(f"num_circuits must be > 0, got {num_circuits}.")
    if t_max < 0:
        raise ValueError(f"t_max must be >= 0, got {t_max}.")
    _ = float(min_snap_fraction)  # Backward-compatible no-op.

    target_by_symbol = _normalize_target_params(target_params, symbols)

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


def generate_near_clifford_param_sets(
    target_params: dict[sympy.Symbol, float] | dict[str, float],
    symbols: list[sympy.Symbol],
    *,
    num_circuits: int,
    t_max: int | None = None,
    non_clifford_params_max: int | None = None,
    target_phase_noncliff_snap_fraction: float | None = None,
    circuit: cirq.Circuit,
    min_snap_fraction: float = 0.0,
    seed: int = 0,
) -> list[dict[sympy.Symbol, float]]:
    """Generate near-Clifford training resolvers.

    Preferred mode:
      - ``non_clifford_params_max``: cap number of unsnapped non-Clifford symbols.
      - ``target_phase_noncliff_snap_fraction``: desired snapped proportion of
        phase-type symbols (``ph_*``), covering T/Tdag and Z^(±3/4)-style blocks.

    Backward-compatible legacy mode:
      - ``t_max`` only: preserves the historical operation-count budget behavior.
    """
    if non_clifford_params_max is None and t_max is None:
        raise ValueError(
            "Either `non_clifford_params_max` (preferred) or legacy `t_max` must be provided."
        )
    if non_clifford_params_max is not None:
        return _generate_near_clifford_param_sets_symbol_budget(
            target_params,
            symbols,
            num_circuits=int(num_circuits),
            non_clifford_params_max=int(non_clifford_params_max),
            target_phase_noncliff_snap_fraction=target_phase_noncliff_snap_fraction,
            seed=int(seed),
        )
    return _generate_near_clifford_param_sets_legacy_op_budget(
        target_params,
        symbols,
        num_circuits=int(num_circuits),
        t_max=int(t_max),
        circuit=circuit,
        min_snap_fraction=float(min_snap_fraction),
        seed=int(seed),
    )


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
    from shot_measurement import run_mitigation

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
    max_print_terms = min(30, len(coeffs_rem))
    ranked_indices = np.argsort(-np.abs(weights)) if len(weights) else np.arange(len(coeffs_rem))
    ranked_indices = ranked_indices[:max_print_terms]

    print(
        f"REM branch coefficients (top {max_print_terms} by |weight|):"
    )
    for k in ranked_indices:
        a_k, b_k = coeffs_rem[k]
        label = term_labels[k] if k < len(term_labels) else "UNKNOWN"
        weight = float(weights[k]) if k < len(weights) else float("nan")
        print(
            f"term[{k:03d}] {label}  weight={weight: .12f}  "
            f"a={float(a_k): .12f}, b={float(b_k): .12f}"
        )

    if coeffs_unmit.size:
        print(
            f"UNMIT branch coefficients (top {max_print_terms} by |weight|):"
        )
        for k in ranked_indices:
            a_k, b_k = coeffs_unmit[k]
            label = term_labels[k] if k < len(term_labels) else "UNKNOWN"
            weight = float(weights[k]) if k < len(weights) else float("nan")
            print(
                f"term[{k:03d}] {label}  weight={weight: .12f}  "
                f"a={float(a_k): .12f}, b={float(b_k): .12f}"
            )

    return out

