from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import cirq
import numpy as np
import sympy

CZ_NORMAL_TAG = "cz_normal"
CZ_HIGH_TAG = "cz_high"
CZ_ONSITE_TAG = "cz_onsite_normal"

PAULI_CHAR_TO_GATE = {"I": None, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

# Heuristic defaults loosely scaled to Google Weber/Sycamore typical medians (~0.1% 1Q RB,
# ~0.9% isolated 2Q XEB); see docs/noise_model.md. Not a certified hardware calibration.
DEFAULT_AMP_DAMP_GAMMA = 0
DEFAULT_PHASE_DAMP_GAMMA = 0
DEFAULT_DEPOL_PROB = 1.5e-2
DEFAULT_LEAKAGE_APPROX_PROB = 1.5e-3
DEFAULT_HIGH_CZ_MULTIPLIER = 1.0


def params_per_layer(num_spatial_orbitals: int) -> int:
    return (num_spatial_orbitals // 2) + ((num_spatial_orbitals - 1) // 2) + num_spatial_orbitals


def ordered_parameter_symbols(num_spatial_orbitals: int, num_layers: int) -> list[sympy.Symbol]:
    symbols = []
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


def cz_tag_for_horizontal_pair(q0: cirq.GridQubit, q1: cirq.GridQubit) -> str:
    if q0.row == q1.row and abs(q0.col - q1.col) == 1:
        return CZ_HIGH_TAG if min(q0.col, q1.col) % 2 == 1 else CZ_NORMAL_TAG
    return CZ_NORMAL_TAG


def tagged_cz(q0: cirq.Qid, q1: cirq.Qid, tag: str) -> cirq.Operation:
    return cirq.CZ(q0, q1).with_tags(tag)


def decompose_fsim_theta_symbolic(
    theta: sympy.Symbol | float, q0: cirq.Qid, q1: cirq.Qid, cz_tag: str
) -> list[list[cirq.Operation]]:
    theta_exponent = theta / sympy.pi
    return [
        [
            cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=0.5).on(q0),
            cirq.PhasedXPowGate(phase_exponent=-0.75, exponent=0.5).on(q1),
        ],
        [tagged_cz(q0, q1, cz_tag)],
        [
            cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=theta_exponent).on(q0),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=theta_exponent).on(q1),
        ],
        [tagged_cz(q0, q1, cz_tag)],
        [
            cirq.PhasedXPowGate(phase_exponent=0.75, exponent=0.5).on(q0),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(q1),
        ],
    ]


def decompose_fsim_phi_symbolic(
    phi: sympy.Symbol | float, q0: cirq.Qid, q1: cirq.Qid, cz_tag: str = CZ_ONSITE_TAG
) -> list[list[cirq.Operation]]:
    t = -phi / sympy.pi
    return [
        [cirq.ZPowGate(exponent=t / 2).on(q0), cirq.ZPowGate(exponent=t / 2).on(q1)],
        [cirq.H(q1)],
        [tagged_cz(q0, q1, cz_tag)],
        [cirq.H(q1)],
        [cirq.ZPowGate(exponent=-t / 2).on(q1)],
        [cirq.H(q1)],
        [tagged_cz(q0, q1, cz_tag)],
        [cirq.H(q1)],
    ]


def _phased_x_pow_rz_rx_rz_layers(
    q0: cirq.Qid,
    q1: cirq.Qid,
    p0: sympy.Expr | float,
    t0: sympy.Expr | float,
    p1: sympy.Expr | float,
    t1: sympy.Expr | float,
) -> list[list[cirq.Operation]]:
    """PhasedXPowGate(p, t) ≡ Z^{-p} X^{t} Z^{p} ≃ Rz(-πp) Rx(πt) Rz(πp) (same unitary up to global phase)."""

    return [
        [
            cirq.rz(-sympy.pi * p0).on(q0),
            cirq.rz(-sympy.pi * p1).on(q1),
        ],
        [
            cirq.rx(sympy.pi * t0).on(q0),
            cirq.rx(sympy.pi * t1).on(q1),
        ],
        [
            cirq.rz(sympy.pi * p0).on(q0),
            cirq.rz(sympy.pi * p1).on(q1),
        ],
    ]


def _hadamard_as_rz_ry(q: cirq.Qid) -> list[list[cirq.Operation]]:
    """H ≃ Rz(π) Ry(π/2) (same unitary up to global phase)."""

    return [[cirq.rz(np.pi).on(q)], [cirq.ry(np.pi / 2).on(q)]]


def decompose_fsim_theta_symbolic_rxryrz(
    theta: sympy.Symbol | float, q0: cirq.Qid, q1: cirq.Qid, cz_tag: str
) -> list[list[cirq.Operation]]:
    """Same unitary as ``decompose_fsim_theta_symbolic`` using ``CZ`` + ``Rz`` / ``Rx`` only."""

    theta_exponent = theta / sympy.pi
    layers: list[list[cirq.Operation]] = []
    layers.extend(_phased_x_pow_rz_rx_rz_layers(q0, q1, -0.25, 0.5, -0.75, 0.5))
    layers.append([tagged_cz(q0, q1, cz_tag)])
    layers.extend(
        _phased_x_pow_rz_rx_rz_layers(
            q0, q1, -0.25, theta_exponent, 0.25, theta_exponent
        )
    )
    layers.append([tagged_cz(q0, q1, cz_tag)])
    layers.extend(_phased_x_pow_rz_rx_rz_layers(q0, q1, 0.75, 0.5, 0.25, 0.5))
    return layers


def decompose_fsim_phi_symbolic_rxryrz(
    phi: sympy.Symbol | float, q0: cirq.Qid, q1: cirq.Qid, cz_tag: str = CZ_ONSITE_TAG
) -> list[list[cirq.Operation]]:
    """Same unitary as ``decompose_fsim_phi_symbolic`` using ``CZ`` + ``Rz`` / ``Ry`` (no ``H`` / ``ZPowGate``)."""

    t = -phi / sympy.pi
    layers: list[list[cirq.Operation]] = [
        [
            cirq.rz(sympy.pi * t / 2).on(q0),
            cirq.rz(sympy.pi * t / 2).on(q1),
        ],
    ]
    layers.extend(_hadamard_as_rz_ry(q1))
    layers.append([tagged_cz(q0, q1, cz_tag)])
    layers.extend(_hadamard_as_rz_ry(q1))
    layers.append([cirq.rz(-sympy.pi * t / 2).on(q1)])
    layers.extend(_hadamard_as_rz_ry(q1))
    layers.append([tagged_cz(q0, q1, cz_tag)])
    layers.extend(_hadamard_as_rz_ry(q1))
    return layers


def append_depth_grouped_blocks(
    circuit: cirq.Circuit, blocks: list[list[list[cirq.Operation]]]
) -> None:
    if not blocks:
        return
    for ops_at_depth in zip(*blocks):
        flat_ops = [op for sublist in ops_at_depth for op in sublist]
        circuit.append(flat_ops, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)


def prepare_decomposed_ansatz_cirq(
    num_spatial_orbitals: int, num_layers: int = 1
) -> tuple[cirq.Circuit, list[cirq.GridQubit]]:
    qubits = [cirq.GridQubit(0, i) for i in range(num_spatial_orbitals)] + [
        cirq.GridQubit(1, i) for i in range(num_spatial_orbitals)
    ]

    circuit = cirq.Circuit()
    p_idx = 0
    circuit.append([cirq.X(qubits[i]) for i in range(1, len(qubits), 2)])

    for layer in range(num_layers):
        even_odd_blocks = []
        for i in range(0, num_spatial_orbitals - 1, 2):
            theta = sympy.Symbol(f"th_{layer}_{p_idx}")
            alpha_tag = cz_tag_for_horizontal_pair(qubits[i], qubits[i + 1])
            beta_tag = cz_tag_for_horizontal_pair(
                qubits[i + num_spatial_orbitals], qubits[i + 1 + num_spatial_orbitals]
            )
            even_odd_blocks.append(
                decompose_fsim_theta_symbolic(theta, qubits[i], qubits[i + 1], alpha_tag)
            )
            even_odd_blocks.append(
                decompose_fsim_theta_symbolic(
                    theta,
                    qubits[i + num_spatial_orbitals],
                    qubits[i + 1 + num_spatial_orbitals],
                    beta_tag,
                )
            )
            p_idx += 1
        append_depth_grouped_blocks(circuit, even_odd_blocks)

        odd_even_blocks = []
        for i in range(1, num_spatial_orbitals - 1, 2):
            theta = sympy.Symbol(f"th_{layer}_{p_idx}")
            alpha_tag = cz_tag_for_horizontal_pair(qubits[i], qubits[i + 1])
            beta_tag = cz_tag_for_horizontal_pair(
                qubits[i + num_spatial_orbitals], qubits[i + 1 + num_spatial_orbitals]
            )
            odd_even_blocks.append(
                decompose_fsim_theta_symbolic(theta, qubits[i], qubits[i + 1], alpha_tag)
            )
            odd_even_blocks.append(
                decompose_fsim_theta_symbolic(
                    theta,
                    qubits[i + num_spatial_orbitals],
                    qubits[i + 1 + num_spatial_orbitals],
                    beta_tag,
                )
            )
            p_idx += 1
        append_depth_grouped_blocks(circuit, odd_even_blocks)

        onsite_blocks = []
        for i in range(num_spatial_orbitals):
            phi = sympy.Symbol(f"ph_{layer}_{p_idx}")
            onsite_blocks.append(
                decompose_fsim_phi_symbolic(
                    phi,
                    qubits[i],
                    qubits[i + num_spatial_orbitals],
                    CZ_ONSITE_TAG,
                )
            )
            p_idx += 1
        append_depth_grouped_blocks(circuit, onsite_blocks)

    return circuit, qubits


def prepare_original_fsim_ansatz_cirq(
    num_spatial_orbitals: int, num_layers: int = 1
) -> tuple[cirq.Circuit, list[cirq.GridQubit]]:
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
            onsite_moments.append(cirq.FSimGate(0, phi).on(qubits[i], qubits[i + num_spatial_orbitals]))
            p_idx += 1
        circuit.append(onsite_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circuit, qubits


def operation_has_tag(operation: cirq.Operation, tag: str) -> bool:
    return tag in getattr(operation, "tags", ())


class LocationAwareDecomposedNoise(cirq.NoiseModel):
    def __init__(
        self,
        amp_damp_gamma: float = DEFAULT_AMP_DAMP_GAMMA,
        phase_damp_gamma: float = DEFAULT_PHASE_DAMP_GAMMA,
        depol_prob: float = DEFAULT_DEPOL_PROB,
        high_cz_multiplier: float = DEFAULT_HIGH_CZ_MULTIPLIER,
        leakage_approx_prob: float = DEFAULT_LEAKAGE_APPROX_PROB,
    ):
        self.amp_damp_gamma = amp_damp_gamma
        self.phase_damp_gamma = phase_damp_gamma
        self.depol_prob = depol_prob
        self.high_cz_multiplier = high_cz_multiplier
        self.leakage_approx_prob = leakage_approx_prob

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
            for q in operation.qubits:
                yield cirq.amplitude_damp(self.amp_damp_gamma / 10.0).on(q)
                yield cirq.phase_damp(self.phase_damp_gamma / 10.0).on(q)
                yield cirq.depolarize(self.depol_prob / 10.0).on(q)
            return

        yield operation


def load_hamiltonian_paths(workspace: Path, h_atom: int, bond_length: float | int) -> tuple[Path, Path, Path]:
    local_folder = workspace / "Pauli_Ham"
    colab_folder = Path("/content/drive/My Drive/Quantum_chemistry/pauli_Ham")
    save_folder = local_folder if local_folder.exists() else colab_folder
    bond_token = f"{bond_length}".rstrip("0").rstrip(".") if isinstance(bond_length, float) else str(bond_length)
    pkl_path = save_folder / f"H{h_atom}_bond_{bond_token}.pkl"
    text_path = save_folder / f"H{h_atom}_bond_{bond_token}_pauli_convention.txt"
    return save_folder, pkl_path, text_path


def pauli_text_to_pauli_sum(path: Path, qubits: list[cirq.Qid]) -> cirq.PauliSum:
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
    workspace: Path, ansatz_qubits: list[cirq.Qid], h_atom: int, bond_length: float | int
) -> cirq.PauliSum:
    save_folder, pkl_path, text_path = load_hamiltonian_paths(workspace, h_atom, bond_length)
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


def scale_noise_params_for_zne(
    noise_scale: float,
    *,
    amp_damp_gamma: float,
    phase_damp_gamma: float,
    depol_prob: float,
    leakage_approx_prob: float,
    high_cz_multiplier: float,
) -> dict[str, float]:
    if noise_scale <= 0:
        raise ValueError(f"noise_scale must be > 0, got {noise_scale}.")

    def clip01(value: float) -> float:
        return float(min(1.0, max(0.0, value)))

    return {
        "amp_damp_gamma": clip01(amp_damp_gamma * noise_scale),
        "phase_damp_gamma": clip01(phase_damp_gamma * noise_scale),
        "depol_prob": clip01(depol_prob * noise_scale),
        "high_cz_multiplier": float(high_cz_multiplier),
        "leakage_approx_prob": clip01(leakage_approx_prob * noise_scale),
    }


def trace_energy_at_noise_scale(
    ansatz_circuit: cirq.Circuit,
    resolver: cirq.ParamResolver,
    qubits: list[cirq.Qid],
    hamiltonian_matrix: np.ndarray,
    *,
    noise_scale: float,
    amp_damp_gamma: float,
    phase_damp_gamma: float,
    depol_prob: float,
    leakage_approx_prob: float,
    high_cz_multiplier: float,
    simulator_seed: int = 1234,
) -> float:
    scaled = scale_noise_params_for_zne(
        noise_scale,
        amp_damp_gamma=amp_damp_gamma,
        phase_damp_gamma=phase_damp_gamma,
        depol_prob=depol_prob,
        leakage_approx_prob=leakage_approx_prob,
        high_cz_multiplier=high_cz_multiplier,
    )
    noise_model = LocationAwareDecomposedNoise(**scaled)
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
    amp_damp_gamma: float = DEFAULT_AMP_DAMP_GAMMA,
    phase_damp_gamma: float = DEFAULT_PHASE_DAMP_GAMMA,
    depol_prob: float = DEFAULT_DEPOL_PROB,
    high_cz_multiplier: float = DEFAULT_HIGH_CZ_MULTIPLIER,
    leakage_approx_prob: float = DEFAULT_LEAKAGE_APPROX_PROB,
) -> dict[str, object]:
    scales = [float(s) for s in noise_scales]
    trace_energies = [
        trace_energy_at_noise_scale(
            ansatz_circuit,
            resolver,
            qubits,
            hamiltonian_matrix,
            noise_scale=scale,
            amp_damp_gamma=amp_damp_gamma,
            phase_damp_gamma=phase_damp_gamma,
            depol_prob=depol_prob,
            leakage_approx_prob=leakage_approx_prob,
            high_cz_multiplier=high_cz_multiplier,
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
    """Generate `num_circuits` near-Clifford resolvers using `t_max`-bounded
    greedy parameter snapping.

    For each training circuit:
      1. Start from the target VQE parameters (no snaps).
      2. Optionally pre-snap a `min_snap_fraction` random subset for diversity.
      3. While `count_non_clifford_ops(...) > t_max`, randomly pick an
         unsnapped symbol and snap it.
      4. Return the resolver dict for this training circuit.
    """
    if num_circuits <= 0:
        raise ValueError(f"num_circuits must be > 0, got {num_circuits}.")
    if t_max < 0:
        raise ValueError(f"t_max must be >= 0, got {t_max}.")
    if not (0.0 <= float(min_snap_fraction) <= 1.0):
        raise ValueError(
            f"min_snap_fraction must be in [0, 1], got {min_snap_fraction}."
        )

    target_by_symbol: dict[sympy.Symbol, float] = {}
    for sym in symbols:
        if sym in target_params:
            target_by_symbol[sym] = float(target_params[sym])
        elif str(sym) in target_params:
            target_by_symbol[sym] = float(target_params[str(sym)])
        else:
            raise KeyError(f"Target parameter missing for symbol {sym!s}.")

    rng = np.random.default_rng(int(seed))
    resolvers: list[dict[sympy.Symbol, float]] = []

    for circ_idx in range(num_circuits):
        local_rng = np.random.default_rng(int(seed) + 1000 * (circ_idx + 1))
        resolver: dict[sympy.Symbol, float] = dict(target_by_symbol)
        snapped: set[sympy.Symbol] = set()

        if min_snap_fraction > 0.0:
            n_min_snap = int(np.ceil(min_snap_fraction * len(symbols)))
            n_min_snap = min(n_min_snap, len(symbols))
            order = list(symbols)
            local_rng.shuffle(order)
            for sym in order[:n_min_snap]:
                resolver[sym] = clifford_snap_value_for_symbol(sym, target_by_symbol[sym])
                snapped.add(sym)

        guard = 0
        max_iterations = 4 * len(symbols) + 8
        while True:
            t_remaining = count_non_clifford_ops(circuit, resolver)
            if t_remaining <= t_max:
                break
            unsnapped = [s for s in symbols if s not in snapped]
            if not unsnapped:
                break
            pick = unsnapped[local_rng.integers(0, len(unsnapped))]
            resolver[pick] = clifford_snap_value_for_symbol(pick, target_by_symbol[pick])
            snapped.add(pick)
            guard += 1
            if guard > max_iterations:
                break

        resolvers.append(resolver)

    if rng is not None:
        _ = rng.random()
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

