"""Combined CDR + Gaussian-Process Error Mitigator for VQE (Design 2).

Unlike the two-stage design (separate per-Pauli affine CDR backbone + a GP on the
residual), here ONE Gaussian Process maps the feature row directly to the ideal
expectation:

    O_mit(theta, P_i) = GP( features(theta, P_i, O_noisy) )   ~= O_ideal

The classic CDR line ``a * O_noisy + b`` is NOT fit separately; it is absorbed
INTO the GP kernel as a linear (DotProduct) term on the ``O_noisy`` feature
column, with a Matern/RBF term on the angle+Pauli features capturing the coherent,
angle-dependent residual on top:

    k = c_lin * k_lin(O_noisy) [* k_obs(pauli)]      # the learned CDR slope/bias
      + c_base * k_base(angle, pauli)                # coherent / angle residual
      + WhiteKernel                                  # shot noise

The GP returns a variance at every prediction, which drives an uncertainty-gated
top-up loop during VQE.

This module is a thin layer on top of the user's EXISTING cirq codebase
(``main_cursor_lib_ML_tog`` and ``shot_measurement_ML_tog``). It does NOT
re-implement circuit construction, simulation, or shot estimation; it binds the
adapter methods to those existing functions. A plain per-Pauli CDR backbone is
still fit, but ONLY as a baseline reference for validation/plots -- it is not part
of the mitigated value.

The public API (config, adapter, Mitigator, validation, VQE loops) mirrors the
two-stage ``gp_mitigator_ML_sep`` module so the two notebooks differ only in the
error-mitigation core.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Keep this experiment folder self-contained: import the LOCAL ``*_ML_tog`` copies
# of the cirq pipeline that live next to this module, regardless of the caller's
# working directory. These are the editable "together" copies for the ML-CDR
# experiments; the June_main originals are left untouched.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from main_cursor_lib_ML_tog import (
    clifford_snap_value_for_symbol,
)
from shot_measurement_ML_tog import (
    _simulate_noiseless_state_for_resolver,
    _simulate_noisy_rho_for_resolver,
    estimate_energy_from_noisy_rho_shots,
    exact_pauli_expectation_from_int_row,
    int_observable_to_pauli_string,
    pauli_sum_to_int_observables,
)

try:  # scikit-learn is the GP backend (the single combined GP).
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        DotProduct,
        Hyperparameter,
        Kernel,
        Matern,
        RBF,
        WhiteKernel,
    )

    _SKLEARN_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - surfaced lazily when GP is used.
    _SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# Section 1 - Global configuration (defaults, not constants)
# ---------------------------------------------------------------------------


@dataclass
class MitigatorConfig:
    # --- Problem / ansatz ---
    n_params: int = 3
    n_qubits: int = 8
    theta_init: "np.ndarray | None" = None  # warm-start center, len == n_params

    # --- Near-Clifford data generation ---
    n_warmstart_circuits: int = 30
    # int, or a tuple/list of ints sampled uniformly per circuit (mixed designs,
    # e.g. (1, 2): mostly-1-non-Clifford budget with some 2-D face coverage).
    n_nonclifford_gates: "int | tuple[int, ...]" = 2
    warmstart_spread: float = 0.3  # angle spread (radians) around theta_init
    clifford_snap_step: float = float(np.pi) / 2.0
    shots: int = 10000
    rng_seed: int = 0
    # Grid-diverse snapping: snapped coordinates land on a RANDOM Clifford node
    # within +/- clifford_node_hops grid steps of theta (0 = always the nearest
    # node, the legacy behavior). Same cost per circuit, but the training set
    # then covers axis lines through MANY grid nodes, which pins the
    # cross-parameter interaction structure at grid resolution -- the missing
    # information when only 1 gate per circuit is non-Clifford.
    clifford_node_hops: int = 1
    # Fully-Clifford anchor circuits sampled over a wider grid (anchor_node_hops
    # steps). Their rows are tagged "anchor" and are never pruned, so the GP
    # stays pinned globally even after large theta moves. 0 = off.
    n_clifford_anchor_circuits: int = 8
    anchor_node_hops: int = 2

    # --- Observables (real list comes from the Hamiltonian) ---
    n_observables: int = 100  # informational only

    # --- Feature encoding ---
    max_harmonic: int = 1
    include_noisy_feature_in_gp: bool = True
    pauli_onehot: bool = True
    pauli_summaries: bool = True
    # Theta-location interaction features. Static per-parameter descriptors
    # (gate depth, downstream noise) are invisible to the kernel for a fixed
    # circuit (constant columns), so they are encoded as ROW-VARYING
    # interactions: sin(theta_j) * depth_frac_j, sin(theta_j) * downstream_2q_frac_j,
    # and sin(theta_j) * lightcone_overlap(j, P). The last one couples the angle
    # block to the observable block so information pools across Paulis.
    include_location_features: bool = True
    # Filled automatically from the adapter (see CirqBackendAdapter
    # .param_location_descriptors); shape (n_params, 2) = [depth_frac, down_frac].
    param_locations: "np.ndarray | None" = None
    # (n_params, n_qubits) 0/1 forward-lightcone masks per parameter.
    param_lightcones: "np.ndarray | None" = None

    # --- Baseline per-Pauli CDR backbone (reference only; NOT part of the
    #     combined mitigated value, which is the single GP below). Kept so the
    #     validation/plots can show "plain CDR" alongside the combined method. ---
    affine_regularization: float = 0.0
    refit_affine_on_topup: bool = True

    # --- Combined single-GP kernel (CDR absorbed in the kernel) ---
    kernel_type: str = "matern"  # "matern" or "rbf" for the coherent-residual term
    matern_nu: float = 2.5
    use_ard: bool = True
    # ARD on the (4 * n_qubits) Pauli one-hot dims is a lot of hyperparameters
    # for a few hundred rows; False = one shared length scale for the one-hot
    # block (summaries keep their own scales when use_ard=True).
    pauli_ard: bool = False
    # use_product_kernel here means: multiply the linear CDR term k_lin(o_noisy) by
    # a per-observable kernel k_obs(pauli) -> a learned, observable-dependent CDR
    # slope (True), instead of a single shared slope (False).
    use_product_kernel: bool = True
    noise_variance_init: float = 1e-3
    normalize_targets: bool = True
    gp_n_restarts: int = 2  # marginal-likelihood restarts (not in spec; small default)
    # --- Measurement-reliance controls -----------------------------------
    # prior_mean_mode:
    #   "backbone" -> the GP is trained on RESIDUALS y = o_ideal - (a_P*o_noisy + b_P)
    #                 around the per-Pauli affine CDR backbone, and the backbone is
    #                 added back at prediction. The mitigated value then ALWAYS
    #                 carries a_P * O_noisy, so the device signal contributes to
    #                 both the value and the analytic gradient, and an off-cloud GP
    #                 can only be wrong at the (small) residual scale.
    #   "zero"     -> legacy combined design: GP maps features -> O_ideal directly.
    prior_mean_mode: str = "backbone"
    # Upper ConstantKernel bound on the PURE-ANGLE additive term. Bounding it
    # (e.g. 0.1 in normalized-target units) stops the angle-only path from
    # absorbing the whole signal and bypassing the measurement, which is the
    # classical-model degeneracy seen with 1 non-Clifford gate per circuit.
    angle_term_amplitude_bound: float = 0.1
    # Upper bound on DotProduct sigma_0 (the CDR bias channel). Keeping it small
    # stops the o_noisy-independent offset from soaking up angle structure.
    dotproduct_sigma0_bound: float = 1.0
    # Heteroscedastic per-row noise: alpha_row = o_noisy_var estimated from the
    # binomial shot noise of each Pauli (~ (1 - o^2)/shots) instead of relying on
    # one shared WhiteKernel level. The WhiteKernel stays as a learned floor.
    heteroscedastic_alpha: bool = True
    # Report EPISTEMIC predictive std only: subtract the learned WhiteKernel
    # (aleatoric shot-noise) level from the predictive variance. Without this the
    # std has a shot-noise floor that no amount of training data can push below,
    # so an uncertainty gate can become permanently unreachable.
    epistemic_std: bool = True
    # Cap on rows actually used to FIT the exact GP (the baseline backbone still
    # uses all rows). Exact-GP hyperparameter optimization with ARD builds an (N, N, n_hyper)
    # gradient tensor, so N must stay bounded; rows are randomly subsampled down to
    # this many before fitting. ~1000-2000 is plenty for an exact GP.
    max_gp_train_points: int = 1500

    # --- Top-up controller ---
    uncertainty_threshold: float = 0.05
    topup_batch_size: int = 12
    topup_radius: "float | None" = None  # None -> tie to optimizer_step_max
    drop_faraway_points: bool = True
    max_gp_points: int = 5000
    # (1) Trust-region trigger: top up whenever theta has moved more than
    #     ``topup_move_fraction * effective_topup_radius`` from the center of the
    #     last top-up, so the surrogate stays valid as the optimizer travels --
    #     destination-agnostic, fires even when the (overconfident) uncertainty
    #     gate stays silent off the near-Clifford cloud.
    topup_on_move: bool = True
    topup_move_fraction: float = 1.0
    # (2) Mandatory periodic top-up every ``topup_every`` iterations (0 = off).
    #     Cheap insurance against a frozen, overconfident GP trapping the optimizer.
    topup_every: int = 0
    # (3) Top-up-until-satisfied: after a top-up fires, keep adding local batches
    #     while the quadrature energy-scale std stays above ``uncertainty_threshold``,
    #     up to ``max_topup_retries`` extra batches (0 = off -> a single top-up).
    #     Bounds the shot cost while forcing the surrogate to be trustworthy at
    #     the current theta before the gradient/step are computed.
    max_topup_retries: int = 0
    # Prequential validation: every top-up batch has EXACT o_ideal values, so
    # BEFORE its rows are added the current GP is evaluated on them and the
    # mean absolute per-circuit ENERGY error is recorded. That is a real,
    # unbiased local error estimate immune to GP overconfidence; the retry loop
    # is driven by max(prequential_error, weighted_std) > uncertainty_threshold.
    prequential_topup_check: bool = True
    # If the prequential error is STILL above threshold after all retries, use
    # the backbone-only (plain per-Pauli CDR) prediction for this iteration
    # instead of trusting a locally-invalid GP mean.
    fallback_to_backbone_on_mistrust: bool = True
    # Move-scaled top-ups: when the trust-region (move) trigger fires, take
    # ceil(moved / effective_topup_radius) batches (capped by this value) so a
    # large theta jump is re-covered before the next gradient step. 1 = legacy.
    max_move_topup_batches: int = 3
    # (4) Convergence validation: when |grad| < ``convergence_grad_tol`` fire one
    #     REAL near-Clifford top-up at the current theta and re-check before
    #     declaring convergence; only stop if the energy then moves less than
    #     ``convergence_energy_tol`` and the gradient is still below tolerance.
    #     ``convergence_grad_tol = 0`` disables early stopping (run the full
    #     iteration budget -- the legacy behavior).
    convergence_grad_tol: float = 0.0
    convergence_energy_tol: float = 1e-3

    # --- Optimizer ---
    optimizer_step_max: float = 0.2
    max_vqe_iterations: int = 100

    # --- Validation ---
    holdout_fraction: float = 0.2
    target_energy_accuracy: float = 0.05

    # --- Branch selection (this repo exposes both an unmitigated and a
    #     readout-error-mitigated noisy expectation value per term). The
    #     notebook's live objective is ``cdr_rem_corrected`` so we default to the
    #     REM branch; flip to fit/predict on the raw unmitigated branch. ---
    use_rem_branch: bool = True

    def effective_topup_radius(self) -> float:
        return float(self.topup_radius if self.topup_radius is not None else self.optimizer_step_max)


# ---------------------------------------------------------------------------
# Section 4 - Feature construction
# ---------------------------------------------------------------------------


def encode_angles(theta: np.ndarray, max_harmonic: int) -> np.ndarray:
    """Fourier angle block: ``[cos(k t_j), sin(k t_j) for k in 1..max_harmonic]``.

    Encoding sinusoids (never raw angles) gives the GP the correct periodic
    inductive bias; an ordinary RBF/Matern kernel on these features is then
    automatically periodic in theta. Length = ``n_params * 2 * max_harmonic``.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    if max_harmonic < 1:
        raise ValueError(f"max_harmonic must be >= 1, got {max_harmonic}.")
    feats: list[float] = []
    for t in theta:
        for k in range(1, int(max_harmonic) + 1):
            feats.append(float(np.cos(k * t)))
            feats.append(float(np.sin(k * t)))
    return np.asarray(feats, dtype=float)


def encode_pauli(
    pauli: str, n_qubits: int, onehot: bool = True, summaries: bool = True
) -> np.ndarray:
    """Pauli block from a per-qubit ``{I,X,Y,Z}`` label string.

    - one-hot: per qubit a 4-dim indicator over [I, X, Y, Z]  (length 4 * n_qubits)
    - summaries: ``[weight, n_XY, n_Z]`` where weight = #non-identity factors.
    """
    label = str(pauli)
    if len(label) != n_qubits:
        raise ValueError(
            f"Pauli label '{label}' has length {len(label)} but n_qubits={n_qubits}."
        )
    idx = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    feats: list[float] = []
    if onehot:
        for ch in label:
            block = [0.0, 0.0, 0.0, 0.0]
            block[idx[ch]] = 1.0
            feats.extend(block)
    if summaries:
        weight = float(sum(1 for ch in label if ch != "I"))
        n_xy = float(sum(1 for ch in label if ch in ("X", "Y")))
        n_z = float(sum(1 for ch in label if ch == "Z"))
        feats.extend([weight, n_xy, n_z])
    return np.asarray(feats, dtype=float)


def _location_features_enabled(config: MitigatorConfig) -> bool:
    return bool(
        config.include_location_features
        and config.param_locations is not None
        and config.param_lightcones is not None
    )


# Number of location-interaction features per parameter:
#   sin(theta_j) * depth_frac_j, sin(theta_j) * downstream_2q_frac_j,
#   sin(theta_j) * lightcone_overlap(j, P).
_N_LOC_PER_PARAM = 3


def _pauli_support_mask(pauli: str) -> np.ndarray:
    return np.asarray([1.0 if ch != "I" else 0.0 for ch in str(pauli)], dtype=float)


def lightcone_overlap(config: MitigatorConfig, pauli: str) -> np.ndarray:
    """Per-parameter fraction of the Pauli's non-identity support that lies in
    the forward lightcone of the gate driven by that parameter. Shape (n_params,).
    """
    lc = np.asarray(config.param_lightcones, dtype=float)  # (n_params, n_qubits)
    support = _pauli_support_mask(pauli)
    w = float(support.sum())
    if w <= 0.0:
        return np.zeros(lc.shape[0], dtype=float)
    return (lc @ support) / w


def encode_location(theta: np.ndarray, pauli: str, config: MitigatorConfig) -> np.ndarray:
    """Row-varying theta-location interaction features.

    Static per-parameter descriptors (depth, downstream noise) are constant
    columns for a fixed circuit and therefore invisible to the kernel, so they
    are encoded as interactions with ``sin(theta_j)`` -- and with the Pauli's
    lightcone overlap, which couples the angle block to the observable block.
    Length = ``_N_LOC_PER_PARAM * n_params``.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    locs = np.asarray(config.param_locations, dtype=float)  # (n_params, 2)
    ov = lightcone_overlap(config, pauli)  # (n_params,)
    feats: list[float] = []
    for j, t in enumerate(theta):
        s = float(np.sin(t))
        feats.extend([s * float(locs[j, 0]), s * float(locs[j, 1]), s * float(ov[j])])
    return np.asarray(feats, dtype=float)


def feature_index_map(config: MitigatorConfig) -> dict[str, list[int]]:
    """Record which feature indices belong to which block.

    Blocks: ``angle`` (Fourier), ``noisy`` (optional O_noisy scalar),
    ``location`` (theta-location interactions), ``pauli``. The pauli block is
    further split into one-hot and summary indices so the kernel can share a
    single length scale over the one-hot dims (``pauli_ard=False``).
    """
    n_angle = int(config.n_params) * 2 * int(config.max_harmonic)
    n_noisy = 1 if config.include_noisy_feature_in_gp else 0
    n_loc = _N_LOC_PER_PARAM * int(config.n_params) if _location_features_enabled(config) else 0
    n_onehot = 4 * int(config.n_qubits) if config.pauli_onehot else 0
    n_summary = 3 if config.pauli_summaries else 0
    angle_idx = list(range(0, n_angle))
    noisy_idx = list(range(n_angle, n_angle + n_noisy))
    loc_start = n_angle + n_noisy
    loc_idx = list(range(loc_start, loc_start + n_loc))
    pauli_start = loc_start + n_loc
    onehot_idx = list(range(pauli_start, pauli_start + n_onehot))
    summary_idx = list(range(pauli_start + n_onehot, pauli_start + n_onehot + n_summary))
    return {
        "angle": angle_idx,
        "noisy": noisy_idx,
        "location": loc_idx,
        "pauli": onehot_idx + summary_idx,
        "pauli_onehot": onehot_idx,
        "pauli_summaries": summary_idx,
        "continuous": angle_idx + noisy_idx,  # angle + noisy for the product kernel
    }


def build_feature_row(row: dict, config: MitigatorConfig) -> np.ndarray:
    """Build the GP input vector for one row:
    ``[angle | (o_noisy) | (location) | pauli]``."""
    parts = [encode_angles(row["theta"], config.max_harmonic)]
    if config.include_noisy_feature_in_gp:
        parts.append(np.asarray([float(row["o_noisy"])], dtype=float))
    if _location_features_enabled(config):
        parts.append(encode_location(row["theta"], row["pauli"], config))
    parts.append(
        encode_pauli(
            row["pauli"],
            config.n_qubits,
            onehot=config.pauli_onehot,
            summaries=config.pauli_summaries,
        )
    )
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Section 2 - Integration adapter (binds spec methods to existing repo code)
# ---------------------------------------------------------------------------


class CirqBackendAdapter:
    """Binds the spec's ``QuantumBackendAdapter`` to the existing cirq pipeline.

    The ansatz is a fixed symbolic ``cirq.Circuit``; "building" it for a given
    ``theta`` is just constructing a resolver mapping symbols -> floats.
    """

    def __init__(
        self,
        *,
        circuit,
        qubits,
        symbols,
        pauli_sum,
        base_noise_cfg: dict,
        shot_cfg: dict,
        readout_cal: dict | None = None,
        simulator_seed: int = 1234,
        use_rem_branch: bool = True,
    ) -> None:
        self.circuit = circuit
        self.qubits = list(qubits)
        self.symbols = list(symbols)
        self.pauli_sum = pauli_sum
        self.base_noise_cfg = dict(base_noise_cfg)
        self.shot_cfg = dict(shot_cfg)
        self.readout_cal = dict(readout_cal or {})
        self.simulator_seed = int(simulator_seed)
        self.use_rem_branch = bool(use_rem_branch)

        # The two-stage CDR + GP mitigator measures every circuit with the OGM
        # (operator-grouping) scheme. The per-Pauli (``direct_pauli``) fallback is
        # intentionally disabled: if the OGM basis file or the shadowgrouping
        # backend is unavailable we raise instead of silently degrading.
        self._require_ogm_available()

        observables_int, weights, offset = pauli_sum_to_int_observables(
            pauli_sum, self.qubits
        )
        self.observables_int = observables_int
        self.weights = np.asarray(weights, dtype=float)
        self.offset = float(offset)
        self.obs_labels = [
            int_observable_to_pauli_string(row) for row in observables_int
        ]
        self.coeff_by_pauli = {
            label: float(w) for label, w in zip(self.obs_labels, self.weights)
        }

        # Per-parameter location descriptors (depth fraction, downstream noisy-2q
        # fraction) and forward-lightcone qubit masks, computed once from the
        # symbolic circuit. Used by the theta-location interaction features.
        self.param_locations, self.param_lightcones = self._compute_param_locations()

        # --- Shot accounting -------------------------------------------------
        # EVERY quantum execution on this backend flows through ``run_noisy``
        # (warm start, per-iteration energy, parameter-shift gradient and
        # top-ups all call it), so counting those calls gives the exact hardware
        # cost. Each call consumes ``shots`` shots (the OGM scheme spreads that
        # budget across measurement groups internally). ``ideal_energy`` /
        # ``simulate_ideal`` are noiseless statevector evaluations and cost ZERO
        # shots, so they are deliberately not counted.
        self.n_circuit_evals = 0
        self.total_shots = 0

    def reset_shot_counter(self) -> None:
        """Zero the run_noisy call / shot counters (e.g. before a fresh run)."""
        self.n_circuit_evals = 0
        self.total_shots = 0

    def shot_report(self) -> dict:
        """Current empirical shot usage: ``run_noisy`` calls and total shots."""
        return {
            "circuit_evaluations": int(self.n_circuit_evals),
            "total_shots": int(self.total_shots),
        }

    # -- theta-location descriptors (classical, computed once) ------------
    def _compute_param_locations(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-parameter gate-location descriptors from the symbolic circuit.

        Returns ``(locations, lightcones)``:
          * ``locations``  shape (n_params, 2): ``[depth_frac, downstream_2q_frac]``
            of the (first) gate driven by each symbol -- depth position in the
            circuit and the fraction of noisy 2-qubit gates that come AFTER it
            (later rotations are less decohered).
          * ``lightcones`` shape (n_params, n_qubits): 0/1 forward-lightcone
            masks (qubits reachable from the driven gate through subsequent ops).
        """
        import cirq

        ops = list(self.circuit.all_operations())
        n_ops = max(1, len(ops))
        two_q_positions = [i for i, op in enumerate(ops) if len(op.qubits) == 2]
        n_two_q = max(1, len(two_q_positions))
        qubit_index = {q: i for i, q in enumerate(self.qubits)}
        n_qubits = len(self.qubits)

        locations = np.zeros((len(self.symbols), 2), dtype=float)
        lightcones = np.zeros((len(self.symbols), n_qubits), dtype=float)
        for j, sym in enumerate(self.symbols):
            name = str(sym)
            driven = [
                i for i, op in enumerate(ops) if name in cirq.parameter_names(op)
            ]
            if not driven:
                continue
            first = driven[0]
            depth_frac = float(first) / float(n_ops)
            down_frac = float(sum(1 for p in two_q_positions if p > first)) / float(
                n_two_q
            )
            locations[j] = (depth_frac, down_frac)
            # Forward lightcone: start from the union of qubits of ALL gates the
            # symbol drives, then sweep subsequent ops that touch the set.
            cone: set = set()
            for i in driven:
                cone.update(ops[i].qubits)
            start = min(driven)
            for op in ops[start:]:
                if cone.intersection(op.qubits):
                    cone.update(op.qubits)
            for q in cone:
                if q in qubit_index:
                    lightcones[j, qubit_index[q]] = 1.0
        return locations, lightcones

    # -- OGM availability (no silent per-Pauli fallback) ------------------
    def _require_ogm_available(self) -> None:
        """Raise unless the OGM measurement backend is fully available.

        The mitigator does not fall back to per-Pauli (``direct_pauli``)
        measurement; OGM is mandatory so the assessed performance reflects the
        real grouped-measurement layout.
        """
        scheme = str(self.shot_cfg.get("measurement_scheme", "ogm")).lower()
        if scheme == "direct_pauli":
            raise ValueError(
                "The two-stage CDR+GP mitigator requires the OGM measurement "
                "scheme; the per-Pauli 'direct_pauli' fallback is disabled. "
                "Set shot_cfg['measurement_scheme'] = 'ogm'."
            )
        ogm_file = self.shot_cfg.get("ogm_file")
        if not ogm_file or not Path(str(ogm_file)).is_file():
            raise FileNotFoundError(
                f"OGM measurement-basis file is not available: {ogm_file!r}. "
                "Generate the OGM basis for this molecule/bond "
                "(June_main/OGM_measurement_basis/OGM_<MOLECULE>_bond_<bond>.txt) "
                "before running the two-stage CDR+GP mitigator."
            )
        sg_root = self.shot_cfg.get("shadowgrouping_root")
        if not sg_root or not Path(str(sg_root)).is_dir():
            raise FileNotFoundError(
                f"shadowgrouping_root is not available: {sg_root!r}. The OGM "
                "scheme needs the shadowgrouping package; set "
                "shot_cfg['shadowgrouping_root'] to its location."
            )

    # -- ansatz / parameter helpers --------------------------------------
    def resolver_from_theta(self, theta: np.ndarray) -> dict:
        theta = np.asarray(theta, dtype=float).ravel()
        return {sym: float(theta[j]) for j, sym in enumerate(self.symbols)}

    def theta_from_resolver(self, resolver: dict) -> np.ndarray:
        return np.asarray([float(resolver[sym]) for sym in self.symbols], dtype=float)

    def build_ansatz(self, theta: np.ndarray) -> dict:
        """Spec ``build_ansatz``: returns the resolver for the fixed circuit."""
        return self.resolver_from_theta(theta)

    # -- near-Clifford generation (LOCAL around theta) -------------------
    def _snap_step_for_symbol(self, sym, snap_step: float) -> float:
        """Clifford grid step for this symbol (pi for ph_*, snap_step otherwise)."""
        return float(np.pi) if str(sym).startswith("ph_") else float(snap_step)

    def generate_near_clifford(
        self,
        theta: np.ndarray,
        n_circuits: int,
        n_nonclifford: "int | tuple[int, ...] | list[int]",
        snap_step: float,
        spread: float,
        seed: int,
        node_hops: int = 0,
    ) -> list[dict]:
        """Local near-Clifford resolvers around ``theta``.

        Most parameters are snapped to the Clifford grid; a random subset of size
        ``n_nonclifford`` is left non-Clifford by jittering within ``spread``
        radians of ``theta``. Locality matters because the GP uses the angles as
        features (warm-start spread and top-up radius both flow here).

        * ``n_nonclifford`` may be an int, or a tuple/list of ints cycled
          round-robin over the circuits (balanced mixed designs, e.g. ``(1, 2)``
          gives exactly half the circuits 1 and half 2 non-Clifford gates; which
          parameters are non-Clifford stays random).
        * ``node_hops > 0``: each snapped coordinate lands on a RANDOM Clifford
          node within ``+/- node_hops`` grid steps of the nearest one, instead of
          always the nearest. Cost per circuit is unchanged, but the training set
          then covers axis lines through many grid nodes, which supplies the
          cross-parameter interaction information a single-node axis design
          fundamentally lacks.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        n_params = len(self.symbols)
        if isinstance(n_nonclifford, (tuple, list)):
            nnc_choices = [int(max(0, min(int(v), n_params))) for v in n_nonclifford]
            if not nnc_choices:
                nnc_choices = [0]
        else:
            nnc_choices = [int(max(0, min(int(n_nonclifford), n_params)))]
        hops = int(max(0, node_hops))
        rng = np.random.default_rng(int(seed))
        resolvers: list[dict] = []
        for i_circ in range(int(n_circuits)):
            n_nc = int(nnc_choices[i_circ % len(nnc_choices)])
            if n_nc > 0:
                nc_idx = set(
                    int(j)
                    for j in rng.choice(n_params, size=n_nc, replace=False).tolist()
                )
            else:
                nc_idx = set()
            resolver: dict = {}
            for j, sym in enumerate(self.symbols):
                tj = float(theta[j])
                if j in nc_idx:
                    resolver[sym] = tj + float(rng.uniform(-spread, spread))
                else:
                    base = _snap_to_clifford(sym, tj, snap_step)
                    if hops > 0:
                        step = self._snap_step_for_symbol(sym, snap_step)
                        base += float(rng.integers(-hops, hops + 1)) * step
                    resolver[sym] = base
            resolvers.append(resolver)
        return resolvers

    def generate_clifford_anchors(
        self,
        theta: np.ndarray,
        n_circuits: int,
        snap_step: float,
        seed: int,
        node_hops: int = 2,
    ) -> list[dict]:
        """Fully-Clifford resolvers on random grid nodes within ``node_hops``
        steps of ``theta``. These are the cheapest possible training circuits and
        act as never-pruned global anchors that keep the GP pinned after large
        theta moves.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        hops = int(max(1, node_hops))
        rng = np.random.default_rng(int(seed))
        resolvers: list[dict] = []
        for _ in range(int(n_circuits)):
            resolver: dict = {}
            for j, sym in enumerate(self.symbols):
                base = _snap_to_clifford(sym, float(theta[j]), snap_step)
                step = self._snap_step_for_symbol(sym, snap_step)
                resolver[sym] = base + float(rng.integers(-hops, hops + 1)) * step
            resolvers.append(resolver)
        return resolvers

    # -- simulation (classical exact) ------------------------------------
    def simulate_ideal(self, resolver: dict) -> dict:
        state = _simulate_noiseless_state_for_resolver(
            self.circuit, resolver, self.qubits, simulator_seed=self.simulator_seed
        )
        return {
            label: float(
                exact_pauli_expectation_from_int_row(state, obs_row, self.qubits)
            )
            for label, obs_row in zip(self.obs_labels, self.observables_int)
        }

    # -- noisy execution (shots) -----------------------------------------
    def run_noisy(self, resolver: dict, shots: int, sampling_seed: int) -> dict:
        # Shot accounting: one physical circuit execution at ``shots`` shots.
        self.n_circuit_evals += 1
        self.total_shots += int(shots)
        rho = _simulate_noisy_rho_for_resolver(
            self.circuit,
            resolver,
            self.qubits,
            self.base_noise_cfg,
            simulator_seed=self.simulator_seed,
        )
        est = estimate_energy_from_noisy_rho_shots(
            rho,
            self.pauli_sum,
            self.qubits,
            num_shots=int(shots),
            measurement_scheme=str(self.shot_cfg.get("measurement_scheme", "direct_pauli")),
            p_0_success=self.readout_cal.get("p_0_success"),
            p_1_success=self.readout_cal.get("p_1_success"),
            apply_rem=True,
            apply_readout_noise=bool(self.shot_cfg.get("apply_readout_noise", True)),
            sampling_seed=int(sampling_seed),
            epsilon=float(self.shot_cfg.get("epsilon", 0.1)),
            ogm_file=self.shot_cfg.get("ogm_file"),
            shadowgrouping_root=self.shot_cfg.get("shadowgrouping_root"),
            return_per_term=True,
        )
        per_term_unmit = np.asarray(est["per_term_unmitigated"], dtype=float).ravel()
        per_term_rem = np.asarray(est["per_term_rem"], dtype=float).ravel()
        per_term_n = np.asarray(
            est.get("per_term_n_samples", np.full(len(self.obs_labels), int(shots))),
            dtype=int,
        ).ravel()
        key = "per_term_rem" if self.use_rem_branch else "per_term_unmitigated"
        per_term = per_term_rem if key == "per_term_rem" else per_term_unmit
        primary = {label: float(per_term[k]) for k, label in enumerate(self.obs_labels)}
        unmit_by_pauli = {
            label: float(per_term_unmit[k]) for k, label in enumerate(self.obs_labels)
        }
        rem_by_pauli = {
            label: float(per_term_rem[k]) for k, label in enumerate(self.obs_labels)
        }
        n_samples_by_pauli = {
            label: int(per_term_n[k]) for k, label in enumerate(self.obs_labels)
        }
        return {
            "primary": primary,
            "unmit_by_pauli": unmit_by_pauli,
            "rem_by_pauli": rem_by_pauli,
            "n_samples_by_pauli": n_samples_by_pauli,
            "energy_unmitigated": float(est["energy_unmitigated"]),
            "energy_rem": float(est["energy_rem"]),
        }

    # -- row collection (one row per (circuit, observable)) ---------------
    def collect_rows(
        self, resolvers: list[dict], *, shots: int, seed_base: int, anchor: bool = False
    ) -> list[dict]:
        rows: list[dict] = []
        for i, resolver in enumerate(resolvers):
            theta = self.theta_from_resolver(resolver)
            ideal = self.simulate_ideal(resolver)
            noisy = self.run_noisy(resolver, shots=shots, sampling_seed=seed_base + i)
            measured = noisy["primary"]
            n_samples = noisy.get("n_samples_by_pauli", {})
            for label in self.obs_labels:
                rows.append(
                    {
                        "theta": theta,
                        "pauli": label,
                        "o_noisy": float(measured[label]),
                        "o_ideal": float(ideal[label]),
                        "n_samples": int(n_samples.get(label, shots)),
                        "anchor": bool(anchor),
                    }
                )
        return rows

    # -- energy assembly --------------------------------------------------
    def energy_from_values(self, value_by_pauli: dict) -> float:
        e = self.offset
        for label, w in zip(self.obs_labels, self.weights):
            e += float(w) * float(value_by_pauli[label])
        return float(e)

    def ideal_energy(self, theta: np.ndarray) -> float:
        return self.energy_from_values(
            self.simulate_ideal(self.resolver_from_theta(theta))
        )


def _snap_to_clifford(symbol, value: float, snap_step: float) -> float:
    """Snap ``value`` to the nearest Clifford-equivalent angle for ``symbol``.

    Uses the repo's th_*/ph_* aware rule when possible; otherwise falls back to
    the nearest multiple of ``snap_step``.
    """
    try:
        return float(clifford_snap_value_for_symbol(symbol, float(value)))
    except Exception:
        step = float(snap_step)
        return float(round(float(value) / step) * step)


# ---------------------------------------------------------------------------
# Section 5 - Baseline per-Pauli CDR backbone (reference only)
# ---------------------------------------------------------------------------


class CDRBackbone:
    """Per-observable affine CDR fit: ``O_ideal ~= a_i * O_noisy + b_i``.

    In this COMBINED design the CDR line is absorbed inside the single GP kernel,
    so this standalone backbone is NOT used to form the mitigated value. It is fit
    purely as a "plain CDR" baseline so the validation tables and the VQE plots can
    show classic CDR next to the combined GP. Same per-Pauli least-squares the repo
    performs in ``train_cf_models_per_pauli``.
    """

    def __init__(self, affine_regularization: float = 0.0) -> None:
        self.affine_regularization = float(affine_regularization)
        self.coeffs: dict[str, tuple[float, float]] = {}

    def fit(self, rows: list[dict]) -> None:
        by_pauli: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            by_pauli.setdefault(row["pauli"], []).append(
                (float(row["o_noisy"]), float(row["o_ideal"]))
            )
        self.coeffs = {}
        lam = self.affine_regularization
        for pauli, pairs in by_pauli.items():
            x = np.asarray([p[0] for p in pairs], dtype=float)
            y = np.asarray([p[1] for p in pairs], dtype=float)
            if len(x) >= 2 and float(np.std(x)) > 0.0:
                # Ridge on the slope only (penalize a, never the offset b).
                a_design = np.stack([x, np.ones_like(x)], axis=1)
                gram = a_design.T @ a_design
                gram[0, 0] += lam
                rhs = a_design.T @ y
                a, b = np.linalg.solve(gram, rhs)
            else:
                a, b = 1.0, float(np.mean(y - x))
            self.coeffs[pauli] = (float(a), float(b))

    def apply(self, pauli: str, o_noisy: float) -> float:
        a, b = self.coeffs.get(pauli, (1.0, 0.0))
        return float(a) * float(o_noisy) + float(b)

    def residual(self, row: dict) -> float:
        return float(row["o_ideal"]) - self.apply(row["pauli"], row["o_noisy"])


# ---------------------------------------------------------------------------
# Section 6 - Combined single GP (CDR absorbed in the kernel)
# ---------------------------------------------------------------------------


if _SKLEARN_AVAILABLE:

    class _SubsetKernel(Kernel):
        """Apply an inner kernel to a fixed subset of feature columns.

        Lets us form the product kernel ``k_angle(continuous) * k_obs(pauli)`` with
        scikit-learn, which otherwise applies every kernel to all columns.
        Hyperparameters are delegated to the inner kernel.
        """

        def __init__(self, kernel, indices):
            # Keep the exact objects passed in (no copying): sklearn.clone checks
            # parameter identity after re-instantiating from get_params.
            self.kernel = kernel
            self.indices = indices

        def get_params(self, deep=True):
            params = dict(kernel=self.kernel, indices=self.indices)
            if deep:
                for key, val in self.kernel.get_params().items():
                    params["kernel__" + key] = val
            return params

        @property
        def hyperparameters(self):
            out = []
            for hp in self.kernel.hyperparameters:
                out.append(
                    Hyperparameter(
                        "kernel__" + hp.name,
                        hp.value_type,
                        hp.bounds,
                        hp.n_elements,
                    )
                )
            return out

        @property
        def theta(self):
            return self.kernel.theta

        @theta.setter
        def theta(self, theta):
            self.kernel.theta = theta

        @property
        def bounds(self):
            return self.kernel.bounds

        def __call__(self, X, Y=None, eval_gradient=False):
            Xs = np.asarray(X)[:, self.indices]
            Ys = None if Y is None else np.asarray(Y)[:, self.indices]
            return self.kernel(Xs, Ys, eval_gradient=eval_gradient)

        def diag(self, X):
            return self.kernel.diag(np.asarray(X)[:, self.indices])

        def is_stationary(self):
            return self.kernel.is_stationary()


def _make_base_kernel(n_dims: int, config: MitigatorConfig, *, force_scalar: bool = False):
    use_ard = bool(config.use_ard) and not force_scalar
    length_scale = np.ones(n_dims) if use_ard else 1.0
    if config.kernel_type.lower() == "rbf":
        return RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3))
    return Matern(
        length_scale=length_scale,
        length_scale_bounds=(1e-3, 1e3),
        nu=float(config.matern_nu),
    )


def _sum_white_noise_level(kernel) -> float:
    """Total WhiteKernel noise level of a fitted composite kernel.

    White terms only appear as top-level summands in ``build_kernel``, so a
    recursive walk over ``Sum`` nodes suffices (a White inside a ``Product``
    would not contribute a clean diagonal anyway).
    """
    from sklearn.gaussian_process.kernels import Sum

    if isinstance(kernel, Sum):
        return _sum_white_noise_level(kernel.k1) + _sum_white_noise_level(kernel.k2)
    if isinstance(kernel, WhiteKernel):
        return float(kernel.noise_level)
    return 0.0


def _make_pauli_kernel(feature_index_map_: dict, config: MitigatorConfig):
    """Kernel over the Pauli block.

    ``pauli_ard=True``  -> per-dimension length scales over all Pauli dims (legacy).
    ``pauli_ard=False`` -> ONE shared length scale over the (4 * n_qubits)
    one-hot dims (a few hundred rows cannot support 30+ length scales), with the
    3 summary dims keeping their own ARD scales.
    """
    if not _SKLEARN_AVAILABLE:  # pragma: no cover
        raise ImportError(f"scikit-learn unavailable: {_SKLEARN_IMPORT_ERROR!r}")
    obs_idx = feature_index_map_["pauli"]
    onehot_idx = feature_index_map_.get("pauli_onehot", [])
    summary_idx = feature_index_map_.get("pauli_summaries", [])
    if len(obs_idx) == 0:
        return None
    if bool(config.pauli_ard) or len(onehot_idx) == 0:
        return _SubsetKernel(_make_base_kernel(len(obs_idx), config), obs_idx)
    k = _SubsetKernel(
        _make_base_kernel(len(onehot_idx), config, force_scalar=True), onehot_idx
    )
    if len(summary_idx) > 0:
        k = k * _SubsetKernel(_make_base_kernel(len(summary_idx), config), summary_idx)
    return k


# ---------------------------------------------------------------------------
# Analytic input-gradient of the (fitted) kernel: d k(x*, x_n) / d x*.
#
# This is the purely CLASSICAL half of the analytic VQE gradient. The GP mean is
#     m(x*) = y_mean + y_std * sum_n alpha_n * k(x*, x_n),
# so  dm/dx* = y_std * sum_n alpha_n * dk(x*, x_n)/dx*.
# The quantum device is never involved here: kernel, training points x_n, dual
# weights alpha_n and hyperparameters all live on the classical machine after
# the fit. The only device-supplied quantity (d O_noisy / d theta, via a
# parameter-shift on the real measurement) is chained in later, in
# ``Mitigator.energy_gradient``.
# ---------------------------------------------------------------------------


def _leaf_kernel_value_and_input_grad(kernel, xs: np.ndarray, Xs: np.ndarray):
    """Value and d/dx* gradient of a leaf kernel restricted to its own columns.

    ``xs`` is the (sliced) query point, shape (k,); ``Xs`` the (sliced) training
    points, shape (m, k). Returns ``(v, g)`` with ``v`` shape (m,) the kernel
    values ``k(xs, Xs[n])`` and ``g`` shape (m, k) the derivative w.r.t. ``xs``.
    Supports the exact leaf kernels used in ``SingleGPModel.build_kernel``.
    """
    m, k = Xs.shape
    if isinstance(kernel, DotProduct):
        sigma0 = float(kernel.sigma_0)
        v = sigma0 * sigma0 + Xs @ xs
        # d/dx*_d ( sigma0^2 + sum_d x*_d * x_n,d ) = x_n,d
        return v, Xs.copy()
    if isinstance(kernel, (Matern, RBF)):
        ls = np.asarray(kernel.length_scale, dtype=float)
        if ls.ndim == 0:
            ls = np.full(k, float(ls))
        diff = xs[None, :] - Xs  # (m, k) = x* - x_n
        scaled = diff / ls[None, :]
        # NOTE: sklearn's ``Matern`` is a SUBCLASS of ``RBF``, so the Matern
        # branch must run first -- ``isinstance(matern, RBF)`` is True.
        if not isinstance(kernel, Matern):
            r2 = np.sum(scaled * scaled, axis=1)
            v = np.exp(-0.5 * r2)
            # d/dx*_d exp(-1/2 sum ((x*-x_n)/l)^2) = -k * (x*_d - x_n,d)/l_d^2
            g = -v[:, None] * (diff / (ls[None, :] ** 2))
            return v, g
        # Matern: k(r) with r = ||(x*-x_n)/l||_2.
        r = np.sqrt(np.sum(scaled * scaled, axis=1))
        nu = float(kernel.nu)
        if abs(nu - 0.5) < 1e-8:
            v = np.exp(-r)
            dvdr = -np.exp(-r)
        elif abs(nu - 1.5) < 1e-8:
            a = np.sqrt(3.0) * r
            v = (1.0 + a) * np.exp(-a)
            dvdr = -3.0 * r * np.exp(-a)
        elif abs(nu - 2.5) < 1e-8:
            a = np.sqrt(5.0) * r
            v = (1.0 + a + (5.0 / 3.0) * r * r) * np.exp(-a)
            dvdr = -(5.0 / 3.0) * r * (1.0 + a) * np.exp(-a)
        else:
            raise ValueError(
                f"Analytic Matern input-gradient supports nu in {{0.5,1.5,2.5}}, got nu={nu}."
            )
        # dr/dx*_d = (x*_d - x_n,d)/(l_d^2 * r); at r=0 the gradient is 0 for
        # these nu (dvdr carries a factor r), so guard the division.
        safe_r = np.where(r > 0.0, r, 1.0)
        g = (dvdr / safe_r)[:, None] * (diff / (ls[None, :] ** 2))
        g = np.where((r > 0.0)[:, None], g, 0.0)
        return v, g
    if isinstance(kernel, ConstantKernel):
        return np.full(m, float(kernel.constant_value)), np.zeros((m, k))
    if isinstance(kernel, WhiteKernel):
        # White contributes 0 to the cross-covariance k(x*, x_train) for x* not
        # exactly equal to a training point, hence 0 to the predictive mean.
        return np.zeros(m), np.zeros((m, k))
    raise ValueError(
        f"Unsupported leaf kernel for analytic input gradient: {type(kernel).__name__}."
    )


def _kernel_value_and_input_grad(kernel, x_star: np.ndarray, X_train: np.ndarray):
    """Recursively compute ``(k(x*, X_train), d k/d x*)`` for a composite kernel.

    Handles ``Sum``/``Product`` of the kernels assembled in ``build_kernel``
    (ConstantKernel, WhiteKernel, and ``_SubsetKernel``-wrapped DotProduct /
    Matern / RBF). ``x_star`` is shape (D,), ``X_train`` shape (m, D); returns
    ``v`` shape (m,) and ``g`` shape (m, D) (gradient w.r.t. ``x_star``).
    """
    from sklearn.gaussian_process.kernels import Product, Sum

    m, D = X_train.shape
    if isinstance(kernel, Sum):
        v1, g1 = _kernel_value_and_input_grad(kernel.k1, x_star, X_train)
        v2, g2 = _kernel_value_and_input_grad(kernel.k2, x_star, X_train)
        return v1 + v2, g1 + g2
    if isinstance(kernel, Product):
        v1, g1 = _kernel_value_and_input_grad(kernel.k1, x_star, X_train)
        v2, g2 = _kernel_value_and_input_grad(kernel.k2, x_star, X_train)
        return v1 * v2, v1[:, None] * g2 + v2[:, None] * g1
    if isinstance(kernel, ConstantKernel):
        return np.full(m, float(kernel.constant_value)), np.zeros((m, D))
    if isinstance(kernel, WhiteKernel):
        return np.zeros(m), np.zeros((m, D))
    if isinstance(kernel, _SubsetKernel):
        idx = list(kernel.indices)
        v, g_sub = _leaf_kernel_value_and_input_grad(
            kernel.kernel, np.asarray(x_star)[idx], np.asarray(X_train)[:, idx]
        )
        g = np.zeros((m, D))
        g[:, idx] = g_sub
        return v, g
    # Bare leaf kernel applied to all dimensions (not used by build_kernel, but
    # handled for completeness).
    return _leaf_kernel_value_and_input_grad(kernel, np.asarray(x_star), np.asarray(X_train))


class SingleGPModel:
    """One GP over all observables; target = raw ``O_ideal`` (no separate CDR).

    The CDR line is built INTO the kernel: a linear ``DotProduct`` term on the
    ``o_noisy`` feature column reproduces ``a * o_noisy + b`` (the slope/bias are
    learned as kernel hyperparameters), optionally multiplied by a per-observable
    kernel so the effective CDR slope can vary by Pauli. A Matern/RBF term on the
    angle+pauli block then captures the coherent, angle-dependent residual, and a
    WhiteKernel absorbs shot noise. ``predict`` returns ``(mean = O_mit, std)``.
    """

    def __init__(self, config: MitigatorConfig) -> None:
        if not _SKLEARN_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for SingleGPModel. Install it into the "
                f"active environment. Original import error: {_SKLEARN_IMPORT_ERROR!r}"
            )
        self.config = config
        self.gp: "GaussianProcessRegressor | None" = None
        self._index_map = feature_index_map(config)

    def build_kernel(self, feature_index_map_: dict, config: MitigatorConfig):
        noisy_idx = feature_index_map_["noisy"]
        angle_idx = feature_index_map_["angle"]
        loc_idx = feature_index_map_.get("location", [])
        obs_idx = feature_index_map_["pauli"]
        cont_idx = angle_idx + loc_idx  # theta-dependent (angle + location) block
        base_idx = cont_idx + obs_idx

        if not config.include_noisy_feature_in_gp or len(noisy_idx) == 0:
            raise ValueError(
                "The combined (Design 2) mitigator requires an o_noisy feature "
                "column for the in-kernel CDR term; set "
                "include_noisy_feature_in_gp=True."
            )

        white = WhiteKernel(
            noise_level=float(config.noise_variance_init),
            noise_level_bounds=(1e-8, 1e2),
        )

        # k_lin: DotProduct on o_noisy reproduces the CDR line a*o_noisy + b
        # (slope via the ConstantKernel amplitude, bias via DotProduct sigma_0).
        # sigma_0 (the o_noisy-independent bias channel) is bounded so it cannot
        # soak up angle structure and bypass the measurement.
        sigma0_hi = max(float(config.dotproduct_sigma0_bound), 1e-4)
        k_lin = ConstantKernel(1.0, (1e-3, 1e3)) * _SubsetKernel(
            DotProduct(
                sigma_0=min(1.0, sigma0_hi), sigma_0_bounds=(1e-5, sigma0_hi)
            ),
            noisy_idx,
        )
        # Optionally make the effective CDR slope observable-dependent.
        if config.use_product_kernel and len(obs_idx) > 0:
            k_obs = _make_pauli_kernel(feature_index_map_, config)
            k_lin = k_lin * k_obs

        kernel = k_lin

        # k_base: coherent / angle-dependent residual over (angles, location,
        # pauli). In the legacy "zero" prior-mean mode this pure-angle path can
        # absorb the whole O_ideal(theta) signal and turn the model into a
        # classical surrogate that ignores the measurement; its amplitude is
        # therefore bounded by ``angle_term_amplitude_bound`` (in normalized-
        # target units). In "backbone" mode the target is already the small
        # CDR residual, so the legacy wide bounds are kept.
        if len(base_idx) > 0:
            if str(config.prior_mean_mode).lower() == "zero":
                amp_hi = max(float(config.angle_term_amplitude_bound), 2e-3)
            else:
                amp_hi = 1e3
            c_base = ConstantKernel(
                min(1.0, 0.5 * amp_hi), (1e-3, amp_hi)
            )
            if bool(config.pauli_ard) or len(obs_idx) == 0:
                k_base = c_base * _SubsetKernel(
                    _make_base_kernel(len(base_idx), config), base_idx
                )
            else:
                # Separable form so the one-hot block can share one length scale.
                k_base = c_base * _SubsetKernel(
                    _make_base_kernel(len(cont_idx), config), cont_idx
                ) * _make_pauli_kernel(feature_index_map_, config)
            kernel = kernel + k_base

        return kernel + white

    def fit(
        self,
        X: np.ndarray,
        y_target: np.ndarray,
        sample_alpha: "np.ndarray | None" = None,
    ) -> None:
        """Fit the GP. ``y_target`` is O_ideal ("zero" mode) or the CDR-backbone
        residual ("backbone" mode). ``sample_alpha`` is an optional per-row noise
        variance (heteroscedastic shot noise), in the units of ``y_target``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y_target, dtype=float).ravel()
        alpha_rows = (
            None
            if sample_alpha is None
            else np.asarray(sample_alpha, dtype=float).ravel()
        )
        cap = int(getattr(self.config, "max_gp_train_points", 0) or 0)
        if cap > 0 and X.shape[0] > cap:
            # Subsample rows so the exact-GP gradient tensor stays bounded.
            sub = np.random.default_rng(int(self.config.rng_seed)).choice(
                X.shape[0], size=cap, replace=False
            )
            X = X[sub]
            y = y[sub]
            if alpha_rows is not None:
                alpha_rows = alpha_rows[sub]
        kernel = self.build_kernel(self._index_map, self.config)
        if alpha_rows is None:
            alpha = 1e-10  # WhiteKernel carries the (learned) shot noise.
        else:
            # sklearn adds ``alpha`` to K in NORMALIZED-target space when
            # normalize_y=True, so rescale the physical variances by var(y).
            scale = float(np.var(y)) if bool(self.config.normalize_targets) else 1.0
            alpha = alpha_rows / max(scale, 1e-12) + 1e-10
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=bool(self.config.normalize_targets),
            n_restarts_optimizer=int(self.config.gp_n_restarts),
            random_state=int(self.config.rng_seed),
        )
        self.gp.fit(X, y)

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.gp is None:
            raise RuntimeError("SingleGPModel.fit must be called before predict.")
        X_star = np.asarray(X_star, dtype=float)
        mean, std = self.gp.predict(X_star, return_std=True)
        std = np.asarray(std, dtype=float)
        if bool(getattr(self.config, "epistemic_std", False)):
            # sklearn's predictive variance includes the WhiteKernel diagonal
            # (aleatoric shot noise). Subtract it so the reported std measures
            # EPISTEMIC uncertainty -- the part more data can actually reduce.
            # With normalize_y the kernel lives in normalized-target units, so
            # the noise level is rescaled by y_train_std^2 back to output units.
            white = _sum_white_noise_level(self.gp.kernel_)
            y_std_arr = np.asarray(
                getattr(self.gp, "_y_train_std", 1.0), dtype=float
            ).ravel()
            y_std_f = float(y_std_arr[0]) if y_std_arr.size else 1.0
            var = np.maximum(std**2 - white * y_std_f**2, 0.0)
            std = np.sqrt(var)
        return np.asarray(mean, dtype=float), std

    def predict_input_gradient(self, X_star: np.ndarray) -> np.ndarray:
        """Analytic ``d mean / d x*`` of the GP posterior mean (per query row).

        The mean is ``m(x*) = y_mean + y_std * sum_n alpha_n k(x*, x_n)``, so
        ``dm/dx* = y_std * sum_n alpha_n dk(x*, x_n)/dx*``. Returns shape
        ``(n_queries, n_features)``. Purely classical (no device access).
        """
        if self.gp is None:
            raise RuntimeError("SingleGPModel.fit must be called before predict_input_gradient.")
        X_star = np.asarray(X_star, dtype=float)
        gp = self.gp
        kernel = gp.kernel_
        X_train = np.asarray(gp.X_train_, dtype=float)
        alpha = np.asarray(gp.alpha_, dtype=float).ravel()
        y_std = getattr(gp, "_y_train_std", 1.0)
        y_std_arr = np.asarray(y_std, dtype=float).ravel()
        y_std_f = float(y_std_arr[0]) if y_std_arr.size else 1.0
        out = np.zeros((X_star.shape[0], X_star.shape[1]), dtype=float)
        for i, x in enumerate(X_star):
            _, kg = _kernel_value_and_input_grad(kernel, x, X_train)  # (m, D)
            out[i] = y_std_f * (kg.T @ alpha)
        return out


# ---------------------------------------------------------------------------
# Shared fit / predict helpers (used by the Mitigator AND the validation
# functions so the backbone prior mean and heteroscedastic noise are applied
# identically everywhere).
# ---------------------------------------------------------------------------


def _use_backbone_prior(config: MitigatorConfig) -> bool:
    return str(config.prior_mean_mode).lower() == "backbone"


def _row_alphas(rows: list[dict], config: MitigatorConfig) -> "np.ndarray | None":
    """Per-row shot-noise variance ``~ (1 - o^2) / n_samples`` (binomial), using
    the effective per-term sample counts returned by the OGM estimator. ``None``
    when ``heteroscedastic_alpha`` is off (the shared WhiteKernel then carries
    all the noise).
    """
    if not bool(config.heteroscedastic_alpha):
        return None
    out = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        o = float(r["o_noisy"])
        n = max(1, int(r.get("n_samples", config.shots)))
        out[i] = max(1.0 - min(o * o, 1.0), 0.05) / n
    return out


def _backbone_values(
    backbone: "CDRBackbone", rows: list[dict]
) -> np.ndarray:
    return np.asarray(
        [backbone.apply(r["pauli"], r["o_noisy"]) for r in rows], dtype=float
    )


def _fit_backbone_and_gp(
    rows: list[dict], config: MitigatorConfig
) -> "tuple[CDRBackbone, SingleGPModel]":
    """Fit the per-Pauli CDR backbone, then the single GP.

    In ``prior_mean_mode="backbone"`` the GP is trained on the RESIDUALS
    ``o_ideal - (a_P * o_noisy + b_P)`` so the mitigated value always carries
    the measured signal through the backbone; in ``"zero"`` mode the GP maps
    features directly to ``o_ideal`` (legacy combined design).
    """
    backbone = CDRBackbone(config.affine_regularization)
    backbone.fit(rows)
    gp = SingleGPModel(config)
    X = np.stack([build_feature_row(r, config) for r in rows], axis=0)
    y = np.asarray([float(r["o_ideal"]) for r in rows], dtype=float)
    if _use_backbone_prior(config):
        y = y - _backbone_values(backbone, rows)
    gp.fit(X, y, sample_alpha=_row_alphas(rows, config))
    return backbone, gp


def _predict_rows(
    gp: "SingleGPModel",
    backbone: "CDRBackbone",
    rows: list[dict],
    config: MitigatorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Mitigated per-row prediction ``(mean, std)`` honoring the prior mean."""
    X = np.stack([build_feature_row(r, config) for r in rows], axis=0)
    mean, std = gp.predict(X)
    if _use_backbone_prior(config):
        mean = mean + _backbone_values(backbone, rows)
    return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


# ---------------------------------------------------------------------------
# Section 7 - Combined mitigator (public API)
# ---------------------------------------------------------------------------


class Mitigator:
    def __init__(
        self, backend: CirqBackendAdapter, hamiltonian, config: MitigatorConfig
    ) -> None:
        self.adapter = backend
        self.hamiltonian = hamiltonian  # kept for parity with the spec signature
        self.config = config
        # Auto-fill the theta-location descriptors from the adapter's circuit so
        # the location interaction features work without extra notebook wiring.
        if config.include_location_features and config.param_locations is None:
            config.param_locations = np.asarray(backend.param_locations, dtype=float)
            config.param_lightcones = np.asarray(backend.param_lightcones, dtype=float)
        # The single GP maps features to the target (O_ideal or the backbone
        # residual, depending on ``prior_mean_mode``); the backbone doubles as
        # the plain-CDR baseline for validation/plots.
        self.gp = SingleGPModel(config)
        self.backbone = CDRBackbone(config.affine_regularization)
        self.rows: list[dict] = []
        self._current_theta: np.ndarray | None = None
        if config.theta_init is None:
            config.theta_init = np.zeros(int(config.n_params), dtype=float)

    # -- convenience ------------------------------------------------------
    @property
    def coeff_by_pauli(self) -> dict:
        return self.adapter.coeff_by_pauli

    def _feature_matrix(self, rows: list[dict]) -> np.ndarray:
        return np.stack([build_feature_row(r, self.config) for r in rows], axis=0)

    def _fit_gp_from_rows(self) -> None:
        # Fit the backbone first (its line defines the residual target in
        # "backbone" mode), then the single GP -- via the shared helper so the
        # validation paths use the exact same recipe.
        self.backbone, self.gp = _fit_backbone_and_gp(self.rows, self.config)

    # -- warm start -------------------------------------------------------
    def warmstart(self) -> None:
        theta0 = np.asarray(self.config.theta_init, dtype=float).ravel()
        self._current_theta = theta0
        resolvers = self.adapter.generate_near_clifford(
            theta0,
            n_circuits=int(self.config.n_warmstart_circuits),
            n_nonclifford=self.config.n_nonclifford_gates,
            snap_step=float(self.config.clifford_snap_step),
            spread=float(self.config.warmstart_spread),
            seed=int(self.config.rng_seed),
            node_hops=int(self.config.clifford_node_hops),
        )
        self.rows = self.adapter.collect_rows(
            resolvers, shots=int(self.config.shots), seed_base=int(self.config.rng_seed) + 1
        )
        # Fully-Clifford anchor circuits over a wider grid: never pruned, keep
        # the GP pinned globally even after large theta moves.
        n_anchor = int(getattr(self.config, "n_clifford_anchor_circuits", 0) or 0)
        if n_anchor > 0:
            anchor_res = self.adapter.generate_clifford_anchors(
                theta0,
                n_circuits=n_anchor,
                snap_step=float(self.config.clifford_snap_step),
                seed=int(self.config.rng_seed) + 91,
                node_hops=int(self.config.anchor_node_hops),
            )
            self.rows.extend(
                self.adapter.collect_rows(
                    anchor_res,
                    shots=int(self.config.shots),
                    seed_base=int(self.config.rng_seed) + 2001,
                    anchor=True,
                )
            )
        self._fit_gp_from_rows()  # fits the combined GP (and the baseline backbone)

    # -- prediction -------------------------------------------------------
    def mitigate(self, theta, o_noisy_by_pauli: dict) -> dict:
        o_mit, _ = self.predict_with_uncertainty(theta, o_noisy_by_pauli)
        return o_mit

    def predict_with_uncertainty(
        self, theta, o_noisy_by_pauli: dict
    ) -> tuple[dict, dict]:
        theta = np.asarray(theta, dtype=float).ravel()
        labels = list(o_noisy_by_pauli.keys())
        rows = [
            {"theta": theta, "pauli": label, "o_noisy": float(o_noisy_by_pauli[label])}
            for label in labels
        ]
        # Honors prior_mean_mode: in "backbone" mode the mean is
        # (a_P * o_noisy + b_P) + GP residual, so the measurement always flows
        # through the prediction.
        mean, std = _predict_rows(self.gp, self.backbone, rows, self.config)
        o_mit: dict = {}
        std_by: dict = {}
        for k, label in enumerate(labels):
            o_mit[label] = float(mean[k])
            std_by[label] = float(std[k])
        return o_mit, std_by

    # -- online updates ---------------------------------------------------
    def update_with_rows(self, new_rows: list[dict], current_theta=None) -> None:
        if current_theta is not None:
            self._current_theta = np.asarray(current_theta, dtype=float).ravel()
        self.rows.extend(new_rows)
        if (
            self.config.drop_faraway_points
            and len(self.rows) > int(self.config.max_gp_points)
        ):
            self._prune_rows()
        self._fit_gp_from_rows()  # refits the combined GP (and baseline backbone)

    def _prune_rows(self) -> None:
        cap = int(self.config.max_gp_points)
        # Anchor rows (fully-Clifford global pins) are NEVER pruned; the
        # distance-based cap applies to the remaining rows.
        anchors = [r for r in self.rows if r.get("anchor")]
        others = [r for r in self.rows if not r.get("anchor")]
        cap_others = max(0, cap - len(anchors))
        if self._current_theta is not None:
            dist = np.asarray(
                [
                    float(np.linalg.norm(np.asarray(r["theta"]) - self._current_theta))
                    for r in others
                ],
                dtype=float,
            )
            keep = np.argsort(dist)[:cap_others]
            others = [others[i] for i in sorted(keep.tolist())]
        else:
            others = others[-cap_others:] if cap_others > 0 else []
        self.rows = anchors + others

    def mitigated_energy(self, theta, o_noisy_by_pauli: dict) -> float:
        o_mit = self.mitigate(theta, o_noisy_by_pauli)
        return self.adapter.energy_from_values(o_mit)

    # -- analytic energy gradient ----------------------------------------
    def energy_gradient(
        self, theta, o_noisy_by_pauli: dict, do_noisy_dtheta: dict
    ) -> np.ndarray:
        """Analytic ``dE_mit/dtheta`` at the CURRENT theta (in-distribution).

        Chains the classical GP-mean input gradient with the feature Jacobian:

            dE/dtheta_j = sum_P w_P * [ d mean_P/d(angle) . d(angle)/dtheta_j
                                        + d mean_P/d(o_noisy) * d o_noisy_P/dtheta_j ]

        - ``d(angle)/dtheta_j`` (Fourier features) is closed form (classical).
        - the location interaction features also depend on theta and their
          Jacobian (``cos(theta_j) * descriptor``) is included per Pauli row.
        - ``d o_noisy_P/dtheta_j`` is the only device-supplied term and must be
          passed in (parameter-shift on the real noisy measurement). It maps
          ``pauli label -> np.ndarray`` of length ``n_params``.
        - in ``prior_mean_mode="backbone"`` the backbone term contributes
          ``a_P * d o_noisy_P/dtheta_j`` on top of the GP-residual gradient.

        Pauli features are constant in theta, so they contribute nothing.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        labels = list(self.adapter.obs_labels)
        rows = [
            {"theta": theta, "pauli": label, "o_noisy": float(o_noisy_by_pauli[label])}
            for label in labels
        ]
        X = self._feature_matrix(rows)
        dmean = self.gp.predict_input_gradient(X)  # (P, D), classical

        idx = feature_index_map(self.config)
        angle_idx = idx["angle"]
        noisy_idx = idx["noisy"]
        loc_idx = idx.get("location", [])
        n_params = int(self.config.n_params)
        n_harm = int(self.config.max_harmonic)

        # d(angle features)/d theta_j: nonzero only for the angle dims of param j.
        n_angle = len(angle_idx)
        dangle = np.zeros((n_angle, n_params), dtype=float)
        for i in range(n_params):
            ti = float(theta[i])
            base = i * 2 * n_harm
            for kk in range(1, n_harm + 1):
                c = base + (kk - 1) * 2
                dangle[c, i] = -kk * np.sin(kk * ti)      # d cos(k t_i)/dt_i
                dangle[c + 1, i] = kk * np.cos(kk * ti)   # d sin(k t_i)/dt_i

        # angle contribution: (P, n_angle) @ (n_angle, n_params) -> (P, n_params)
        d_omit_dtheta = dmean[:, angle_idx] @ dangle

        # location-interaction contribution (features = sin(theta_j) * c_jk with
        # c_jk = [depth_frac, down_frac, lightcone_overlap(j, P)]):
        # d/dtheta_j = cos(theta_j) * c_jk. The overlap depends on the Pauli, so
        # the Jacobian varies per row.
        if len(loc_idx) > 0:
            locs = np.asarray(self.config.param_locations, dtype=float)
            cos_t = np.cos(theta)
            for i, label in enumerate(labels):
                ov = lightcone_overlap(self.config, label)  # (n_params,)
                for j in range(n_params):
                    cols = loc_idx[
                        _N_LOC_PER_PARAM * j : _N_LOC_PER_PARAM * (j + 1)
                    ]
                    c_vec = np.asarray(
                        [locs[j, 0], locs[j, 1], float(ov[j])], dtype=float
                    )
                    d_omit_dtheta[i, j] += float(
                        np.dot(dmean[i, cols], c_vec)
                    ) * float(cos_t[j])

        # device-supplied noisy-measurement contribution: through the GP's
        # o_noisy feature column AND, in backbone mode, through the backbone
        # slope a_P (the always-on measurement channel).
        do = np.stack(
            [np.asarray(do_noisy_dtheta[label], dtype=float).ravel() for label in labels],
            axis=0,
        )  # (P, n_params)
        if len(noisy_idx) > 0:
            d_omit_dtheta = d_omit_dtheta + dmean[:, noisy_idx[0]][:, None] * do
        if _use_backbone_prior(self.config):
            slopes = np.asarray(
                [self.backbone.coeffs.get(label, (1.0, 0.0))[0] for label in labels],
                dtype=float,
            )
            d_omit_dtheta = d_omit_dtheta + slopes[:, None] * do

        weights = np.asarray(self.adapter.weights, dtype=float)  # aligned with obs_labels
        grad = weights @ d_omit_dtheta  # (n_params,)
        return np.asarray(grad, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Section 8 - Top-up controller
# ---------------------------------------------------------------------------


def needs_topup(std_by_pauli: dict, coeff_by_pauli: dict, config: MitigatorConfig) -> bool:
    return weighted_uncertainty(std_by_pauli, coeff_by_pauli) > float(
        config.uncertainty_threshold
    )


def weighted_uncertainty(std_by_pauli: dict, coeff_by_pauli: dict) -> float:
    """Energy-scale predictive uncertainty: ``sqrt(sum_i c_i^2 sigma_i^2)``.

    Quadrature aggregation (independent per-Pauli errors) instead of the old L1
    sum ``sum_i |c_i| sigma_i``: with ~100 Hamiltonian terms the L1 form is a
    perfectly-correlated worst case ~10x larger than the realistic energy error,
    which made any Eh-scale threshold permanently unreachable and forced the
    top-up gate / backbone fallback to fire on every iteration.
    """
    total = 0.0
    for label, std in std_by_pauli.items():
        c = float(coeff_by_pauli.get(label, 0.0))
        total += (c * float(std)) ** 2
    return float(np.sqrt(total))


def sample_local_rows(
    adapter: CirqBackendAdapter, theta, config: MitigatorConfig, seed: int
) -> list[dict]:
    theta = np.asarray(theta, dtype=float).ravel()
    resolvers = adapter.generate_near_clifford(
        theta,
        n_circuits=int(config.topup_batch_size),
        n_nonclifford=config.n_nonclifford_gates,
        snap_step=float(config.clifford_snap_step),
        spread=float(config.effective_topup_radius()),
        seed=int(seed),
        node_hops=int(config.clifford_node_hops),
    )
    return adapter.collect_rows(resolvers, shots=int(config.shots), seed_base=int(seed) + 1)


def prequential_topup_error(mitigator: "Mitigator", new_rows: list[dict]) -> float:
    """Mean absolute per-circuit ENERGY error of the CURRENT model on a fresh
    near-Clifford batch, evaluated BEFORE the batch is added.

    Every top-up row carries an exact ``o_ideal``, so this is a real, unbiased
    local error estimate -- immune to the GP's own (possibly overconfident)
    predictive std. The metric is the SIGNED energy error per circuit,
    ``|sum_P c_P * (pred_P - ideal_P)|`` averaged over the batch circuits: the
    quantity the VQE actually consumes, in which per-Pauli errors cancel (the
    old L1 form ``sum_P |c_P| |err_P|`` overstated it ~10x and locked the
    controller into permanent backbone fallback).
    """
    if not new_rows:
        return 0.0
    mean, _ = _predict_rows(mitigator.gp, mitigator.backbone, new_rows, mitigator.config)
    coeff = mitigator.coeff_by_pauli
    # Group rows by circuit via their theta vector (top-up thetas carry
    # continuous jitter, so collisions are not a concern in practice).
    err_by_circuit: dict[tuple, float] = {}
    for r, m in zip(new_rows, mean):
        key = tuple(np.round(np.asarray(r["theta"], dtype=float), 12).tolist())
        err_by_circuit[key] = err_by_circuit.get(key, 0.0) + float(
            coeff.get(r["pauli"], 0.0)
        ) * (float(m) - float(r["o_ideal"]))
    return float(np.mean([abs(v) for v in err_by_circuit.values()]))


# ---------------------------------------------------------------------------
# Section 9 - VQE loop integration (bounded-step / trust region)
# ---------------------------------------------------------------------------


def run_vqe_with_mitigator(
    mitigator: Mitigator,
    theta_init: np.ndarray,
    *,
    max_iters: int | None = None,
    step_max: float | None = None,
    learning_rate: float = 0.5,
    fd_eps: float = 0.05,
    gradient_mode: str = "parameter_shift",
    ideal_energy_fn: Callable[[np.ndarray], float] | None = None,
    verbose: bool = True,
) -> dict:
    """Fixed-learning-rate gradient descent on the mitigated energy.

    Matches the optimizer in ``June_main/main_HF.ipynb``: a **parameter-shift**
    gradient (exact +/- pi/2 shifts, prefactor 1/2 -- each ansatz angle drives a
    single RX/RZX with eigenvalues +/-1) and a **full fixed-LR update**
    ``theta <- theta - lr * grad`` on every coordinate, for a fixed number of
    iterations. Set ``gradient_mode='finite_difference'`` to use central
    differences with ``fd_eps`` instead.

    Per iteration:
      1. measure the REAL circuit's noisy expectation values for all Paulis,
      2. predict mitigated values + uncertainty,
      3. top up locally if uncertain, then re-predict,
      4. energy = sum_i c_i * O_mit[P_i]; gradient; fixed-LR step.

    ``step_max`` is OFF by default (None) to match main_HF's un-clipped fixed-LR
    update; pass a float to re-enable trust-region step clipping. Top-up frequency
    should taper toward zero as the optimizer settles.
    """
    cfg = mitigator.config
    adapter = mitigator.adapter
    max_iters = int(cfg.max_vqe_iterations if max_iters is None else max_iters)
    # step_max stays None unless explicitly provided -> no clipping (main_HF style).

    mode = str(gradient_mode).strip().lower()
    if mode == "gp_analytic":
        mode = "analytic"
    if mode not in {"parameter_shift", "finite_difference", "analytic"}:
        raise ValueError(
            "gradient_mode must be 'parameter_shift', 'finite_difference' or "
            f"'analytic', got {gradient_mode!r}."
        )
    shift = float(np.pi) / 2.0 if mode == "parameter_shift" else float(fd_eps)
    grad_scale = 0.5 if mode == "parameter_shift" else 1.0 / (2.0 * float(fd_eps))

    theta = np.asarray(theta_init, dtype=float).ravel()
    rng = np.random.default_rng(int(cfg.rng_seed) + 777)

    def measure(th: np.ndarray) -> dict:
        return adapter.run_noisy(
            adapter.resolver_from_theta(th),
            shots=int(cfg.shots),
            sampling_seed=int(rng.integers(1, 2**31 - 1)),
        )

    def mitigated_energy(th: np.ndarray) -> float:
        bundle = measure(th)
        return adapter.energy_from_values(mitigator.mitigate(th, bundle["primary"]))

    def compute_grad(th: np.ndarray, measured_th: dict) -> np.ndarray:
        if mode == "analytic":
            # Quantum part: parameter-shift on the REAL noisy measurement gives
            # d O_noisy_P/d theta_j (exact for single-generator gates, up to shot
            # noise). Same device cost as parameter_shift -- the difference is the
            # GP gradient is evaluated analytically at the current (in-distribution)
            # theta instead of querying the surrogate at the far-away +/- pi/2 points.
            half = float(np.pi) / 2.0
            do_noisy = {label: np.zeros(len(th)) for label in measured_th}
            for j in range(len(th)):
                tp = th.copy()
                tm = th.copy()
                tp[j] += half
                tm[j] -= half
                bp = measure(tp)["primary"]
                bm = measure(tm)["primary"]
                for label in measured_th:
                    do_noisy[label][j] = 0.5 * (bp[label] - bm[label])
            return mitigator.energy_gradient(th, measured_th, do_noisy)
        g = np.zeros_like(th)
        for j in range(len(th)):
            tp = th.copy()
            tm = th.copy()
            tp[j] += shift
            tm[j] -= shift
            g[j] = (mitigated_energy(tp) - mitigated_energy(tm)) * grad_scale
        return g

    def do_topup(
        th: np.ndarray, it_seed: int, n_batches: int = 1
    ) -> tuple[dict, dict, "float | None"]:
        """Add ``n_batches`` local near-Clifford batches; return the refreshed
        prediction and the PREQUENTIAL error of the last batch (the current
        model's per-circuit energy error on the fresh rows, measured BEFORE they were
        added -- a real local accuracy estimate, not the GP's own std)."""
        preq_err: "float | None" = None
        for b in range(int(max(1, n_batches))):
            new_rows = sample_local_rows(
                adapter, th, cfg, seed=int(it_seed) + 101 * b
            )
            if bool(cfg.prequential_topup_check):
                preq_err = prequential_topup_error(mitigator, new_rows)
            mitigator.update_with_rows(new_rows, current_theta=th)
        o_mit_, std_ = mitigator.predict_with_uncertainty(th, measured)
        return o_mit_, std_, preq_err

    history: list[dict] = []
    topup_total = 0
    # Shot accounting: snapshot the backend counters so the caller gets the exact
    # number of circuit executions (and shots) this optimization consumed.
    evals_at_start = int(adapter.n_circuit_evals)
    shots_at_start = int(adapter.total_shots)
    # Center of the most recent top-up (trust region). Seeded at theta_init since
    # the warm-start cloud is built around it.
    last_topup_theta = theta.copy()
    converged = False
    for it in range(max_iters):
        bundle = measure(theta)
        measured = bundle["primary"]
        o_mit, std = mitigator.predict_with_uncertainty(theta, measured)

        # --- Decide whether to top up (uncertainty OR trust-region OR periodic) ---
        unc_trigger = needs_topup(std, mitigator.coeff_by_pauli, cfg)
        moved = float(np.linalg.norm(theta - last_topup_theta))
        move_trigger = bool(cfg.topup_on_move) and moved > (
            float(cfg.topup_move_fraction) * cfg.effective_topup_radius()
        )
        periodic_trigger = (
            int(cfg.topup_every) > 0 and it > 0 and (it % int(cfg.topup_every) == 0)
        )
        topped = 0
        iter_topups = 0  # number of top-up batches this iter (for shot accounting)
        preq_err: "float | None" = None
        used_backbone_fallback = False
        if unc_trigger or move_trigger or periodic_trigger:
            # Move-scaled: a large theta jump gets proportionally more batches
            # (capped) so the new region is re-covered before the gradient/step.
            n_batches = 1
            if move_trigger:
                n_batches = int(
                    min(
                        max(1, int(np.ceil(moved / max(cfg.effective_topup_radius(), 1e-9)))),
                        int(max(1, cfg.max_move_topup_batches)),
                    )
                )
            o_mit, std, preq_err = do_topup(
                theta, int(cfg.rng_seed) + 10_000 + it, n_batches=n_batches
            )
            last_topup_theta = theta.copy()
            topped = 1
            iter_topups += n_batches
            topup_total += n_batches

            def _trust_metric() -> float:
                m = weighted_uncertainty(std, mitigator.coeff_by_pauli)
                if preq_err is not None:
                    m = max(m, float(preq_err))
                return float(m)

            # (3) Top-up-until-satisfied: keep adding local batches while the GP
            #     is still untrustworthy at theta -- by its own std OR by the
            #     (overconfidence-immune) prequential error -- bounded by
            #     ``max_topup_retries`` so the shot cost stays capped.
            retries = 0
            while (
                int(cfg.max_topup_retries) > 0
                and _trust_metric() > float(cfg.uncertainty_threshold)
                and retries < int(cfg.max_topup_retries)
            ):
                o_mit, std, preq_err = do_topup(
                    theta, int(cfg.rng_seed) + 30_000 + it * 1000 + retries
                )
                iter_topups += 1
                topup_total += 1
                retries += 1

            # If the model is STILL locally invalid by real (prequential) error,
            # do not trust its mean for this iteration: use the plain per-Pauli
            # CDR backbone, which is guaranteed to track the measurement.
            if (
                bool(cfg.fallback_to_backbone_on_mistrust)
                and preq_err is not None
                and float(preq_err) > float(cfg.uncertainty_threshold)
            ):
                o_mit = {
                    label: mitigator.backbone.apply(label, measured[label])
                    for label in measured
                }
                used_backbone_fallback = True

        energy = adapter.energy_from_values(o_mit)
        unc = weighted_uncertainty(std, mitigator.coeff_by_pauli)
        backbone_by_pauli = {
            label: mitigator.backbone.apply(label, measured[label])
            for label in measured
        }

        grad = compute_grad(theta, measured)
        grad_evals = 1  # gradient evaluations this iter (each = 2*n_params circuits)

        # --- (4) Convergence validation: a small surrogate gradient is only
        #     trustworthy if a REAL near-Clifford batch at theta doesn't move the
        #     answer. If we haven't already refit here, top up once and re-check. ---
        if float(cfg.convergence_grad_tol) > 0 and float(np.linalg.norm(grad)) < float(
            cfg.convergence_grad_tol
        ):
            if topped == 0:
                o_mit, std, preq_err = do_topup(theta, int(cfg.rng_seed) + 20_000 + it)
                last_topup_theta = theta.copy()
                topped = 1
                iter_topups += 1
                topup_total += 1
                energy_new = adapter.energy_from_values(o_mit)
                unc = weighted_uncertainty(std, mitigator.coeff_by_pauli)
                grad = compute_grad(theta, measured)
                grad_evals = 2  # re-evaluated the gradient after the validation top-up
                energy_shift = abs(energy_new - energy)
                energy = energy_new
                if float(np.linalg.norm(grad)) < float(
                    cfg.convergence_grad_tol
                ) and energy_shift < float(cfg.convergence_energy_tol):
                    converged = True
            else:
                # Surrogate was just refit with real data this iter -> trust it.
                converged = True

        step = -learning_rate * grad
        if step_max is not None:
            max_comp = float(np.max(np.abs(step))) if step.size else 0.0
            if max_comp > step_max and max_comp > 0:
                step *= step_max / max_comp

        rec = {
            "iter": it,
            "theta": theta.copy(),
            "energy_mitigated": float(energy),
            "energy_unmitigated": float(bundle["energy_unmitigated"]),
            "energy_rem": float(bundle["energy_rem"]),
            "energy_backbone": float(adapter.energy_from_values(backbone_by_pauli)),
            "weighted_uncertainty": float(unc),
            "prequential_error": float(preq_err) if preq_err is not None else float("nan"),
            "used_backbone_fallback": bool(used_backbone_fallback),
            "topped_up": int(topped),
            "n_topups": int(iter_topups),
            "topup_cum": int(topup_total),
            "grad_evals": int(grad_evals),
            "grad_norm": float(np.linalg.norm(grad)),
            "n_rows": int(len(mitigator.rows)),
        }
        if ideal_energy_fn is not None:
            rec["energy_ideal"] = float(ideal_energy_fn(theta))
        history.append(rec)
        if verbose:
            extra = (
                f"  E_ideal={rec['energy_ideal']: .6f}"
                if "energy_ideal" in rec
                else ""
            )
            preq_str = (
                f"  preq={preq_err: .4e}" if preq_err is not None else ""
            )
            fb_str = "  [backbone fallback]" if used_backbone_fallback else ""
            print(
                f"[GP-VQE] iter={it:03d}  E_mit={energy: .6f}{extra}  "
                f"unc={unc: .4e}{preq_str}  topup={iter_topups} (cum={topup_total})  "
                f"|g|={rec['grad_norm']: .3e}  rows={rec['n_rows']}{fb_str}"
            )

        if converged:
            if verbose:
                print(
                    f"[GP-VQE] converged at iter={it:03d}: |g|={rec['grad_norm']:.3e} "
                    f"< {float(cfg.convergence_grad_tol):.3e} and validated with a "
                    f"real near-Clifford top-up."
                )
            break

        theta = theta + step

    return {
        "theta_final": theta,
        "history": history,
        "topup_total": int(topup_total),
        "converged": bool(converged),
        # Empirical (numerical) shot cost of this optimization, measured straight
        # from the backend counters (excludes warm start, which happens before).
        "circuit_evals": int(adapter.n_circuit_evals - evals_at_start),
        "total_shots": int(adapter.total_shots - shots_at_start),
    }


def gradient_circuit_evals(config: MitigatorConfig) -> int:
    """Circuit executions per gradient evaluation.

    All gradient modes cost the same: each of the ``n_params`` angles needs a
    ``+`` and a ``-`` shift (parameter-shift +/- pi/2, finite-difference +/- eps,
    or the analytic mode's parameter-shift on the real device for
    ``dO_noisy/dtheta``) -> ``2 * n_params`` physical circuit executions.
    """
    return 2 * int(config.n_params)


def analytic_shot_count(
    config: MitigatorConfig,
    history: list[dict],
    *,
    include_warmstart: bool = True,
) -> dict:
    """Closed-form (analytic) reconstruction of the hardware shot cost.

    Counts EVERYTHING that touches the device, exactly as the real run does, so
    it can be cross-checked against the empirical backend counter
    (``adapter.shot_report()`` / ``vqe_out['total_shots']``):

      * warm start          : ``n_warmstart_circuits + n_clifford_anchor_circuits``
                               executions (once),
      * per VQE iteration    : 1 energy measurement
                               + ``grad_evals * 2 * n_params`` gradient circuits
                               + ``n_topups * topup_batch_size`` top-up circuits.

    ``grad_evals`` (1, or 2 on a convergence-validation iteration) and
    ``n_topups`` (the number of top-up BATCHES, including move-scaled and retry
    batches) are read back from ``history`` so the data-dependent top-up /
    early-stop behavior is reflected exactly. Every execution consumes
    ``config.shots`` shots, so ``total_shots = circuit_evals * shots``.
    """
    grad_evals_cost = gradient_circuit_evals(config)
    shots = int(config.shots)

    n_anchor = int(getattr(config, "n_clifford_anchor_circuits", 0) or 0)
    warmstart_evals = (
        int(config.n_warmstart_circuits) + n_anchor if include_warmstart else 0
    )
    energy_evals = 0
    gradient_evals = 0
    topup_evals = 0
    for rec in history:
        energy_evals += 1
        gradient_evals += int(rec.get("grad_evals", 1)) * grad_evals_cost
        # ``n_topups`` counts every batch this iter (top-up-until-satisfied can fire
        # several); fall back to the 0/1 ``topped_up`` flag for legacy histories.
        n_topups = int(rec.get("n_topups", rec.get("topped_up", 0)))
        topup_evals += n_topups * int(config.topup_batch_size)

    breakdown = {
        "warm_start": warmstart_evals,
        "energy": energy_evals,
        "gradient": gradient_evals,
        "topup": topup_evals,
    }
    circuit_evals = warmstart_evals + energy_evals + gradient_evals + topup_evals
    return {
        "shots_per_circuit": shots,
        "circuit_evals": int(circuit_evals),
        "circuit_evals_breakdown": breakdown,
        "total_shots": int(circuit_evals * shots),
        "total_shots_breakdown": {k: int(v * shots) for k, v in breakdown.items()},
        "n_iterations": len(history),
        "n_params": int(config.n_params),
    }


def run_noiseless_reference_vqe(
    adapter: CirqBackendAdapter,
    theta_init: np.ndarray,
    *,
    max_iters: int,
    learning_rate: float = 0.5,
    gradient_mode: str = "parameter_shift",
    fd_eps: float = 0.05,
) -> dict:
    """Decoupled PURE-noiseless optimizer reference.

    Mirrors ``_build_noiseless_reference_curve`` in ``June_main/main_HF.ipynb``:
    a fixed-learning-rate gradient descent on the EXACT statevector energy
    (``adapter.ideal_energy``) via the parameter-shift rule, with NO noise, NO
    shots, and NO mitigation. This is the curve that should reach chemical
    accuracy; it is independent of the noisy/mitigated optimization so it is an
    honest "what the ansatz+optimizer can do" baseline.

    Returns per-iteration ideal energies and thetas, indexed ``0..max_iters``
    (entry 0 is ``theta_init`` before any update).
    """
    mode = str(gradient_mode).strip().lower()
    if mode not in {"parameter_shift", "finite_difference"}:
        raise ValueError(
            f"gradient_mode must be 'parameter_shift' or 'finite_difference', got {gradient_mode!r}."
        )
    shift = float(np.pi) / 2.0 if mode == "parameter_shift" else float(fd_eps)
    grad_scale = 0.5 if mode == "parameter_shift" else 1.0 / (2.0 * float(fd_eps))

    theta = np.asarray(theta_init, dtype=float).ravel()
    energies = [float(adapter.ideal_energy(theta))]
    thetas = [theta.copy()]
    for _ in range(int(max_iters)):
        grad = np.zeros_like(theta)
        for j in range(len(theta)):
            tp = theta.copy()
            tm = theta.copy()
            tp[j] += shift
            tm[j] -= shift
            grad[j] = (adapter.ideal_energy(tp) - adapter.ideal_energy(tm)) * grad_scale
        theta = theta - float(learning_rate) * grad
        energies.append(float(adapter.ideal_energy(theta)))
        thetas.append(theta.copy())
    return {
        "energies": energies,
        "thetas": thetas,
        "theta_final": theta,
        "energy_final": float(energies[-1]),
    }


# ---------------------------------------------------------------------------
# Section 10 - Validation (run BEFORE the live loop)
# ---------------------------------------------------------------------------


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def holdout_validation(
    adapter: CirqBackendAdapter, config: MitigatorConfig, *, seed: int | None = None
) -> dict:
    """Hold out a fraction of warm-start circuits; compare per-row prediction of
    O_ideal by: unmitigated, plain CDR (baseline), and the combined single GP.
    """
    seed = int(config.rng_seed if seed is None else seed)
    theta0 = (
        np.zeros(int(config.n_params))
        if config.theta_init is None
        else np.asarray(config.theta_init, dtype=float)
    )
    resolvers = adapter.generate_near_clifford(
        theta0,
        n_circuits=int(config.n_warmstart_circuits),
        n_nonclifford=config.n_nonclifford_gates,
        snap_step=float(config.clifford_snap_step),
        spread=float(config.warmstart_spread),
        seed=seed,
        node_hops=int(config.clifford_node_hops),
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(resolvers))
    n_hold = max(1, int(round(config.holdout_fraction * len(resolvers))))
    hold_set = set(order[:n_hold].tolist())
    train_res = [resolvers[i] for i in range(len(resolvers)) if i not in hold_set]
    hold_res = [resolvers[i] for i in range(len(resolvers)) if i in hold_set]

    train_rows = adapter.collect_rows(train_res, shots=int(config.shots), seed_base=seed + 1)
    hold_rows = adapter.collect_rows(hold_res, shots=int(config.shots), seed_base=seed + 5000)

    backbone, gp = _fit_backbone_and_gp(train_rows, config)
    gp_pred, _ = _predict_rows(gp, backbone, hold_rows, config)
    o_ideal = np.asarray([r["o_ideal"] for r in hold_rows], dtype=float)
    o_noisy = np.asarray([r["o_noisy"] for r in hold_rows], dtype=float)
    backbone_pred = _backbone_values(backbone, hold_rows)

    return {
        "n_train_circuits": len(train_res),
        "n_holdout_circuits": len(hold_res),
        "n_holdout_rows": len(hold_rows),
        "rmse_unmitigated": _rmse(o_noisy, o_ideal),
        "rmse_cdr_only": _rmse(backbone_pred, o_ideal),
        "rmse_single_gp": _rmse(gp_pred, o_ideal),
    }


def extrapolation_validation(
    adapter: CirqBackendAdapter,
    config: MitigatorConfig,
    *,
    train_spread: float = 0.3,
    test_spread: float = 0.8,
    seed: int | None = None,
) -> dict:
    """Train on a tight angle sub-range, test on a wider one, to measure the
    near-Clifford -> arbitrary-angle generalization gap (the real risk).
    """
    seed = int(config.rng_seed if seed is None else seed)
    theta0 = (
        np.zeros(int(config.n_params))
        if config.theta_init is None
        else np.asarray(config.theta_init, dtype=float)
    )

    train_res = adapter.generate_near_clifford(
        theta0,
        n_circuits=int(config.n_warmstart_circuits),
        n_nonclifford=config.n_nonclifford_gates,
        snap_step=float(config.clifford_snap_step),
        spread=float(train_spread),
        seed=seed,
        node_hops=int(config.clifford_node_hops),
    )
    test_res = adapter.generate_near_clifford(
        theta0,
        n_circuits=max(8, int(config.topup_batch_size)),
        n_nonclifford=config.n_nonclifford_gates,
        snap_step=float(config.clifford_snap_step),
        spread=float(test_spread),
        seed=seed + 999,
        node_hops=int(config.clifford_node_hops),
    )
    train_rows = adapter.collect_rows(train_res, shots=int(config.shots), seed_base=seed + 1)
    test_rows = adapter.collect_rows(test_res, shots=int(config.shots), seed_base=seed + 7000)

    backbone, gp = _fit_backbone_and_gp(train_rows, config)
    gp_pred, gp_std = _predict_rows(gp, backbone, test_rows, config)
    o_ideal = np.asarray([r["o_ideal"] for r in test_rows], dtype=float)
    o_noisy = np.asarray([r["o_noisy"] for r in test_rows], dtype=float)
    backbone_pred = _backbone_values(backbone, test_rows)
    return {
        "train_spread": float(train_spread),
        "test_spread": float(test_spread),
        "rmse_unmitigated": _rmse(o_noisy, o_ideal),
        "rmse_cdr_only": _rmse(backbone_pred, o_ideal),
        "rmse_single_gp": _rmse(gp_pred, o_ideal),
        "mean_gp_std": float(np.mean(gp_std)),
    }


def nnc_generalization_validation(
    adapter: CirqBackendAdapter,
    config: MitigatorConfig,
    *,
    nnc_train: "int | tuple[int, ...] | None" = None,
    nnc_test: "int | None" = None,
    n_test_circuits: int | None = None,
    seed: int | None = None,
) -> dict:
    """Train with few non-Clifford gates per circuit, test on FULLY non-Clifford
    thetas -- exactly the regime the live VQE operates in.

    ``nnc_train`` defaults to ``config.n_nonclifford_gates``; ``nnc_test``
    defaults to ``n_params`` (all parameters jittered, no snapping), which is
    what a real VQE theta looks like. This measures the classical-model
    degeneracy directly: a model that only interpolates ``O_ideal(theta)`` from
    axis-aligned slices fails here, while one that leans on ``O_noisy``
    (backbone prior / in-kernel CDR) degrades gracefully.
    """
    seed = int(config.rng_seed if seed is None else seed)
    theta0 = (
        np.zeros(int(config.n_params))
        if config.theta_init is None
        else np.asarray(config.theta_init, dtype=float)
    )
    nnc_train = config.n_nonclifford_gates if nnc_train is None else nnc_train
    nnc_test = int(config.n_params) if nnc_test is None else int(nnc_test)
    n_test = int(
        max(8, config.topup_batch_size) if n_test_circuits is None else n_test_circuits
    )

    train_res = adapter.generate_near_clifford(
        theta0,
        n_circuits=int(config.n_warmstart_circuits),
        n_nonclifford=nnc_train,
        snap_step=float(config.clifford_snap_step),
        spread=float(config.warmstart_spread),
        seed=seed,
        node_hops=int(config.clifford_node_hops),
    )
    test_res = adapter.generate_near_clifford(
        theta0,
        n_circuits=n_test,
        n_nonclifford=nnc_test,
        snap_step=float(config.clifford_snap_step),
        spread=float(config.warmstart_spread),
        seed=seed + 4242,
        node_hops=0,
    )
    train_rows = adapter.collect_rows(train_res, shots=int(config.shots), seed_base=seed + 1)
    test_rows = adapter.collect_rows(test_res, shots=int(config.shots), seed_base=seed + 9000)

    backbone, gp = _fit_backbone_and_gp(train_rows, config)
    gp_pred, gp_std = _predict_rows(gp, backbone, test_rows, config)
    o_ideal = np.asarray([r["o_ideal"] for r in test_rows], dtype=float)
    o_noisy = np.asarray([r["o_noisy"] for r in test_rows], dtype=float)
    backbone_pred = _backbone_values(backbone, test_rows)
    return {
        "nnc_train": nnc_train,
        "nnc_test": nnc_test,
        "n_test_circuits": n_test,
        "rmse_unmitigated": _rmse(o_noisy, o_ideal),
        "rmse_cdr_only": _rmse(backbone_pred, o_ideal),
        "rmse_single_gp": _rmse(gp_pred, o_ideal),
        "mean_gp_std": float(np.mean(gp_std)),
    }


def measurement_reliance_report(
    mitigator: Mitigator,
    rows: "list[dict] | None" = None,
    *,
    max_rows: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """How much does the mitigated prediction actually USE the noisy measurement?

    Two complementary probes on (a sample of) the training rows:

      1. gradient probe -- mean ``|d mean / d o_noisy|`` from the analytic GP
         input gradient, plus the backbone slope ``|a_P|`` when the backbone
         prior mean is active (the always-on measurement channel).
      2. permutation probe -- re-predict the same rows with ``o_noisy`` SHUFFLED
         within each Pauli group. If the predictions barely move, the model is
         ignoring the measurement: it has degenerated into a classical surrogate
         of O_ideal(theta) (the nnc=1 failure mode).

    ``degenerate`` is True when the permutation-induced RMS prediction change is
    < 5% of the target scale.
    """
    config = mitigator.config
    rows = list(mitigator.rows if rows is None else rows)
    if not rows:
        raise ValueError("No rows available; run warmstart() first.")
    rng = np.random.default_rng(int(seed))
    if len(rows) > int(max_rows):
        pick = rng.choice(len(rows), size=int(max_rows), replace=False)
        rows = [rows[i] for i in sorted(pick.tolist())]

    # (1) gradient probe.
    idx = feature_index_map(config)
    noisy_col = idx["noisy"][0] if idx["noisy"] else None
    gp_grad_reliance = 0.0
    if noisy_col is not None:
        X = np.stack([build_feature_row(r, config) for r in rows], axis=0)
        dmean = mitigator.gp.predict_input_gradient(X)
        gp_grad_reliance = float(np.mean(np.abs(dmean[:, noisy_col])))
    backbone_slope = 0.0
    if _use_backbone_prior(config):
        slopes = [
            abs(float(mitigator.backbone.coeffs.get(r["pauli"], (1.0, 0.0))[0]))
            for r in rows
        ]
        backbone_slope = float(np.mean(slopes))
    total_reliance = gp_grad_reliance + backbone_slope

    # (2) permutation probe: shuffle o_noisy within each Pauli group.
    mean_orig, _ = _predict_rows(mitigator.gp, mitigator.backbone, rows, config)
    by_pauli: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_pauli.setdefault(r["pauli"], []).append(i)
    rows_perm = [dict(r) for r in rows]
    for label, idxs in by_pauli.items():
        if len(idxs) < 2:
            continue
        perm = rng.permutation(idxs)
        for i_dst, i_src in zip(idxs, perm.tolist()):
            rows_perm[i_dst]["o_noisy"] = float(rows[i_src]["o_noisy"])
    mean_perm, _ = _predict_rows(mitigator.gp, mitigator.backbone, rows_perm, config)
    perm_delta_rms = float(np.sqrt(np.mean((mean_perm - mean_orig) ** 2)))
    target_scale = float(np.std([r["o_ideal"] for r in rows]) + 1e-12)
    perm_ratio = perm_delta_rms / target_scale
    degenerate = bool(perm_ratio < 0.05)

    out = {
        "gp_grad_reliance": gp_grad_reliance,
        "backbone_slope_mean": backbone_slope,
        "total_reliance": total_reliance,
        "perm_delta_rms": perm_delta_rms,
        "target_scale": target_scale,
        "perm_ratio": perm_ratio,
        "degenerate": degenerate,
        "n_rows_probed": len(rows),
    }
    if verbose:
        print(
            f"[reliance] mean |d mean/d o_noisy| (GP)   : {gp_grad_reliance:.4f}\n"
            f"[reliance] mean backbone slope |a_P|      : {backbone_slope:.4f}\n"
            f"[reliance] total measurement reliance     : {total_reliance:.4f}\n"
            f"[reliance] permutation RMS shift / scale  : {perm_delta_rms:.4e} / "
            f"{target_scale:.4e} = {perm_ratio:.2%}"
        )
        if degenerate:
            print(
                "[reliance] WARNING: predictions barely respond to o_noisy -- the "
                "model has degenerated into a CLASSICAL surrogate of O_ideal(theta). "
                "Use prior_mean_mode='backbone', lower angle_term_amplitude_bound, "
                "or add grid-diverse training data (clifford_node_hops > 0)."
            )
    return out


def make_eval_set(
    adapter: CirqBackendAdapter,
    center: np.ndarray,
    *,
    radius: float,
    n: int,
    seed: int,
    shots: int,
) -> dict:
    """Build a fixed evaluation set of thetas around ``center`` (the region a VQE
    would explore), with their exact ideal energies and a single noisy measurement
    each. Reused across configs so model comparisons are apples-to-apples and no
    extra circuits are spent per config at eval time.
    """
    center = np.asarray(center, dtype=float).ravel()
    rng = np.random.default_rng(int(seed))
    thetas: list[np.ndarray] = []
    ideal_energies: list[float] = []
    measured_list: list[dict] = []
    for i in range(int(n)):
        theta = center + rng.uniform(-radius, radius, size=center.shape[0])
        thetas.append(theta)
        ideal_energies.append(float(adapter.ideal_energy(theta)))
        measured_list.append(
            adapter.run_noisy(
                adapter.resolver_from_theta(theta),
                shots=int(shots),
                sampling_seed=int(seed) + 100_000 + i,
            )["primary"]
        )
    return {
        "thetas": thetas,
        "ideal_energies": ideal_energies,
        "measured_list": measured_list,
    }


def evaluate_mitigator_on_thetas(mitigator: Mitigator, eval_set: dict) -> dict:
    """Energy-level MAE over the eval set for: unmitigated, backbone-only, full."""
    adapter = mitigator.adapter
    full_err: list[float] = []
    bb_err: list[float] = []
    noisy_err: list[float] = []
    unc_list: list[float] = []
    for theta, e_ideal, measured in zip(
        eval_set["thetas"], eval_set["ideal_energies"], eval_set["measured_list"]
    ):
        o_mit, std = mitigator.predict_with_uncertainty(theta, measured)
        e_full = adapter.energy_from_values(o_mit)
        bb = {lab: mitigator.backbone.apply(lab, measured[lab]) for lab in measured}
        e_bb = adapter.energy_from_values(bb)
        e_noisy = adapter.energy_from_values(measured)
        full_err.append(abs(e_full - e_ideal))
        bb_err.append(abs(e_bb - e_ideal))
        noisy_err.append(abs(e_noisy - e_ideal))
        unc_list.append(weighted_uncertainty(std, mitigator.coeff_by_pauli))
    return {
        "mae_full": float(np.mean(full_err)),
        "mae_backbone": float(np.mean(bb_err)),
        "mae_noisy": float(np.mean(noisy_err)),
        "median_weighted_uncertainty": float(np.median(unc_list)),
        "warmstart_circuits": int(mitigator.config.n_warmstart_circuits),
    }


def clifford_anchor_check(
    mitigator: Mitigator, theta_clifford: np.ndarray | None = None
) -> dict:
    """A fully-Clifford ``theta`` has a classically-known exact energy; confirm the
    mitigated pipeline lands on it.
    """
    cfg = mitigator.config
    adapter = mitigator.adapter
    if theta_clifford is None:
        base = (
            np.zeros(int(cfg.n_params))
            if cfg.theta_init is None
            else np.asarray(cfg.theta_init, dtype=float)
        )
        theta_clifford = np.asarray(
            [_snap_to_clifford(s, float(base[j]), cfg.clifford_snap_step)
             for j, s in enumerate(adapter.symbols)],
            dtype=float,
        )
    bundle = adapter.run_noisy(
        adapter.resolver_from_theta(theta_clifford),
        shots=int(cfg.shots),
        sampling_seed=int(cfg.rng_seed) + 31,
    )
    measured = bundle["primary"]
    e_mit = mitigator.mitigated_energy(theta_clifford, measured)
    e_ideal = adapter.ideal_energy(theta_clifford)
    e_noisy = adapter.energy_from_values(measured)
    return {
        "theta_clifford": np.asarray(theta_clifford, dtype=float),
        "energy_ideal": float(e_ideal),
        "energy_noisy": float(e_noisy),
        "energy_mitigated": float(e_mit),
        "abs_error_mitigated": float(abs(e_mit - e_ideal)),
        "abs_error_noisy": float(abs(e_noisy - e_ideal)),
    }
