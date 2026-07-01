#!/usr/bin/env python3
"""Verify the GP-mitigator shot accounting: ANALYTICAL == NUMERICAL.

Every physical circuit execution in the combined CDR+GP mitigator goes through
``CirqBackendAdapter.run_noisy``, which spends ``config.shots`` shots. The total
hardware cost is therefore::

    total_shots = shots * (warm_start
                           + sum_iter[ 1 (energy)
                                       + grad_evals * 2 * n_params (gradient)
                                       + n_topups * topup_batch_size (top-up) ])

Two independent counts must agree:

  * NUMERICAL  -- read straight from the backend's ``run_noisy`` call counter,
  * ANALYTICAL -- ``gp_mitigator_ML_tog.analytic_shot_count`` (closed form), and
                  a hand-written expected value per scenario.

To stay fast and dependency-light this test drives the REAL optimizer loop
(``run_vqe_with_mitigator``) with tiny fakes for the quantum backend and the GP
so the control flow (and thus the shot count) is exactly the production one,
without cirq / shadowgrouping / OGM. Each scenario exercises a different top-up
trigger: none, uncertainty, periodic, trust-region (move), and the
convergence-validation early stop.

Run directly (``python test_gp_shot_accounting.py``) or via pytest.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_TOG_DIR = REPO_ROOT / "ML_CDR" / "ML_CDR_together"
if str(ML_TOG_DIR) not in sys.path:
    sys.path.insert(0, str(ML_TOG_DIR))

import gp_mitigator_ML_tog as gpm  # noqa: E402

LABELS = ["Z0", "Z1"]
SHOTS = 8192  # matches the notebook's GP_SHOTS (GLOBAL_NUM_SHOTS)


# ---------------------------------------------------------------------------
# Lightweight fakes: only the methods run_vqe_with_mitigator actually calls.
# run_noisy is the single chokepoint that the production counter increments, so
# routing every fake execution through it makes the NUMERICAL count authentic.
# ---------------------------------------------------------------------------
class FakeBackend:
    def __init__(self, shots: int) -> None:
        self.shots = int(shots)
        self.n_circuit_evals = 0
        self.total_shots = 0

    def reset_shot_counter(self) -> None:
        self.n_circuit_evals = 0
        self.total_shots = 0

    def shot_report(self) -> dict:
        return {
            "circuit_evaluations": int(self.n_circuit_evals),
            "total_shots": int(self.total_shots),
        }

    def resolver_from_theta(self, theta):
        return {"theta": np.asarray(theta, dtype=float)}

    def run_noisy(self, resolver, shots: int, sampling_seed: int) -> dict:
        # The one place shots are spent (exactly like the real adapter).
        self.n_circuit_evals += 1
        self.total_shots += int(shots)
        primary = {label: 0.0 for label in LABELS}
        return {
            "primary": primary,
            "unmit_by_pauli": dict(primary),
            "rem_by_pauli": dict(primary),
            "energy_unmitigated": 0.0,
            "energy_rem": 0.0,
        }

    def generate_near_clifford(self, theta, *, n_circuits, n_nonclifford, snap_step, spread, seed):
        # Real adapter returns one resolver per requested circuit.
        return [self.resolver_from_theta(theta) for _ in range(int(n_circuits))]

    def collect_rows(self, resolvers, *, shots: int, seed_base: int) -> list:
        rows = []
        for i, resolver in enumerate(resolvers):
            self.run_noisy(resolver, shots=shots, sampling_seed=seed_base + i)
            for label in LABELS:
                rows.append({"theta": resolver["theta"], "pauli": label,
                             "o_noisy": 0.0, "o_ideal": 0.0})
        return rows

    def energy_from_values(self, value_by_pauli: dict) -> float:
        return float(sum(value_by_pauli.values()))


class _Backbone:
    @staticmethod
    def apply(label, value):
        return float(value)


class FakeMitigator:
    """Drives the real loop. ``std_value`` and ``grad_fn`` make the top-up
    triggers and convergence deterministic and controllable."""

    def __init__(self, config, adapter, *, std_value=0.0, grad_fn=None) -> None:
        self.config = config
        self.adapter = adapter
        self.coeff_by_pauli = {label: 1.0 for label in LABELS}
        self.backbone = _Backbone()
        self.rows: list = []
        self._std_value = float(std_value)
        # grad_fn(theta) -> np.ndarray of length n_params (default: constant zero).
        self._grad_fn = grad_fn or (lambda th: np.zeros(len(th)))

    def predict_with_uncertainty(self, theta, o_noisy_by_pauli):
        o_mit = {label: float(o_noisy_by_pauli[label]) for label in o_noisy_by_pauli}
        std = {label: self._std_value for label in o_noisy_by_pauli}
        return o_mit, std

    def mitigate(self, theta, o_noisy_by_pauli):
        return {label: float(o_noisy_by_pauli[label]) for label in o_noisy_by_pauli}

    def update_with_rows(self, new_rows, current_theta=None):
        self.rows.extend(new_rows)

    def energy_gradient(self, theta, measured, do_noisy):
        return np.asarray(self._grad_fn(theta), dtype=float)


def _base_config(**overrides):
    cfg = gpm.MitigatorConfig(
        n_params=3,
        n_qubits=2,
        n_observables=len(LABELS),
        shots=SHOTS,
        n_warmstart_circuits=10,
        topup_batch_size=7,
        # All triggers OFF by default; each scenario flips exactly one on.
        uncertainty_threshold=1e9,
        topup_on_move=False,
        topup_every=0,
        convergence_grad_tol=0.0,
        topup_radius=0.1,
        optimizer_step_max=0.1,
    )
    return replace(cfg, **overrides)


def _run(cfg, *, std_value=0.0, grad_fn=None, max_iters, learning_rate=0.5,
         gradient_mode="analytic"):
    adapter = FakeBackend(cfg.shots)
    mitig = FakeMitigator(cfg, adapter, std_value=std_value, grad_fn=grad_fn)
    vqe_out = gpm.run_vqe_with_mitigator(
        mitig,
        theta_init=np.zeros(cfg.n_params),
        max_iters=max_iters,
        step_max=None,
        learning_rate=learning_rate,
        gradient_mode=gradient_mode,
        ideal_energy_fn=None,
        verbose=False,
    )
    return adapter, vqe_out


def _check(adapter, vqe_out, cfg, *, expected_topup_iters, expected_iters,
           expected_topup_batches=None):
    """Assert NUMERICAL == ANALYTICAL == hand formula (VQE only, no warm start).

    ``expected_topup_iters`` counts iterations that top up at all (the 0/1
    ``topped_up`` flag); ``expected_topup_batches`` counts the TOTAL number of
    top-up batches (``n_topups``), which can exceed the iteration count once
    top-up-until-satisfied fires several batches per iter. Defaults to equal.
    """
    if expected_topup_batches is None:
        expected_topup_batches = expected_topup_iters
    history = vqe_out["history"]
    n_params = cfg.n_params
    grad_cost = gpm.gradient_circuit_evals(cfg)
    assert grad_cost == 2 * n_params

    # grad_evals is 2 only on a convergence-validation iteration, else 1.
    grad_eval_units = sum(int(h["grad_evals"]) for h in history)
    hand_evals = (
        len(history)                                   # energy
        + grad_eval_units * grad_cost                  # gradient
        + expected_topup_batches * cfg.topup_batch_size  # top-ups (all batches)
    )
    hand_shots = hand_evals * cfg.shots

    # NUMERICAL (backend counter) -- run_vqe self-report and the meter must match.
    assert adapter.n_circuit_evals == vqe_out["circuit_evals"]
    assert adapter.total_shots == vqe_out["total_shots"]
    num_evals = adapter.n_circuit_evals
    num_shots = adapter.total_shots

    # ANALYTICAL (closed form from history, warm start excluded for VQE-only cmp).
    ana = gpm.analytic_shot_count(cfg, history, include_warmstart=False)

    assert len(history) == expected_iters, (len(history), expected_iters)
    assert sum(int(h["topped_up"]) for h in history) == expected_topup_iters
    assert sum(int(h["n_topups"]) for h in history) == expected_topup_batches
    assert num_evals == hand_evals, (num_evals, hand_evals)
    assert num_shots == hand_shots, (num_shots, hand_shots)
    assert ana["circuit_evals"] == num_evals, (ana["circuit_evals"], num_evals)
    assert ana["total_shots"] == num_shots, (ana["total_shots"], num_shots)
    return ana


def test_no_topups_and_warmstart_term():
    """No triggers: cost is purely energy + gradient. Also check warm-start term."""
    cfg = _base_config()
    n = 5
    adapter, vqe_out = _run(cfg, std_value=0.0, max_iters=n)
    ana = _check(adapter, vqe_out, cfg, expected_topup_iters=0, expected_iters=n)
    # energy + gradient only: n * (1 + 2*n_params)
    assert ana["circuit_evals"] == n * (1 + 2 * cfg.n_params)

    # The warm-start term is exactly n_warmstart_circuits extra executions.
    with_warm = gpm.analytic_shot_count(cfg, vqe_out["history"], include_warmstart=True)
    assert with_warm["circuit_evals"] - ana["circuit_evals"] == cfg.n_warmstart_circuits
    assert with_warm["circuit_evals_breakdown"]["warm_start"] == cfg.n_warmstart_circuits
    assert with_warm["total_shots"] == with_warm["circuit_evals"] * cfg.shots


def test_uncertainty_triggered_topup_every_iter():
    """Large predicted std -> the uncertainty gate fires on every iteration."""
    cfg = _base_config(uncertainty_threshold=0.01)
    n = 4
    # weighted unc = sum |coeff|*std = 2 * 0.5 = 1.0 > 0.01 -> always tops up.
    adapter, vqe_out = _run(cfg, std_value=0.5, max_iters=n)
    _check(adapter, vqe_out, cfg, expected_topup_iters=n, expected_iters=n)


def test_periodic_topups():
    """Mandatory top-up every k iters (it>0 and it%k==0)."""
    k = 3
    cfg = _base_config(topup_every=k)
    n = 10
    adapter, vqe_out = _run(cfg, std_value=0.0, max_iters=n)
    expected = sum(1 for it in range(n) if it > 0 and it % k == 0)  # 3,6,9 -> 3
    _check(adapter, vqe_out, cfg, expected_topup_iters=expected, expected_iters=n)


def test_trust_region_move_topups():
    """Constant gradient -> constant step; theta leaves the trust region each
    iter after the first, firing a move-triggered top-up on iters 1..n-1."""
    cfg = _base_config(topup_on_move=True, topup_move_fraction=1.0,
                       topup_radius=0.1)
    n = 6
    # step = -lr*grad = -0.5*[0.4,0,0] -> |step| = 0.2 > radius(0.1).
    grad_fn = lambda th: np.array([0.4, 0.0, 0.0])
    adapter, vqe_out = _run(cfg, std_value=0.0, grad_fn=grad_fn, max_iters=n,
                            learning_rate=0.5)
    _check(adapter, vqe_out, cfg, expected_topup_iters=n - 1, expected_iters=n)


def test_convergence_validation_early_stop():
    """Quadratic bowl: |grad| shrinks until the convergence-validation top-up
    fires, the energy is stable, so the loop stops EARLY. The validation iter
    does a second gradient eval (grad_evals=2) and one extra top-up."""
    cfg = _base_config(convergence_grad_tol=0.05, convergence_energy_tol=1e-6)
    theta_min = np.array([1.0, 1.0, 1.0])
    grad_fn = lambda th: (np.asarray(th, float) - theta_min)  # K=1
    n_budget = 50
    adapter, vqe_out = _run(cfg, std_value=0.0, grad_fn=grad_fn,
                            max_iters=n_budget, learning_rate=0.5)
    history = vqe_out["history"]

    # Must have stopped before exhausting the budget, and flagged converged.
    assert vqe_out["converged"] is True
    assert len(history) < n_budget

    # Exactly one convergence-validation iteration: the last one, with 2 grad evals
    # and a single (validation) top-up; earlier iters have none.
    last = history[-1]
    assert last["grad_evals"] == 2
    assert last["topped_up"] == 1
    assert sum(int(h["topped_up"]) for h in history) == 1
    assert all(h["grad_evals"] == 1 for h in history[:-1])

    # NUMERICAL == ANALYTICAL on the (shortened) history.
    _check(adapter, vqe_out, cfg, expected_topup_iters=1, expected_iters=len(history))


def test_topup_until_uncertainty_satisfied():
    """top-up-until-satisfied: with std stuck above the threshold every triggered
    iteration fires 1 initial + max_topup_retries extra batches (bounded cap)."""
    retries = 3
    cfg = _base_config(uncertainty_threshold=0.1, max_topup_retries=retries)
    n = 4
    # weighted unc = 2 * 0.5 = 1.0 > 0.1 on every iter (incl. iter 0 via the
    # uncertainty gate); the fake std never drops so the retry loop hits the cap.
    adapter, vqe_out = _run(cfg, std_value=0.5, max_iters=n)
    per_iter = 1 + retries  # initial top-up + capped retries
    _check(adapter, vqe_out, cfg, expected_topup_iters=n, expected_iters=n,
           expected_topup_batches=n * per_iter)
    # Every iteration recorded the full batch count.
    assert all(int(h["n_topups"]) == per_iter for h in vqe_out["history"])


def test_parameter_shift_matches_analytic_mode_cost():
    """Gradient cost is mode-independent: parameter-shift costs the same 2*n_params
    circuits per gradient as the analytic mode."""
    cfg = _base_config()
    n = 4
    a_adapter, a_out = _run(cfg, max_iters=n, gradient_mode="analytic")
    p_adapter, p_out = _run(cfg, max_iters=n, gradient_mode="parameter_shift")
    assert a_adapter.total_shots == p_adapter.total_shots
    assert a_out["circuit_evals"] == p_out["circuit_evals"]
    _check(p_adapter, p_out, cfg, expected_topup_iters=0, expected_iters=n)


SCENARIOS = [
    ("no top-ups", test_no_topups_and_warmstart_term),
    ("uncertainty top-ups", test_uncertainty_triggered_topup_every_iter),
    ("periodic top-ups", test_periodic_topups),
    ("trust-region top-ups", test_trust_region_move_topups),
    ("convergence early stop", test_convergence_validation_early_stop),
    ("top-up until uncertainty satisfied", test_topup_until_uncertainty_satisfied),
    ("parameter-shift == analytic cost", test_parameter_shift_matches_analytic_mode_cost),
]


def main() -> int:
    failures = 0
    for name, fn in SCENARIOS:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as exc:  # pragma: no cover - reporting path
            failures += 1
            print(f"[FAIL] {name}: {exc}")
    if failures:
        print(f"\n{failures}/{len(SCENARIOS)} scenarios FAILED")
        return 1
    print(f"\nAll {len(SCENARIOS)} scenarios passed: analytical == numerical shot counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
