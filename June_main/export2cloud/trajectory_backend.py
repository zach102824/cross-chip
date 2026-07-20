#!/usr/bin/env python3
"""Enable Monte Carlo trajectory noisy simulation for cloud export mains."""

from __future__ import annotations

import os


def install_trajectory_backend() -> None:
    """Select the trajectory noisy-sim backend via ``NOISY_SIM_BACKEND``.

    Existing ``main_*.py`` scripts keep calling ``run_mitigation`` /
    ``estimate_noisy_shots_for_resolver``; this only switches the inner
    density-matrix path to independent statevector trajectories that realize
    the same ``GateArityDepolarizingNoise`` channel.
    """
    os.environ["NOISY_SIM_BACKEND"] = "trajectory"
    print("[trajectory] NOISY_SIM_BACKEND=trajectory (no DensityMatrixSimulator on hot path)")
