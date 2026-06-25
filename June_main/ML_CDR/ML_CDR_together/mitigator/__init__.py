"""Single-GP error mitigator with CDR absorbed in the kernel (Design 2).

One Gaussian Process maps (angle features, o_noisy, Pauli features) -> o_ideal.
The classic CDR line a*o_noisy + b is represented INSIDE the GP by a linear kernel
on the o_noisy feature; an RBF/Matern term on top captures coherent error.
"""

from __future__ import annotations

from .adapters import CirqCDRAdapter, PauliObservable, QuantumBackendAdapter
from .config import MitigatorConfig
from .features import (
    build_feature_matrix,
    build_feature_row,
    encode_angles,
    encode_pauli,
    feature_index_map,
)
from .gp_model import OnColumns, SingleGPMitigatorModel
from .mitigator import Mitigator
from .topup import aggregate_uncertainty, energy_std, needs_topup, sample_local_rows
from .vqe_loop import measure_noisy, mitigated_energy, run_vqe
from . import validation

__all__ = [
    "MitigatorConfig",
    "QuantumBackendAdapter",
    "CirqCDRAdapter",
    "PauliObservable",
    "encode_angles",
    "encode_pauli",
    "build_feature_row",
    "build_feature_matrix",
    "feature_index_map",
    "OnColumns",
    "SingleGPMitigatorModel",
    "Mitigator",
    "needs_topup",
    "sample_local_rows",
    "aggregate_uncertainty",
    "energy_std",
    "run_vqe",
    "measure_noisy",
    "mitigated_energy",
    "validation",
]
