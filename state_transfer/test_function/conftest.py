"""Shared fixtures for the state_transfer tapering tests.

Run from the repo root with the tencirchem venv:

    .venv_h4_tencirchem/bin/python -m pytest state_transfer/test_function -v

Everything defaults to HF (active space (6, 4), 8 -> 6 qubits). The fixtures
are session-scoped so the pyscf Hamiltonian builds run only once per session.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_TRANSFER = REPO_ROOT / "state_transfer"
JUNE_MAIN = REPO_ROOT / "June_main"
UCCSD_DIR = REPO_ROOT / "UCCSD circuit"

for _p in (str(STATE_TRANSFER), str(JUNE_MAIN), str(UCCSD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MOLECULE = "HF"
TEST_BONDS = (1.0, 1.4)  # keep pyscf runs cheap; extend if needed


def load_circuit_generator_module():
    """Import 'UCCSD circuit/improved create UCCSD circuit .py' (space-named)."""
    path = UCCSD_DIR / "improved create UCCSD circuit .py"
    spec = importlib.util.spec_from_file_location("improved_create_uccsd_circuit", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hf_state_vector(n_qubits: int, occupied) -> np.ndarray:
    """Computational-basis determinant (openfermion qubit-0 = MSB ordering)."""
    index = 0
    for q in occupied:
        index |= 1 << (n_qubits - 1 - q)
    vec = np.zeros(2**n_qubits, dtype=complex)
    vec[index] = 1.0
    return vec


@pytest.fixture(scope="session")
def gm():
    import generate_molecular_hamiltonians as gm_module

    return gm_module


@pytest.fixture(scope="session")
def taper_lib():
    import taper_lib as tl

    return tl


@pytest.fixture(scope="session")
def gen():
    return load_circuit_generator_module()


@pytest.fixture(scope="session")
def taper(taper_lib, gm):
    preset = gm.MOLECULE_PRESETS[MOLECULE]
    return taper_lib.build_taper_data(
        n_spatial=preset.active_space[1], n_electrons=preset.active_space[0]
    )


@pytest.fixture(scope="session")
def hamiltonians(gm, taper_lib, taper):
    """{bond: (full_operator, tapered_operator)} for TEST_BONDS."""
    out = {}
    for bond in TEST_BONDS:
        full_op, _meta = gm.build_molecular_hamiltonian(MOLECULE, bond)
        out[bond] = (full_op, taper_lib.taper_qubit_operator(full_op, taper))
    return out
