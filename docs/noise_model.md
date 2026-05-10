# Noise model reference

This document matches the implementation in `main_cursor_lib.py` (`LocationAwareDecomposedNoise`, FSim decomposition, ZNE scaling) and shot-level readout in `shot_measurement.py`. Use it to sanity-check what “noisy simulation” means in this repo.

## Calibration note (heuristic, not device calibration)

Published **Google Weber / Sycamore** typical characterization (May 2021 datasheet, median column) includes roughly **~0.1%** single-qubit RB error per gate, **~0.9%** isolated two-qubit (√iSWAP) XEB error per gate, and **~15 µs** relaxation \(T_1\) ([Weber datasheet PDF](https://quantumai.google/hardware/datasheet/weber.pdf)).

This repo does **not** simulate native √iSWAP or Google’s pulse-level physics; it applies **Kraus channels after each decomposed `CZ` / single-qubit gate**. The scalar defaults below (`DEFAULT_*` in `main_cursor_lib.py`) are chosen only so that **composite severity is in the same ballpark** as those Weber medians. **They are not fitted** to your chemistry ansatz or to a specific calibration run.

With **larger** base rates, ZNE’s `scale_noise_params_for_zne` hits **clipping at 1** sooner for large `noise_scale`. Treat high scales as requiring sanity checks on fits.

## 1. Ansatz shape

The chemistry ansatz is built with `prepare_decomposed_ansatz_cirq`: each parameterized **fSIM** interaction is expanded into **single-qubit rotations** plus **tagged `CZ` gates** (`decompose_fsim_theta_symbolic`, `decompose_fsim_phi_symbolic`). Noise is inserted **after** native gates by Cirq’s `circuit.with_noise(model)`, so every appearance of a gate in that decomposition can acquire noise.

## 2. `LocationAwareDecomposedNoise`

Class: `LocationAwareDecomposedNoise` in `main_cursor_lib.py`.

**Base parameters** (module constants `DEFAULT_AMP_DAMP_GAMMA`, `DEFAULT_PHASE_DAMP_GAMMA`, `DEFAULT_DEPOL_PROB`, `DEFAULT_LEAKAGE_APPROX_PROB`, `DEFAULT_HIGH_CZ_MULTIPLIER` in `main_cursor_lib.py`; same values are used by the layer-3 sweep and scripts):

| Parameter | Default | Role |
|-----------|---------|------|
| `amp_damp_gamma` | **`2e-3`** (`DEFAULT_AMP_DAMP_GAMMA`) | Amplitude damping strength baseline |
| `phase_damp_gamma` | **`3e-3`** (`DEFAULT_PHASE_DAMP_GAMMA`) | Phase damping strength baseline |
| `depol_prob` | **`7e-3`** (`DEFAULT_DEPOL_PROB`) | Depolarizing probability baseline |
| `leakage_approx_prob` | **`6e-3`** (`DEFAULT_LEAKAGE_APPROX_PROB`) | **Extra** depolarization on “high” CZs only |
| `high_cz_multiplier` | **`5.0`** (`DEFAULT_HIGH_CZ_MULTIPLIER`) | Multiplier on AD/PD/depol for **high** horizontal CZs |

**Rules per gate type:**

- **`MeasurementGate`**: unchanged (no noise appended).
- **`CZPowGate`** (includes all tagged CZs from the decomposition): emit the CZ, then **on each qubit** append amplitude damping, phase damping, and depolarizing channels.
  - If the operation is tagged with `CZ_HIGH_TAG` (`"cz_high"`): multiply `amp_damp_gamma`, `phase_damp_gamma`, and `depol_prob` by `high_cz_multiplier`, and add `leakage_approx_prob` to the **depolarizing** probability (then clip to ≤ 1).
  - Otherwise: use the base rates with multiplier `1`.
- **Single-qubit gates** (`PhasedXPowGate`, `ZPowGate`, `H`, etc.): emit the gate, then append AD / PD / depol with strengths **divided by 10** relative to the base (`gamma/10`, `depol_prob/10`).
- **Other multi-qubit gates**: passed through with **no** appended noise (the decomposed ansatz should not rely on this path for fSIM).

Regression coverage: `test function/test_noise_model_implementation.py`.

## 3. Which CZs are “high”?

Horizontal nearest-neighbor pairs on the **same row** use `cz_tag_for_horizontal_pair`:

- For an edge between columns `c` and `c+1`, let `m = min(c, c+1)`. If `m` is **odd**, the tag is `"cz_high"`; if `m` is **even**, `"cz_normal"`.

So on each row of spatial qubits, edges alternate normal / high / normal (for 4 columns: edges 0–1 normal, 1–2 high, 2–3 normal). The same rule applies to the second row of the Jordan–Wigner ladder.

**Onsite** blocks use `CZ_ONSITE_TAG` (`"cz_onsite_normal"`). That tag is **not** `cz_high`, so those CZs use the **standard** (non-multiplied) CZ noise.

## 4. ZNE noise scaling (`scale_noise_params_for_zne`)

For trace ZNE and density-matrix simulations at scaled noise, the code **does not** fake noise by arbitrary factors on the circuit; it builds a new `LocationAwareDecomposedNoise` with **scaled channel strengths**:

- Multiply `amp_damp_gamma`, `phase_damp_gamma`, `depol_prob`, and `leakage_approx_prob` by `noise_scale`, then **clip each to `[0, 1]`**.
- **`high_cz_multiplier` is not scaled** (still applies on top of the scaled base rates at each CZ).

Default ZNE scales in the sweep: `[1.0, 2.0, 3.0]` with linear extrapolation to zero (`fit_order=1`).

## 5. Shot noise and readout

Shot estimation uses a density-matrix simulator with the same noise model, then Pauli measurements / OGM schemes as configured. **Readout error** is optional (`apply_readout_noise`): classical bit flips using per-qubit success probabilities `p_0`, `p_1` (see `apply_asymmetric_readout_noise` in `shot_measurement.py`). That layer is **independent** of the continuous channel parameters above.

## 6. Figures

Running:

```bash
python docs/scripts/plot_noise_model_reference.py --out-dir docs/figures/noise_model
```

writes PNGs that summarize ZNE-scaled strengths and the horizontal-edge tagging pattern for the H4 (4 spatial sites) layout.
