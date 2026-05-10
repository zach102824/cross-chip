# Cross Chips Sim: Noisy VQE + OGM Procedure

This repository evaluates a fixed-parameter H4 ansatz using:

- a decomposed `CZ + 1Q` circuit representation,
- a location-aware noise model,
- finite-shot Hamiltonian estimation (including OGM),
- and readout error mitigation (REM).

The main workflow is in `main_cursor.ipynb`.

## End-to-End Procedure (Notebook Flow)

`main_cursor.ipynb` executes this pipeline:

1. **Configure problem and runtime options**
  - Set molecular/ansatz settings (`H_atom`, `bond_length`, `num_spatial`, `ansatz_layers`).
  - Provide fixed `vqe_parameters`.
  - Choose measurement setup (`measurement_scheme`, `num_shots`, `sampling_seed`).
  - Set readout calibration vectors (`p_0_success`, `p_1_success`) and booleans (`apply_readout_noise`, `apply_rem`).
2. **Build ansatz + Hamiltonian**
  - `prepare_decomposed_ansatz_cirq(...)` builds the decomposed circuit.
  - `load_observable_h(...)` loads `H` from `Pauli_Ham` (`.pkl` preferred, text fallback).
3. **Inject noise**
  - Create `LocationAwareDecomposedNoise(...)`.
  - Apply with `ansatz_circuit.with_noise(chem_noise_model)`.
4. **Simulate noisy density matrix**
  - Use `cirq.DensityMatrixSimulator` to compute `rho_noisy`.
  - Compute reference noisy expectation `Tr[H rho_noisy]` via `trace_energy(...)`.
5. **Estimate finite-shot energies**
  - Call `estimate_energy_from_noisy_rho_shots(...)`.
  - Collect:
    - `energy_unmitigated`
    - `energy_rem` (if REM enabled)
6. **Compute noiseless baseline**
  - Simulate noiseless statevector and evaluate exact `⟨psi|H|psi⟩`.

## Noise Model: `LocationAwareDecomposedNoise`

Implemented in `main_cursor_lib.py`.

### Gate-dependent behavior

- **Measurement gates**: passed through unchanged.
- `**CZPowGate` operations**:
  - keep CZ operation,
  - then apply per-qubit channels:
    - `amplitude_damp(...)`
    - `phase_damp(...)`
    - `depolarize(...)`
- **Single-qubit gates**:
  - keep operation,
  - then apply reduced channels (`gamma/10`, `depol/10`).
- **Other operations**: passed through unchanged.

### Location-aware tagging

The decomposed ansatz tags CZs (`cz_normal`, `cz_high`, `cz_onsite_normal`).

For `cz_high`:

- damping/depolarizing strengths are amplified by `high_cz_multiplier`,
- extra depolarizing component `leakage_approx_prob` is added.

This models non-uniform two-qubit error rates by location class.

## OGM Measurement Path

OGM is implemented in `shot_measurement.py` via `estimate_energy_from_noisy_rho_shots(...)`.

### External dependency

OGM requires an external `shadowgrouping` checkout and an OGM settings file:

- `shadowgrouping_root`: external project path (for imports),
- `ogm_file`: precomputed OGM settings text.

If `measurement_scheme="ogm"` and `ogm_file` is missing, a `ValueError` is raised.

### Core steps

1. Convert `PauliSum` terms into integer-encoded observables.
2. Build measurement settings:
  - OGM uses `SettingSampler(..., ogm_file, epsilon=...)`.
3. Sample bitstrings from `rho` in each requested basis.
4. Optionally apply asymmetric readout noise.
5. Aggregate compatible settings per term.
6. Compute term expectations and sum with coefficients (+ offset).

## Readout Error Mitigation (REM)

REM uses per-qubit calibration arrays:

- `p_0_success[j] = P(measure 0 | true 0)` for qubit `j`,
- `p_1_success[j] = P(measure 1 | true 1)` for qubit `j`.

`rem_z_vectors(...)` computes correction vectors from the inverse readout confusion matrix.  
During estimation, each observed bit contributes corrected Z values when `apply_rem=True`.

Practical interpretation:

- `energy_unmitigated`: shot estimate with raw noisy readout.
- `energy_rem`: readout-mitigated estimate (should often be closer to ideal/noisy-trace reference, depending on variance and model mismatch).

## Output Interpretation

For a typical run you will see:

- `Tr[H rho_noisy]`: reference expectation from full noisy density matrix simulation.
- `Shot estimate (unmitigated)`: finite-shot + readout-noisy estimator.
- `Shot estimate (REM)`: readout-mitigated finite-shot estimator.
- `Exact noiseless energy`: no-noise baseline for the same parameter vector.

Expected qualitative relation:

- noiseless energy is usually lowest (most negative),
- noisy-trace energy is shifted upward by noise,
- unmitigated shot estimate can be further biased by readout noise,
- REM often reduces that readout-induced bias.

## Reproducibility and Tuning

- `random_seed`: controls circuit simulation randomness.
- `sampling_seed`: controls shot/basis sampling.
- `num_shots`: higher values reduce estimator variance.

When debugging, hold seeds fixed and vary one knob at a time.

## Quick Validation Checklist

- `Pauli_Ham` contains expected Hamiltonian files for your target bond length.
- `shadowgrouping_root` exists and is importable when using OGM/shadow grouping schemes.
- `ogm_file` path is valid for the same molecule/bond setup.
- `len(vqe_parameters) == ansatz_layers * params_per_layer(num_spatial)`.
- `len(p_0_success) == len(p_1_success) == 2 * num_spatial`.

## Test Coverage Pointers

Recent tests around this pipeline are in `test function/`, including:

- OGM external import/path checks: `test_ogm_imports_external_root.py`
- OGM shot behavior checks: `test_ogm_measurement_shots.py`
- Noise model wiring checks: `test_noise_model_implementation.py`

## Gate Noise Mitigation via ZNE

Zero-Noise Extrapolation (ZNE) is added for gate-noise mitigation in the current flow.

### What is scaled

For each scale factor `lambda` in `zne_scales`:

- `amp_damp_gamma -> lambda * amp_damp_gamma`
- `phase_damp_gamma -> lambda * phase_damp_gamma`
- `depol_prob -> lambda * depol_prob`
- `leakage_approx_prob -> lambda * leakage_approx_prob`

`high_cz_multiplier` is kept fixed, and probability-like terms are clipped to `[0, 1]`.

### Trace-based ZNE (primary path)

Use `run_trace_zne(...)` from `main_cursor_lib.py`:

1. Simulate noisy density matrix at each noise scale.
2. Compute `Tr[H rho_noisy(scale)]`.
3. Fit a polynomial in scale (default linear).
4. Evaluate fit at `scale = 0` to get `energy_zne`.

Default settings in the notebook:

- `zne_scales = [1.0, 2.0, 3.0]`
- `zne_fit_order = 1`

### Shot-based ZNE (optional)

Use `run_shot_zne(...)` from `shot_measurement.py`:

1. Build `rho_noisy(scale)` from the same noise model path.
2. Run `estimate_energy_from_noisy_rho_shots(...)` at each scale.
3. Extrapolate shot energies to `scale = 0`.

This is optional because finite-shot variance can make extrapolation less stable than trace-based ZNE.

### Interpretation and caveats

- `energy_zne` is an extrapolated estimate of the zero-gate-noise energy.
- If scales are too aggressive, channels saturate and polynomial fit quality degrades.
- Prefer modest scales and fixed seeds for reproducible comparisons.
- Keep readout mitigation (`energy_rem`) conceptually separate from gate-noise mitigation (ZNE).

## Mitigation Mode Selector

**Measurement / shot budget:** how many OGM-style batches `run_mitigation` performs per mode (and how `num_shots` multiplies) is documented in [`docs/measurement_cost.md`](docs/measurement_cost.md).

A single `mitigation_mode` knob in `main_cursor.ipynb` (passed to `run_mitigation(...)` in `shot_measurement.py`) selects which mitigation pipeline runs.

| mode    | what runs                                                                 | output keys                                                                 | cost                                              |
|---------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------|
| `none`  | shot-noisy unmit + REM target energies only                                | `unmit_target`, `rem_target`                                                | 1 noisy density-matrix sim                        |
| `zne`   | trace-ZNE + shot-ZNE on the target circuit                                 | `trace_zne`, `shot_zne` (each contains `energy_zne`, fit coeffs, per-scale) | `len(zne_scales)` density-matrix sims             |
| `cdr`   | CDR Models B (unmit -> exact) and C (REM -> exact) at scale 1.0            | `cdr_models`, `cdr_unmit_corrected`, `cdr_rem_corrected`                    | `cdr_num_circuits` shot evaluations               |
| `both`  | sequential CDR-then-ZNE: train CDR at every ZNE noise scale, polyfit to 0  | all CDR + ZNE keys + `per_scale_cdr_unmit/rem`, `zne_of_cdr_unmit/rem_target` | `len(zne_scales) * cdr_num_circuits` shot evals |

When to pick each:

- `none`: establish noisy / REM baselines.
- `zne`: cheap; best when noise is well-behaved with scale.
- `cdr`: when readout/gate noise has structure that linear extrapolation in scale misses.
- `both`: most rigorous; per-scale CDR fits are noise-scale specific (no assumption that coefficients trained at scale 1 transfer to scales 2 or 3), then a polynomial in scale is fit to the *cleaned* points.

`run_mitigation` raises `ValueError` for unknown mode strings.

## Gate Noise Mitigation via CDR / Clifford fitting

The dispatcher supports two **fitting scopes** (via `cdr_cfg["cdr_fit_scope"]`):

- **`per_pauli`** (**default**): Paper-style observable-level CF — one affine map per Hamiltonian Pauli string `⟨P_k⟩_exact ≈ a_k ⟨P_k⟩_noisy + b_k`, then `E = offset + Σ_k w_k ⟨P_k⟩_corrected`. Uses the **same** shot allocation as total-energy estimation (`return_per_term` in `estimate_energy_from_noisy_rho_shots`); **no** extra `num_shots` multiplier vs legacy CDR.
- **`total_energy`** (legacy): Single regression on total ⟨H⟩: `E_exact ≈ a * E_noisy + b` (Models B/C as below).

Two **training-circuit strategies** (`cdr_cfg["cdr_training"]`):

- **`snap_greedy`** (**default**): `t_max`-bounded greedy nearest-Clifford snapping (below).
- **`random_clifford`**: Each training resolver assigns every parameter independently on the Clifford grid (`generate_random_clifford_analogue_param_sets` in `main_cursor_lib.py`), analogous to Guo et al. Clifford-analogue circuits. **`t_max` is ignored** for this strategy.

Outputs include `cdr_training` and `cdr_fit_scope`. Legacy total-energy models expose `coeffs_unmit_to_exact` / `coeffs_rem_to_exact`; per-Pauli models expose `coeffs_unmit_to_exact_per_term` / `coeffs_rem_to_exact_per_term` (lists of `[slope, intercept]` aligned with Hamiltonian Pauli terms).

### Training-circuit strategy: `t_max`-bounded greedy parameter snapping (`cdr_training="snap_greedy"`)

For each training circuit:

1. Start from the target VQE parameter vector.
2. Optionally pre-snap a `cdr_min_snap_fraction` random subset of parameters (off by default).
3. While the resolved circuit has `count_non_clifford_ops(...) > cdr_t_max`, randomly pick an unsnapped symbol and snap it.
4. Resolver is returned.

Snap rules (derived from the decomposition in `main_cursor_lib.py`):

- `theta` (FSim theta-only block) -> nearest multiple of `pi/2`. Two `PhasedXPowGate` ops per theta block become Clifford.
- `phi` (FSim phi-only block) -> nearest multiple of `pi`. Three `ZPowGate` ops per phi block become Clifford.

### Non-Clifford gate counting (the criterion)

A `PhasedXPowGate(exponent=e)` or `ZPowGate(exponent=e)` is **Clifford iff `2 * e` is (approximately) an integer**, i.e., `e in {..., -0.5, 0, 0.5, 1, 1.5, ...}`. `H` and `CZ` are always Clifford.

`count_non_clifford_ops(circuit, resolver)` substitutes symbols, walks every operation, and counts those whose exponent fails this test.

### Two CDR models (legacy `cdr_fit_scope="total_energy"`)

For each training circuit `c_i`:

- `E_exact_i` = noiseless statevector energy
- `E_unmit_i` = shot-noisy unmitigated estimate (from `estimate_energy_from_noisy_rho_shots`)
- `E_rem_i`   = shot-noisy + REM-corrected estimate

Linear fits:

- Model B: `E_exact ~ a_B * E_unmit + b_B` -> `coeffs_unmit_to_exact`
- Model C: `E_exact ~ a_C * E_rem   + b_C` -> `coeffs_rem_to_exact`

Apply on target:

- `E_cdr_unmit_target = a_B * E_target_unmit + b_B`
- `E_cdr_rem_target   = a_C * E_target_rem   + b_C`

### Per-Pauli CF (default `cdr_fit_scope="per_pauli"`)

For each Pauli term `k` and each training resolver, fit the same linear maps on **scalar** ⟨P_k⟩ expectations (exact from the noiseless statevector; noisy from `estimate_energy_from_noisy_rho_shots(..., return_per_term=True)`). Apply each map to the target’s per-term unmitigated / REM expectations, then sum with Hamiltonian weights and `offset`.

### Sequential CDR-then-ZNE (the `both` mode)

For each `s` in `zne_scales`:

1. Build a noise model scaled by `s` via `scale_noise_params_for_zne`.
2. Re-train Models B and C **at scale s** using the same training resolver list (the snap pattern is scale-independent).
3. Apply Models B/C at scale s to get `E_unmit_cdr(s)`, `E_rem_cdr(s)`.

Then fit `numpy.polyfit(deg=zne_fit_order)` across `(zne_scales, [E_*_cdr(s)])` and evaluate at `s = 0`. Outputs: `zne_of_cdr_unmit_target` and `zne_of_cdr_rem_target`.

This is the most rigorous combination because CDR coefficients are not assumed to generalize across noise scales.

### Defaults (notebook)

- `mitigation_mode = "both"`
- `cdr_num_circuits = 24`
- `cdr_t_max = 32` (target H4 8-layer has 192 non-Clifford ops; budget is ~1/6)
- `cdr_min_snap_fraction = 0.0`
- `cdr_seed = 7`

### Diagnostic plots

Run `python docs/scripts/plot_cdr_diagnostic.py` for a single calibration figure, or `python docs/scripts/plot_cdr_quality_panels.py` for the full multi-panel diagnostic figure (saved as `cdr_quality_panels.png`). For a fixed-parameter sweep of `cdr_num_circuits` vs |error| on the 21-param layer-3 baseline (CSV + plot), use `python docs/scripts/sweep_cdr_num_circuits.py` (outputs in `docs/figures/cdr_sweep/`).

The multi-panel figure has:

- **P1/P2** calibration scatter (E_exact vs E_unmit / E_REM), color-coded by `t_remaining`. Tight, monotonic scatter -> CDR will correct well.
- **P3** residuals histogram for both models. Skew or heavy tails -> bad training distribution or non-linear noise.
- **P4** target energy bar comparison across mitigation modes. Headline read for whether `mitigation_mode = "both"` actually beats either pipeline alone.
- **P5** training-size sweep `cdr_num_circuits in [6, 12, 24, 48]`. Tells you whether more circuits help.
- **P6** snap-diversity sweep `cdr_min_snap_fraction in [0.0, 0.3, 0.5, 0.7]`. Tells you whether forced extra snapping helps or hurts.
- **P7** `cdr_t_max in [8, 16, 32, 64, 128]`. The non-Clifford budget vs error curve — the most important panel for assessing whether the design will scale to 100 qubits where small `t_max` is mandatory for classical tractability.

## Layer-3 parameter sweep (full mitigation comparison)

Script: [`docs/scripts/layer3_mitigation_sweep.py`](docs/scripts/layer3_mitigation_sweep.py).

**What it does:** Uses the 3-layer H4 decomposed ansatz (21 parameters per set). Row 0 is a fixed baseline vector; rows 1…`N-1` add Gaussian noise with log-spaced standard deviation in `[0.05, 2.0]` radians. By default `--num-sets 100` (1 baseline + 99 perturbations). For each parameter set it calls `run_mitigation("both", ...)` (which runs trace ZNE, shot ZNE, per-scale CDR, and ZNE-of-CDR) with OGM at `8192` shots, and writes:

- `docs/figures/layer3_sweep/layer3_sweep_results.csv` and `.npz`
- `layer3_ogm_noisy_vs_exact.png` — OGM unmit/REM vs exact, with `y=x` and a line at the Hamiltonian ground-state energy `E_min` (lowest eigenvalue) for reference
- `layer3_mitigation_parity.png` — one parity subplot per mitigation channel (9 methods)
- `layer3_mitigation_error_bars.png` — mean `|E_pred - E_exact|` across all parameter sets per method
- `layer3_mitigation_heatmap.png` — signed error `E_pred - E_exact` (columns = param index, rows = method)
- `layer3_mitigation_summary.txt` — numeric summary and wall time

**CLI (examples):**

```bash
python docs/scripts/layer3_mitigation_sweep.py \
  --out-dir docs/figures/layer3_sweep \
  --shadowgrouping-root /path/to/shadowgrouping

python docs/scripts/layer3_mitigation_sweep.py --quick   # cdr_num_circuits=6 for faster iteration
python docs/scripts/layer3_mitigation_sweep.py --quick --num-sets 10   # shorter sweep (1 + 9 perturbations)

# Notebook VQE baseline (21 floats in .npy) and custom perturbation / noise overrides:
python docs/scripts/layer3_mitigation_sweep.py --quick --num-sets 10 \
  --baseline-npy path/to/layer3_params.npy \
  --sigma-min 0.05 --sigma-max 2.0 \
  --depol-prob 0.01 \
  --out-dir docs/figures/layer3_sweep_custom

# Re-plot from NPZ without new quantum runs (omit --first-n to use every row in the file):
python docs/scripts/plot_layer3_from_npz.py \
  --npz docs/figures/layer3_sweep/layer3_sweep_results.npz \
  --out-dir docs/figures/layer3_sweep

python docs/scripts/plot_layer3_from_npz.py \
  --npz docs/figures/layer3_sweep/layer3_sweep_results.npz \
  --first-n 10 \
  --out-dir docs/figures/layer3_sweep_n10
```

**Runtime:** Roughly on the order of many notebook-length runs × `--num-sets` (`both` mode trains CDR at each ZNE scale per set). Use `--quick` for smoke runs; full `--cdr_num_circuits=24` can take a long time.

**Tests:** Parameter generation only (no OGM) lives in `test function/test_layer3_sweep_smoke.py`.

### Layer-3 noise ablation (which mitigation helps which noise)

Script: [`docs/scripts/layer3_noise_ablation.py`](docs/scripts/layer3_noise_ablation.py).

**What it does:** Runs the same OGM + `run_mitigation("both", …)` stack as the sweep, but cycles **noise scenarios**: turn on one or two channels at reference strength (`DEFAULT_*` in `main_cursor_lib.py`), leave others at zero, optionally toggle shot readout—see the script docstring and [`docs/noise_model.md`](docs/noise_model.md). Writes:

- `noise_ablation_results.csv` — long format with `scenario_id` plus the usual per-row metrics
- `noise_ablation_summary.csv` — mean/std `|E_pred − E_exact|` per scenario × method
- `noise_ablation_results.npz`
- `noise_ablation_heatmap.png` — scenarios × methods (mean absolute error)
- `noise_ablation_bar_<scenario>.png` — bar chart for `--bar-scenario` (default: first scenario in the run)

```bash
python docs/scripts/layer3_noise_ablation.py --quick --num-sets 10 \
  --out-dir docs/figures/layer3_noise_ablation \
  --shadowgrouping-root /path/to/shadowgrouping

# Subset of scenarios (comma-separated):
python docs/scripts/layer3_noise_ablation.py --quick --num-sets 10 \
  --scenarios amp_only,readout_only,full_ref \
  --bar-scenario amp_only \
  --out-dir docs/figures/layer3_noise_ablation_smoke

# Optional: same four parity PNGs per scenario as the sweep (slow):
python docs/scripts/layer3_noise_ablation.py --quick --num-sets 10 \
  --scenarios depol_only \
  --render-layer3-style \
  --out-dir docs/figures/layer3_noise_ablation_one
```

**Tests:** Scenario preset logic only (no OGM) lives in `test function/test_layer3_noise_ablation_scenarios.py`.

### Caveats

- Training distribution must match the target structurally (we keep the same gate layout and noise tags by snapping per-parameter).
- Raising `cdr_t_max` past the target's non-Clifford count makes CDR degenerate to noisy regression on the target itself.
- Lowering `cdr_t_max` too far makes training circuits drift far from the target distribution.
- For `mitigation_mode = "both"`, ensure `zne_fit_order < len(zne_scales)`.