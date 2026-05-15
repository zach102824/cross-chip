# REM + OGM in `main_cursor.ipynb` (step by step)

This note explains exactly how readout error mitigation (REM) is applied when `measurement_scheme="ogm"` in `main_cursor.ipynb`.

The implementation path is:

1. `main_cursor.ipynb` calls `estimate_energy_from_noisy_rho_shots(...)`.
2. `estimate_energy_from_noisy_rho_shots` is in `shot_measurement.py`.
3. OGM settings are sampled by shadowgrouping (`SettingSampler.find_setting`).
4. Per-term unmitigated and REM-corrected expectations are computed from the same bitstrings.

Below uses the same 8-qubit H4 setup as the notebook cell:

1. `num_shots = 8192`
2. `measurement_scheme = "ogm"`
3. `apply_readout_noise = True`
4. `apply_rem = True`
5. `p_0_success = [0.93, 0.87, 0.87, 0.88, 0.90, 0.89, 0.87, 0.91]`
6. `p_1_success = [0.95, 0.91, 0.95, 0.96, 0.94, 0.94, 0.92, 0.95]`

## Full flow (numbered)

1. Convert Hamiltonian `H` (a `cirq.PauliSum`) into:
   1. integer Pauli rows (`I/X/Y/Z -> 0/1/2/3`),
   2. term coefficients `c_k`,
   3. scalar identity offset.
2. Ask OGM for a shot-by-shot measurement schedule with 8192 rows:
   1. each row is an 8-qubit basis setting like `ZXZXYZYZ`,
   2. duplicate rows mean multiple shots in the same basis.
3. Group duplicated OGM rows and count them (basis -> shot count).
4. For each unique basis:
   1. rotate `rho_noisy` to that basis,
   2. sample the assigned number of ideal bitstrings,
   3. apply asymmetric readout noise to each shot bit using `p_0_success`, `p_1_success`.
5. Precompute REM vectors per qubit with `rem_z_vectors(...)`:
   1. this builds the inverse single-qubit readout map,
   2. it returns two numbers per qubit: value to use when measured bit is `0` and when it is `1`.
6. For each Pauli term `P_k` in `H`:
   1. collect all sampled bases that are compatible with `P_k` (same non-identity axes),
   2. from all those shots, compute unmitigated and REM-corrected single-shot term values,
   3. average to get `⟨P_k⟩_unmit` and `⟨P_k⟩_rem`.
7. Form the two energies:
   1. `E_unmit = offset + Σ_k c_k * ⟨P_k⟩_unmit`,
   2. `E_rem   = offset + Σ_k c_k * ⟨P_k⟩_rem`.
8. Return dictionary fields:
   1. `energy_unmitigated`,
   2. `energy_rem`,
   3. optional per-term arrays if requested.

## Concrete per-term example with one shot `10011100`

Assume one Pauli term is:

1. `P_k = Z0 I1 X2 Z3 I4 I5 Y6 Z7`
2. active qubits are `[0, 2, 3, 6, 7]` (non-identity positions).

Assume one noisy sampled shot bitstring is:

1. `b = 10011100`
2. so active bits are `[b0,b2,b3,b6,b7] = [1,0,1,0,0]`.

### A) Unmitigated value from this one shot

1. Map each active bit with `z = 1 - 2*b`:
   1. `b0=1 -> -1`
   2. `b2=0 -> +1`
   3. `b3=1 -> -1`
   4. `b6=0 -> +1`
   5. `b7=0 -> +1`
2. Multiply active values:
   1. `(-1) * (+1) * (-1) * (+1) * (+1) = +1`
3. This shot contributes `+1` to the unmitigated estimator of this term.

### B) REM-corrected value from this one shot

Using notebook readout calibration arrays, `rem_z_vectors` gives per-qubit values (rounded):

1. qubit 0: bit `0 -> +1.1591`, bit `1 -> -1.1136`
2. qubit 2: bit `0 -> +1.3171`, bit `1 -> -1.1220`
3. qubit 3: bit `0 -> +1.2857`, bit `1 -> -1.0952`
4. qubit 6: bit `0 -> +1.3291`, bit `1 -> -1.2025`
5. qubit 7: bit `0 -> +1.2093`, bit `1 -> -1.1163`

For `b = 10011100` (active bits `[1,0,1,0,0]`):

1. pick values:
   1. q0 uses `-1.1136`
   2. q2 uses `+1.3171`
   3. q3 uses `-1.0952`
   4. q6 uses `+1.3291`
   5. q7 uses `+1.2093`
2. multiply them:
   1. `(-1.1136)*(+1.3171)*(-1.0952)*(+1.3291)*(+1.2093) ≈ +2.58`
3. This shot contributes about `+2.58` to the REM estimator for this term.

Notes:

1. A single REM-corrected shot can be outside `[-1, +1]` because it uses inverse calibration weights.
2. The final term value is the average across many shots and usually comes back to a physical range statistically.

## How this single shot enters total energy

For term coefficient `c_k`:

1. collect all compatible shots for `P_k` from OGM-selected bases,
2. average their unmitigated per-shot values -> `⟨P_k⟩_unmit`,
3. average their REM per-shot values -> `⟨P_k⟩_rem`,
4. add to totals:
   1. unmitigated contribution: `c_k * ⟨P_k⟩_unmit`,
   2. REM contribution: `c_k * ⟨P_k⟩_rem`.

Repeat for every Pauli term, then add Hamiltonian offset.

## Function map (where each step happens)

1. `pauli_sum_to_int_observables`: Hamiltonian term encoding.
2. `_load_shadowgrouping_scheme` + `SettingSampler.find_setting`: OGM shot settings.
3. `_unique_settings_with_counts`: collapse duplicate settings.
4. `sample_measurement_basis_from_rho`: basis-rotated shot sampling.
5. `apply_asymmetric_readout_noise`: inject readout bit flips.
6. `rem_z_vectors`: inverse readout correction vectors.
7. `_term_expectation_from_bitstrings`: per-term unmitigated or REM expectation.
8. `estimate_energy_from_noisy_rho_shots`: final energy assembly.
