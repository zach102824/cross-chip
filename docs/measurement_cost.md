# Measurement / shot budget for `run_mitigation`

Each call to `estimate_energy_from_noisy_rho_shots` consumes **`num_shots`** (default 8192) as its global measurement budget for that energy estimate (OGM / shadowgrouping `N_samples`, or **`direct_pauli`** splitting that budget across Pauli settings).

## Per mode

Let `N_C = cdr_num_circuits`, `S = len(zne_scales)`, **`num_shots`** as configured.

| `mitigation_mode` | Approximate count of `estimate_energy_from_noisy_rho_shots` calls | Notes |
|-------------------|---------------------------------------------------------------------|-------|
| `none` | **1** | Baseline target only. |
| `zne` | **`S`** (when `S ≥ 2`) | Shot-ZNE per scale; trace-ZNE uses density matrices only (no shot loop). |
| `cdr` | **`1 + N_C`** | One baseline target estimate + **one estimate per training resolver** (training uses the same `num_shots` each time). |
| `both` | **`S` + S × (1 + N_C)** | Shot-ZNE on target (`S` estimates if `S ≥ 2`) plus, for **each** noise scale, one target baseline + `N_C` training estimates for CDR/ZNE-of-CDR. |

## Per-Pauli vs total-energy CDR

With **`cdr_fit_scope="per_pauli"`** (default), each estimate still uses **one** `num_shots` batch; per-term ⟨P_k⟩ values are extracted from the **same** samples (`return_per_term=True`). There is **no** extra factor of `L` (number of Pauli strings) in the shot budget versus **`cdr_fit_scope="total_energy"`**.

## Related config keys

- `cdr_training`: `snap_greedy` | `random_clifford` — changes how training **resolvers** are generated, not `num_shots` per estimate.
- `cdr_fit_scope`: `per_pauli` | `total_energy` — changes regression structure, not the shot budget per call.
