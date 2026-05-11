# LiH CDR Shot Cost (per-Pauli, detailed)

This note answers your exact question:

- **With `per_pauli` CDR and `num_circuits=3`, for each Pauli term, how many noisy quantum shots are used?**
- And then: **what is the full total to get one final CDR-corrected energy?**

---

## 1) Configuration in this LiH run

- Workflow: `test_LiH_case/lih_fig13_compiled_ansatz.ipynb` (bottom CDR cell)
- Mode: `run_mitigation("cdr")`
- CDR scope: `cdr_fit_scope="per_pauli"`
- `num_shots = 8192`
- `num_circuits = 3`
- LiH non-identity Pauli terms in fit: `61`
- Measurement scheme: `ogm`

---

## 2) Most important point: shots are **not** split as `8192 / 61`

In OGM, one sampled measurement basis can be compatible with many Pauli terms.
So each Pauli term reuses overlapping shot pools.

That means:

- You do **not** allocate disjoint per-term shot blocks.
- A term's "effective shots" = number of sampled settings compatible with that term.
- Sum of all terms' effective shots can be much larger than 8192 (because of reuse).

---

## 3) Per Pauli term, per one noisy-estimation call

For one call to `estimate_energy_from_noisy_rho_shots(..., num_shots=8192)`:

- Requested sampled settings: `8192`
- For a given Pauli term `k`, define:
  - `S_k` = number of sampled settings compatible with term `k`

Then per-term noisy estimate uses:

- **effective shots for term `k` = `S_k`**, if `S_k > 0`
- fallback to direct 1-shot if `S_k = 0`

So mathematically:

- `N_eff(k) = max(S_k, 1)`

For this LiH + OGM file combination:

- There are **16 terms** with `S_k = 0` in each call, so they get fallback `N_eff=1`.
- The other terms have `N_eff > 1` (varies by term).

---

## 4) Per Pauli term across all CDR fitting work (3 training circuits)

Your per-Pauli CDR model is trained on 3 near-Clifford circuits.

For each Pauli term `k`:

- training x/y pairs are built from 3 noisy-estimation calls
- so that term gets:
  - `N_eff_train(k,1)` shots from circuit 1 call
  - `N_eff_train(k,2)` shots from circuit 2 call
  - `N_eff_train(k,3)` shots from circuit 3 call

Total for that term in CDR training:

- `N_eff_train_total(k) = N_eff_train(k,1) + N_eff_train(k,2) + N_eff_train(k,3)`

If a term is in the always-zero-compatible set (16 terms), this becomes:

- `N_eff_train_total(k) = 1 + 1 + 1 = 3`

---

## 5) How many noisy-estimation calls happen for one final CDR energy

With `mode="cdr"` and `per_pauli`:

1. Baseline target noisy energy call (`return_per_term=False`)
2. 3 training-circuit noisy calls (for per-term fit data)
3. Target per-term noisy call (`return_per_term=True`) for applying fitted models

Total calls:

- `N_calls = num_circuits + 2 = 3 + 2 = 5`

---

## 6) Full shot total for one final CDR-corrected energy

### Nominal sampled OGM shots

- `N_nominal = N_calls * num_shots = 5 * 8192 = 40960`

### Extra fallback direct measurements

Each call has 16 zero-compatible terms, each adds 1 fallback shot:

- `N_fallback_per_call = 16`
- `N_fallback_total = 5 * 16 = 80`

### Grand total

- `N_total = N_nominal + N_fallback_total`
- `N_total = 40960 + 80 = 41040`

---

## 7) Direct answer to your question

For `per_pauli` with 3 training circuits:

- **Per term, per one noisy-estimation call:** `N_eff(k) = max(S_k,1)` (varies by term)
- **Per term across 3 training circuits only:** `sum_{i=1..3} N_eff_train(k,i)`
- For 16 unsupported terms in this LiH run: exactly **3 shots total** in training (1 per circuit fallback)

And for one full final CDR energy result (fit + apply):

- **41040 total measurement samples**

---

## 8) Real-machine variant (skip baseline target call)

If you run on real hardware and do **not** need the initial baseline target noisy-energy call
(`_baseline_target_energies(..., return_per_term=False)`), then:

- `N_calls_real = (num_circuits + 2) - 1 = num_circuits + 1`
- with `num_circuits=3`: `N_calls_real = 4`

Shot total with the same `num_shots=8192`:

- `N_nominal_real = 4 * 8192 = 32768`

Using the same fallback pattern (16 zero-compatible terms per call):

- `N_fallback_real = 4 * 16 = 64`

So the real-machine total becomes:

- `N_total_real = 32768 + 64 = 32832`
