# LiH CDR+REM VQE Measurement Accounting (Full Walkthrough)

This document is a plain-language walkthrough of measurement cost in the LiH VQE notebook:

- `test_LiH_case/lih_fig13_compiled_ansatz.ipynb`
- VQE section with Adam + parameter-shift + CDR objective (`cdr_rem_corrected`)

The intent is to answer one specific confusion:

> Why do I see numbers like `cdr_calls_cum=29` and `energy_evals_cum≈203`, and what exactly is each “energy eval” doing?

We will explain this from top to bottom, including a concrete indexed timeline.

---

## 0) Quick symbol dictionary (no shorthand assumptions)

These symbols appear in formulas and logs:

- `P`: number of trainable parameters in the ansatz  
  In this notebook, `P=3` (`theta1`, `theta2`, `theta3`).

- `T`: number of **main** VQE iterations  
  In this notebook, `T=15`.

- `W`: number of learning-rate candidates in warmup  
  `VQE_LR_GRID=(0.01, 0.03, 0.02)`, so `W=3`.

- `K`: warmup mini-iterations per LR candidate  
  `VQE_WARMUP_ITERS=1`, so `K=1`.

- `C`: number of CDR training circuits (`num_circuits`)  
  In this notebook, `C=5`.

- `S`: shots per noisy energy-estimation call (`num_shots`)  
  In this notebook, `S=8192`.

Counters printed in notebook:

- `cdr_calls_iter`: outer CDR calls made in one main iteration.
- `cdr_calls_cum`: cumulative outer CDR calls so far.
- `energy_evals_cum`: cumulative inner noisy energy-estimation calls (modeled as `cdr_calls_cum * (C+2)`).
- `shots_cum`: cumulative shots (modeled as `energy_evals_cum * S`).

---

## 1) The two-layer counting model

This is the most important idea.

### Layer A: outer calls (`cdr_calls_*`)

An outer call means one invocation of `run_mitigation("cdr", ...)`.  
In your VQE code, this happens in two pathways:

1. `energy_from_params(..., energy_mode="cdr_rem_corrected")` (used in gradient shift evaluations)
2. `_vqe_run_mitigation_triple(...)` (used for post-update logging of raw/REM/REM+CF)

Each of those increments `_n_energy_total` by 1.

### Layer B: inner noisy energy evaluations (`energy_evals_*`)

Inside one `run_mitigation("cdr")`, your notebook uses this cost model:

`energy_evals_per_cdr_call = num_circuits + 2 = C + 2`

With `C=5`, one outer CDR call corresponds to:

- `7` inner noisy energy-evaluation calls.

Therefore:

- `shots_per_cdr_call = 7 * 8192 = 57344`.

So when you see `cdr_calls_cum=29`, that means approximately:

- `29 * 7 = 203` inner noisy energy evaluations,
- `203 * 8192 = 1,662,976` shots.

That is exactly the line you observed.

---

## 2) What one outer CDR call does internally (the 7 inner evals)

For one outer CDR call `k`, we can index inner evaluations `j=1..7`.

Interpretation used by this notebook:

1. Target baseline noisy estimate (`return_per_term=False`)
2. Training circuit 1 noisy estimate
3. Training circuit 2 noisy estimate
4. Training circuit 3 noisy estimate
5. Training circuit 4 noisy estimate
6. Training circuit 5 noisy estimate
7. Target per-term noisy estimate (`return_per_term=True`) for applying fitted per-Pauli model

So one outer call always expands to these seven inner evaluations (in the present code path/config).

If you want a global inner-eval index, you can define:

`eval_id = 7*(k-1) + j`

where `k` is outer call index and `j in {1..7}`.

---

## 3) What happens in one main VQE iteration

Main iteration with `P=3` performs:

1. Gradient call for `theta1 + pi/2`
2. Gradient call for `theta1 - pi/2`
3. Gradient call for `theta2 + pi/2`
4. Gradient call for `theta2 - pi/2`
5. Gradient call for `theta3 + pi/2`
6. Gradient call for `theta3 - pi/2`
7. Post-update logging call (returns raw, REM, REM+CF energy)

So each main iteration always has:

- `cdr_calls_iter = 7`.

And inner-equivalent cost per main iteration:

- `7 * 7 = 49` inner noisy energy evaluations,
- `49 * 8192 = 401,408` shots.

---

## 4) Why iter 1 cumulative is 29 (not 7 or 14)

Because cumulative count includes setup work before main iteration 1.

### Pre-main work:

1. Initial baseline outer call: `+1`
2. LR warmup outer calls:
   - For each LR candidate, warmup mini-loop does `(2P+1)=7` calls
   - With `W=3`, `K=1`: warmup calls = `3*1*7 = 21`

Total before main iteration 1:

- `1 + 21 = 22` outer calls.

Then iteration 1 contributes its own 7:

- `cdr_calls_cum = 22 + 7 = 29`.

Inner-equivalent:

- `29 * 7 = 203` energy evals,
- `203 * 8192 = 1,662,976` shots.

Exactly what your print line shows.

---

## 5) Explicit indexed timeline (outer calls and inner ranges)

Use 1-based outer call index `k`.

### Pre-main phase

- `k=1`: initial baseline call  
  inner eval IDs: `1..7`

- `k=2..8`: warmup for LR candidate #1  
  inner eval IDs: `8..56`

- `k=9..15`: warmup for LR candidate #2  
  inner eval IDs: `57..105`

- `k=16..22`: warmup for LR candidate #3  
  inner eval IDs: `106..154`

### Main iteration 1

- `k=23`: grad theta1(+)
- `k=24`: grad theta1(-)
- `k=25`: grad theta2(+)
- `k=26`: grad theta2(-)
- `k=27`: grad theta3(+)
- `k=28`: grad theta3(-)
- `k=29`: post-update logging

inner eval IDs: `155..203`

### Main iteration t (general)

Outer call range for iteration `t` (1-based):

- `k_start(t) = 22 + 7*(t-1) + 1`
- `k_end(t) = 22 + 7*t`

Inner eval range:

- `e_start(t) = 7*(k_start(t)-1) + 1`
- `e_end(t) = 7*k_end(t)`

For `t=15`:

- `k_end(15) = 127`
- `e_end(15) = 7*127 = 889`

So at the end of iteration 15 (before final recheck), the notebook correctly prints:

- `cdr_calls_cum=127`
- `energy_evals_cum≈889`

### After the loop

Notebook does one extra final recheck call:

- `k=128`
- inner eval IDs `890..896`

That is why full-run inner total is 896.

---

## 6) Total formulas and concrete totals for current settings

Current settings:

- `P=3`, `T=15`, `W=3`, `K=1`, `C=5`, `S=8192`

Total outer calls:

`N_cdr_calls = 1 + W*K*(2P+1) + T*(2P+1) + 1`

Plug in numbers:

`N_cdr_calls = 1 + 3*1*7 + 15*7 + 1 = 128`

Total inner noisy energy evals:

`N_energy_evals = N_cdr_calls*(C+2) = 128*7 = 896`

Total shots:

`N_shots = N_energy_evals*S = 896*8192 = 7,340,032`

These are the totals implied by your current notebook print model.

---

## 7) What the notebook print fields mean in one line

When you see:

`iter=01 ... cdr_calls_iter=7 cdr_calls_cum=29 energy_evals_cum≈203 shots_cum≈1662976`

interpretation:

- `cdr_calls_iter=7`: this iteration did exactly 6 gradient-shift calls + 1 post-update call.
- `cdr_calls_cum=29`: includes pre-main baseline + warmup + current iteration.
- `energy_evals_cum≈203`: multiply cumulative outer calls by 7 internal evals per call.
- `shots_cum≈1662976`: multiply by 8192 shots per internal eval.

---

## 8) Caveats

This document follows the current notebook accounting assumptions exactly:

- inner cost per outer call is modeled as `num_circuits + 2`.

Real hardware totals can change if any of these change:

- `num_circuits`
- `num_shots`
- warmup grid/iters (`W`, `K`)
- number of parameters `P`
- number of VQE iterations `T`
- mitigation scope/implementation path (`per_pauli` vs others)

So always recompute totals from current runtime config.
