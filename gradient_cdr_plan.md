# Gradient-First Error Mitigation for VQE

**Scale Stripping, Slope CDR, and Final Energy Evaluation**

Notes from discussion (OTOC → VQE transfer)

---

## Abstract

Standard Clifford Data Regression (CDR) learns an affine map

$$\langle O\rangle_{\text{ideal}} \approx a\,\langle O\rangle_{\text{noisy}} + b$$

for each observable and is then used both to guide a VQE optimizer and to report a final energy. With parameter-shift VQE the per-iteration measurement count is $2n_{\text{param}}+1$: two shifted evaluations per parameter for gradients, plus one cheap cost-function evaluation. This note keeps that $+1$ cost term, but treats **gradient accuracy** as the main target during search. Multiplicative noise is first removed by a shared division (OTOC-style scale stripping); additive bias is suppressed by left/right differences; a small residual CDR acts on slopes. Absolute-energy CDR is still used for a careful final $E(\theta^\star)$.

---

## 1. Motivation

In parameter-shift VQE, each iteration measures roughly

$$\text{#evaluations} = 2\,n_{\text{param}} + 1,$$

i.e. two shifted circuits per parameter for the gradient, plus one evaluation of the cost at the current $\theta$.

The $+1$ cost evaluation is cheap relative to the $2n_{\text{param}}$ gradient budget, so we **keep it** for monitoring / logging / optional line-search checks. The accuracy bottleneck is the gradient block: that is what decides where the optimizer goes.

Noise on an expectation value is often roughly

$$E^{\text{n}} \approx N\, E^{\text{id}} + \beta,$$

where $N<1$ is a shared multiplicative attenuation and $\beta$ is an additive offset (SPAM, leakage of global bias, etc.). Absolute CDR tries to undo both with a per-observable fit $(a,b)$. When those fits are imperfect or inconsistent across Paulis, reconstructed gradients can be **worse** than raw differences.

The main bet of this note is therefore not “drop the cost function”, but:

> **Can an OTOC-style division improve the accuracy of the $2n_{\text{param}}$ gradient measurements (and, for free, the $+1$ cost)?**

The Mi et al. OTOC experiments suggest a useful split of labor:

- **Division / reference ratio** removes a shared scale $N$ (no-butterfly or reference-Clifford normalization).
- **Differences** cancel a common additive offset.
- **Regression** is reserved for leftover, circuit-dependent error.

---

## 2. Relation to standard CDR

This plan is still in the CDR family: it learns a linear map from noisy training data with classically known ideals. The key difference is the **regression target**.

| | Standard CDR | This plan (search phase) |
|---|---|---|
| Fit object | each observable / energy | parameter-shift gradient |
| Typical map | $y \approx a x + b$ | $g \approx a\, g^{\#}$ (often no $b$) |
| Role of division | folded into $a$ | explicit shared $\hat{N}$ first |
| Used for | absolute $E$ every step | walking to $\theta^\star$ |

If per-observable CDR were perfect, gradients built from mitigated observables would match this plan. In the realistic few-shot regime, fitting the slope directly is more stable for optimization, because two inconsistent absolute fits at $\theta_j\pm\pi/2$ can manufacture a fake force.

---

## 3. Full workflow

### Phase I — Search (gradients decide; keep the cheap $+1$ cost)

Per iteration keep the full $2n_{\text{param}}+1$ pattern:

- $2n_{\text{param}}$: parameter-shift evaluations → gradients (primary decision signal; accuracy target of this plan)
- $+1$: cost $E(\theta)$ at the current point (cheap; keep it for curves, early stopping monitors, or optional accept/reject)

We do **not** drop the cost evaluation. We also do not treat improving that single number as the main win condition during search: the win is better gradient accuracy from division (+ residual slope CDR).

#### Step 1 — Parameter updates from gradients; keep $E(\theta)$

Compute parameter-shift slopes and take the optimizer step from those. Also measure the current cost (the $+1$). Use $E(\theta)$ for logging and diagnostics; optionally as a weak secondary check, but the step direction comes from the mitigated gradients.

#### Step 2 — Divide out multiplicative error (accuracy lever)

From the usual near-Clifford CDR training set $\{U_k\}$ (no extra Clifford budget required), estimate a shared scale

$$\hat{N} = \operatorname{median}_{k,P} \frac{\langle P\rangle^{\text{n}}_{k}}{\langle P\rangle^{\text{id}}_{k}},$$

restricting to Paulis with $|\langle P\rangle^{\text{id}}|$ above a cutoff (e.g. $0.2$) to avoid unstable ratios. Optionally use one $\hat{N}_w$ per Pauli weight $w$.

An OTOC-like alternative is a single echo / reverse-ansatz companion whose measured fidelity proxy is used as $\hat{N}$.

Apply the **same** division to every evaluation in the $2n_{\text{param}}+1$ bundle (both shifted energies and the current cost):

$$\langle P\rangle^{\#} = \frac{\langle P\rangle^{\text{n}}}{\hat{N}}, \qquad E^{\#} = \sum_i c_i \langle P_i\rangle^{\#}.$$

This is the central accuracy hypothesis: if noise is largely a shared shrink $N$, dividing by $\hat{N}$ restores gradient magnitudes (and makes the monitored $E^{\#}(\theta)$ less attenuated) at almost no extra quantum cost.

#### Step 3 — Residual CDR on slopes, then step

Form parameter-shift gradients from cleaned energies:

$$g_j^{\#} = \frac{1}{2}\Bigl( E^{\#}(\theta_j+\pi/2) - E^{\#}(\theta_j-\pi/2) \Bigr).$$

On near-Clifford training circuits, compute the analogous ideal gradients $g_j^{\text{id}}$ classically. Fit a homogeneous residual model

$$g_j^{\text{id}} \approx a_j\, g_j^{\#}$$

(or a shared $a$ across parameters). Prefer no intercept: after differencing, a fit $b$ often invents fake force.

At the real ansatz, apply $g_j^{\text{mit}} = a_j g_j^{\#}$ and take the optimizer step (SGD/Adam/l-BFGS on $g^{\text{mit}}$). The cleaned cost $E^{\#}(\theta)$ can be logged in parallel; it is not the object of the residual slope fit.

### Phase II — Final evaluation (care about absolute energy)

After reaching $\theta^\star$:

1. Remeasure the Pauli set at $\theta^\star$ (optionally with more shots).
2. Apply ordinary absolute CDR, $\langle P\rangle_{\text{ideal}} \approx a\langle P\rangle_{\text{noisy}}+b$, or a single fit on total energy, using the same training set.
3. Report that mitigated $E(\theta^\star)$ as the result.

Optional during search: mitigate energy once at an anchor $\theta_0$ and accumulate mitigated differences for plotting only; still re-evaluate absolutely at $\theta^\star$.

---

## 4. Three-parameter toy model

Consider three variational angles $(a,b,c)$ with true energy

$$E_{\text{true}}(a,b,c)=(a-1)^2+(b-2)^2+(c-0)^2,$$

so the optimum is $(1,2,0)$.

Suppose the device returns

$$E_{\text{noisy}}=0.6\, E_{\text{true}}+3.0.$$

Then:

- **Division** by $0.6$ removes the shrink but leaves a shifted absolute energy: cleaned $=$ true $+5$.
- **Left/right differences** for a parameter-shift cancel the shared $+5$, recovering the correct slope direction and (after division) the correct slope size in this global-noise toy.
- **Residual slope CDR** (multiply by a learned factor near $1$) absorbs leftover warping when real noise is not perfectly global.

Absolute energy remains biased during the search; the walk still heads toward $(1,2,0)$. A final absolute CDR at $\theta^\star$ is what produces the reported energy number.

---

## 5. Why division helps parameter-shift

With $E^{\text{n}} = N E^{\text{id}} + \beta$,

$$E^{\text{n}}_R - E^{\text{n}}_L = N\bigl(E^{\text{id}}_R - E^{\text{id}}_L\bigr).$$

The offset $\beta$ cancels in the difference, but the slope is still scaled by $N$. Dividing measurements by $\hat{N}\approx N$ restores slope magnitude before (or as part of) the residual CDR fit.

Without division, even a perfect left/right cancel of $\beta$ still leaves every gradient too small by $N$, which slows or distorts the walk (learning-rate mismatch, wrong relative step sizes if $N$ is Pauli-/depth-dependent).

The same $\hat{N}$ can be applied to the cheap $+1$ cost evaluation; that may improve the monitored energy curve, but the primary accuracy claim is about the $2n_{\text{param}}$ gradient channels.

This is the same spirit as OTOC no-butterfly / reference-Clifford normalization: estimate a shared attenuation from circuits whose ideal scale is known, then divide.

---

## 6. Recommended defaults

- Per-iteration layout: keep $2n_{\text{param}}+1$ (gradients + cost).
- Training budget $M$: same near-Clifford set as vanilla CDR (no extra Clifford circuits required for the search-phase method).
- $\hat{N}$: median ratio over Paulis with $|\langle P\rangle^{\text{id}}|>0.2$; apply to all $2n_{\text{param}}+1$ evaluations.
- Slope CDR: one shared $a$, ridge regression through the origin; upgrade to per-parameter $a_j$ only if data allow.
- Refit $(\hat{N},a)$ every $K$ iterations (e.g. 5–10) from cached training measurements; remeasure training set only under device drift.
- Success metrics:
  1. gradient fidelity / angle to true $\nabla E$ during search;
  2. true or high-effort final $E(\theta^\star)$.

  The running $+1$ cost is a diagnostic, not the sole score.

---

## 7. Minimal ablation

Hold fixed the near-Clifford count $M$, shot budget, and the $2n_{\text{param}}+1$ evaluation pattern, then compare:

1. Vanilla absolute CDR on each evaluation / energy, then form gradients.
2. Scale-strip only ($\hat{N}$) on all $2n_{\text{param}}+1$ values; raw parameter-shift gradients (no residual slope CDR).
3. Full plan: $\hat{N}$ + residual slope CDR on gradients; keep cleaned $+1$ cost for monitoring; absolute CDR at $\theta^\star$.

Declare a win if division (2 or 3) improves gradient accuracy and/or yields a better true $E(\theta^\star)$ than method 1 at the same quantum cost. Method 2 isolates whether division alone helps; method 3 asks whether residual slope CDR adds more on top.

---

## 8. Pitfalls

- Division by small ideal Pauli values is unstable; use a cutoff or weight-dependent $\hat{N}_w$.
- Independent absolute CDR at $\theta_j\pm\pi/2$ can create fake gradients; mitigate the difference/gradient as one object.
- Differences amplify shot noise; prefer a shared slope factor $a$ and allocate shots to the shift pairs that drive the step.
- Final reported energy still needs Phase II absolute mitigation; gradient-only mitigation is not a substitute for that number.

---

## 9. One-line summary

> Keep $2n_{\text{param}}+1$ evaluations; let gradients drive the walk and keep the cheap cost monitor. Divide all of them by a shared $\hat{N}$ to undo multiplicative shrink; difference out shared offset in the shifts; CDR-correct leftover slope error; absolute-mitigate energy carefully once at $\theta^\star$.
