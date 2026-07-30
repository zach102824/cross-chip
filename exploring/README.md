# Flexible α–β connectivity — HF & Cl2 winners

Error-budget proxy (`error_reduction_methods.md` §0):

```
fidelity ≈ (1-5e-4)^n1 · (1-1e-2)^n_onchip · (1-1e-1)^n_cross
```

## Naming

```
{Molecule}_{nq}q[_2doubles]_disjoint_rzx.{json,_circuit.png}          # winners
HF_6q_freeze_fixed_rzx.{json,_circuit.png}                            # fixed RZX + skip CZ
HF_6q_freeze_fixed_rzx_nn_only.{json,_circuit.png}                    # fixed RZX, NN CZ only
Cl2_10q_2doubles_disjoint_rzx_nn_only.{json,_circuit.png}             # disjoint, NN CZ only
```

| molecule | stem | RZX | on-chip 2q | depth |
|---|---|---|---|---|
| HF 6q winner | `HF_6q_disjoint_rzx` | `(0,3),(1,4),(2,5)` | 0 | 5 |
| HF fixed + skip | `HF_6q_freeze_fixed_rzx` | `(2,5)×3` | 8 (incl. `(0,2)/(3,5)`) | 16 |
| HF fixed NN-only | `HF_6q_freeze_fixed_rzx_nn_only` | `(2,5)×3` | 12 (NN only) | 20 |
| Cl2 winner | `Cl2_10q_2doubles_disjoint_rzx` | `(2,7),(0,5)` | 16 | 16 |
| Cl2 NN-only | `Cl2_10q_2doubles_disjoint_rzx_nn_only` | `(2,6),(1,5)` | 20 | 25 |

## How much do skip / chord edges save?

From `chord_savings.json`:

| case | with long-range CZ | NN-only | Δ error | on-chip CZ saved | depth |
|---|---|---|---|---|---|
| **HF** fixed RZX `(2,5)` | skip `(0,2)/(3,5)`: err 0.334, 8 CZ | err 0.362, 12 CZ | **0.028** | **4** | 16 → 20 |
| **Cl2** disjoint RZX | chords `(0,3)/(5,8)`: err 0.320, 16 CZ | err 0.349, 20 CZ | **0.029** | **4** | 16 → 25 |

HF NN-only keeps courier pairs `[{1,2},{2},∅]` so fan-in can walk `0–1–2` / `3–4–5`
instead of jumping `0–2` / `3–5`. Same freeze+fixed-RZX strategy as
`state_transfer/.../HF_tapered_6q_3doubles_freeze_fixed_rzx`.

Cl2 NN-only re-searches freeze masks under NN CZ; needs more couriers and longer ladders.

## Compilation rules

Defined in `constraints.py` (`allowed_cz_edges(..., chords=True|False)`):

1. **All qubits used** (disjoint winners)
2. **RZX pairs vertex-disjoint** (disjoint winners; fixed-RZX baselines intentionally share `(2,5)`)
3. **Within-spin CZ** — NN + optional long-range; no cross-spin CZ
4. Graph-aware fan-in (`flexible_compile._fanin_on_graph`)
5. **Cl2 extra:** q3 not an RZX hub; α ∈ `{0,1,2}`, β ∈ `{5,6,7}`

## Regenerate

```bash
.venv_py311/bin/python exploring/export_winners.py
.venv_py311/bin/python exploring/export_nn_baselines.py
```

Library: `constraints.py`, `error_budget.py`, `flexible_compile.py`, `freeze.py`, `methods.py`.
