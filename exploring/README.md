# Flexible α–β connectivity — HF & Cl2 winners

Error-budget proxy (`error_reduction_methods.md` §0):

```
fidelity ≈ (1-5e-4)^n1 · (1-1e-2)^n_onchip · (1-1e-1)^n_cross
```

## Naming

```
{Molecule}_{nq}q_disjoint_rzx.{json,_circuit.png}
```

| molecule | stem | RZX pairs | 2q gates | depth |
|---|---|---|---|---|
| HF 6q | `HF_6q_disjoint_rzx` | `(0,3),(1,4),(2,5)` | 3 | 5 |
| Cl2 10q | `Cl2_10q_disjoint_rzx` | `(0,5),(2,7),(3,8)` | 23 (20 CZ + 3 RZX) | 21 |

Both are exhaustive optima over keep-masks × vertex-disjoint bridge schedules,
ranked by (total 2q, depth, total 1q).

## Compilation rules

Defined in `constraints.py`:

1. **All qubits used**
2. **RZX pairs vertex-disjoint** (any α ↔ any β)
3. **Within-spin CZ graph only** — NN + chords `(0,3)` / `(half,half+3)`; no cross-spin CZ
4. Graph-aware fan-in (`flexible_compile._fanin_on_graph`)

## Regenerate

```bash
python exploring/export_winners.py
```

Writes the two stems above plus `winners.json`.

Library: `constraints.py`, `error_budget.py`, `flexible_compile.py`, `freeze.py`, `methods.py`.
