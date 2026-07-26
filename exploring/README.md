# Flexible α–β connectivity exploration

Error-budget proxy (`error_reduction_methods.md` §0):

```
fidelity ≈ (1-5e-4)^n1 · (1-1e-2)^n_onchip · (1-1e-1)^n_cross
```

## Compilation rules (general; enforced for Cl2+)

Defined in `constraints.py`:

1. **All qubits used** — every wire appears in at least one gate.
2. **RZX pairs vertex-disjoint** — each parameterized RZX may join **any α ↔ any β**, but no two RZX may share a qubit.
3. **Within-spin CZ graph only** — nearest-neighbour on each row, plus chords
   `(0,3)` (alpha) and `(half, half+3)` (beta, e.g. `(5,8)` for Cl2).
   **No cross-spin CZ**; α–β coupling is RZX-only.
4. Fan-in uses graph-aware paths (`flexible_compile._fanin_on_graph`); Z-pair
   couriers are re-inserted when a path needs intermediate qubits.

HF’s independent-pair circuit already satisfies these (RZX only, no CZ).

## Baselines

| circuit | 1q | on | cross | fidelity | error |
|---|---|---|---|---|---|
| HF 6q tapered | 26 | 12 | 3 | 0.638 | 0.362 |
| Cl2 10q | 42 | 24 | 7 | 0.368 | 0.632 |

## Winners

### HF 6q — `independent_pairs_freeze`

RZX on `(0,3),(1,4),(2,5)` — fid **0.722** (baseline 0.638).

### Cl2 10q — `disjoint_rzx_all_qubits` (graph-legal)

Selective freeze + three disjoint RZX on the allowed CZ graph:

| double | string | RZX |
|---|---|---|
| t0 | `IYZZXIXZZX` (keep Z2/Z3) | (2, 6) |
| t1 | `YIIZXXIIZX` (keep Z3) | (0, 5) |
| t2 | `IIYZXIIXZX` (keep Z3) | (3, 7) |

CZ edges used are only NN + chords `(0,3),(5,8)`. All qubits 0–9 used.

| | 1q | on | cross | fidelity | error |
|---|---|---|---|---|---|
| baseline | 42 | 24 | 7 | 0.368 | 0.632 |
| **winner** | 32 | 20 | 3 | **0.587** | **0.413** |

## Run

```bash
python exploring/find_cl2_disjoint.py
python exploring/export_winners.py
```
