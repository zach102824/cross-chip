# H4 fixed-ansatz circuits (from scratch)

Target unitary on the tapered 6-qubit register:

```
U = ∏_k exp(-i t_k / 2 · P_k)
```

with the 8 fixed doubles from `UCCSD_Mole/H4.ipynb` at d = 1.00 Å
(param ids `[12, 5, 9, 14, 7, 4, 10, 13]`):

| pid | tapered Pauli |
|-----|---------------|
| 12  | `IYXIXX` |
| 5   | `YZXXZX` |
| 9   | `YZXIXZ` |
| 14  | `IYZIXZ` |
| 7   | `YZZXZZ` |
| 10  | `YZZIXX` |
| 4   | `IIIYXX` |
| 13  | `IYXIXZ` |

## Algebra used

- `{P12, P5, P9, P14, P7, P10}` pairwise commute (rank 4 over GF(2)).
- Dependencies: `P7 = +P12 P5 P14`, `P10 = −P12 P9 P14`.
- `P4`, `P13` anticommute with part of the set → stay as generic Pauli rotations
  after the frame change.
- One Clifford `V` diagonalizes the whole commuting family at once:
  `U = V · (diagonal RZ / Z-parity ladders + 2 generic Pauli rots) · V†`.

## Results (unitary-verified to ~1e-15)

| | 2q gates | cross-chip | depth |
|--|--|--|--|
| **logical** (all-to-all) | **33** | — | **62** |
| **routed** on A–B ladder | **61** | **23** | **84** |

Still above the ~10 CZ aspiration: the Clifford sandwich alone is ~2 × 12 two-qubit
gates, and the conjugated `P4`/`P13` images stay weight 4–5. Natural next cuts:
drop near-zero `pid 13`, or run a Pauli-network synthesizer (e.g. rustiq).

## Machine embedding

From `qubit_connectivity/all_connection_cases.txt`, chips **A+B** with Case-0
A–B rungs, as a 6q ladder:

```
chip A:  0 — 1 — 2
         |   |   |     ← cross-chip rungs
chip B:  3 — 4 — 5
```

Cross-chip 2q options (`state_transfer/qst_formulas.txt`):

1. **QST** — full state transfer through the cable (~66 ns); spare qubit as ancilla.
2. **QST/2** — Bell pair `(|10⟩+|01⟩)/√2`, then gate teleportation.
3. **Native cross-chip RZX**.

## How to regenerate

```bash
.venv_py311/bin/python H4_circuits/build_h4_circuit.py
```

## Artifacts

| file | what |
|------|------|
| `build_h4_circuit.py` | frame search + verify |
| `H4_8doubles_diagframe.json` | gates, budgets, conjugated images, embedding |
| `H4_8doubles_diagframe_logical.png` | logical 6q circuit |
| `H4_8doubles_diagframe_routed.png` | routed onto A–B ladder |
