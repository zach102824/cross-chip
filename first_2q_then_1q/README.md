# first_2q then 1q — approximate UCCSD doubles

`exploring/` compiles doubles exactly (CZ ladders + RZX, Z-freeze).
Here we ask a different question:

> Can a **sparse 2q scaffold** (RZX / CZ) plus **free single-qubit layers**
> match the doubles state `U(θ)|HF⟩` closely?

## Structure

```
|HF⟩  →  [optional U3]  →  RZX/CZ on chosen pairs  →  [U3]×reps
```

Metric: statevector overlap `|⟨ψ_doubles|ψ_ansatz⟩|²` at random `θ ∈ [-0.3,0.3]`.

## Run

```bash
.venv_h4_tencirchem/bin/python first_2q_then_1q/run_approx_doubles.py --quick
.venv_h4_tencirchem/bin/python first_2q_then_1q/run_approx_doubles.py --case HF_6q
.venv_h4_tencirchem/bin/python first_2q_then_1q/run_approx_doubles.py --case Cl2_10q
```

## Why HF vs Cl2 differ

| case | fully-frozen Paulis | exact exploring 2q |
|---|---|---|
| HF 6q | weight-2 (`YIIXII`…) — one vertical pair each | 3 RZX |
| Cl2 10q | weight-4 (`YIIIXXIIIX`…) — two vertical pairs | many CZ + 3 RZX |

Weight-2 doubles are exactly `1q · RZX · 1q`. Weight-4 need entanglement across four qubits, so sparse disjoint RZX + local 1q cannot be exact.
