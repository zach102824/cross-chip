# Omitting CZ(3,4) / CZ(8,9): noiseless energy check

## Setup
- **Cl2** bond 2.2 Å, 8e/5o → 10q; **Br2** bond 2.2 Å, 10e/6o → 12q
- Exact active-space GS from Hamiltonian diagonalisation
- HF prep + noiseless statevector VQE (Nelder–Mead, multi-start)

## Reference: full double-excitation ansatz (keeps boundary CZ)

| molecule | E_HF − E_GS | E_VQE − E_GS | correlation recovered |
|---|---|---|---|
| Cl2 | 38.98 mHa | **1.83 mHa** | **95.3%** |
| Br2 | 17.82 mHa | **0.35 mHa** | **98.1%** |

So the current UCCSD-doubles circuits (with CZ on (3,4)/(8,9) as needed) work well.

## Approximations that avoid CZ(3,4) & CZ(8,9)

| variant | idea | Cl2 E_VQE − E_GS | corr. recovered |
|---|---|---|---|
| Drop boundary CZ gates from exact circuit | delete CZ(3,4)/(8,9) only | **38.98 mHa (= HF)** | **0%** |
| Strip LUMO from Paulis + free RZX(4,9) | omit same-spin LUMO fan-in; keep α–β RZX “beyond q3” | **38.98 mHa (= HF)** | **0%** |
| Product: Π RZX(occα,occβ) × RZX(LUMO) | factor 4-body into 2-body | **38.98 mHa (= HF)** | **0%** |

Br2 shows the same pattern (approx → stuck at HF).

## Why it fails
Each selected double is a **4-local** paired excitation
`a†_{Lα} a†_{Lβ} a_{kβ} a_{kα}`.
Its odd-Y Pauli flips **both** the occupied pair and the LUMO pair.

- Same-spin CZ into the LUMO (the (3,4)/(8,9) edges on Cl2) is what **ties** those flips into one Pauli frame.
- Removing those CZs (or replacing the 4-body with a product of two RZX) yields a unitary that **does not** implement the double excitation on |HF⟩, so VQE cannot pick up the correlation.
- With the CZ graph that excludes (3,4), **q4 is isolated** in the α row (only possible CZ neighbour was q3). Any orbital wired to q4 can only touch the rest of α via RZX to β — not enough for the 4-body generator.

## Takeaway
**Not viable as an approximation for these ansätze:** omitting same-spin boundary CZ (and relying on RZX beyond q3 alone) returns to the HF energy. To drop (3,4)/(8,9) you need a different connectivity (another edge into q4/q9) or a different excitation pool — not a silent deletion of those CZs.

## Artifacts
- `approx_omit_boundary_cz_vqe.py` — strip-LUMO / product experiments
- `Cl2_drop_boundary_cz_results.json` — drop-CZ vs exact VQE numbers
- `Cl2_approx_omit_boundary_cz_results.json`, `Br2_approx_omit_boundary_cz_results.json`
