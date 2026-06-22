import importlib.util
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "gen_cl2", str(HERE / "improved create UCCSD circuit_Cl2.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

from qiskit.circuit import Parameter

doubles = [(5, 11, 8, 2), (5, 11, 7, 1), (5, 11, 9, 3),
           (5, 11, 10, 4), (5, 11, 6, 0)]
num_qubits = 12
thetas = [Parameter(f"t{d}") for d in range(len(doubles))]
qc, strings, signs, theta_idx = gen.create_uccsd_circuit(
    num_qubits, doubles, thetas=thetas, optimize=True, order="auto",
    pair=False, init_state=None)

cz_pairs = []
for inst in qc.data:
    if inst.operation.name == "cz":
        a, b = (qc.find_bit(q).index for q in inst.qubits)
        cz_pairs.append(tuple(sorted((a, b))))

print("total CZ:", len(cz_pairs))
print("distinct CZ edges and multiplicity:")
for edge, c in sorted(Counter(cz_pairs).items()):
    print(f"  {edge}: {c}")

# Physical chip graph from the drawing.
chip_edges = {
    # block A {6,7,8,9}
    (8, 9), (7, 8), (6, 7), (6, 9),
    # block B {4,5,10,11}
    (10, 11), (4, 10), (5, 11), (4, 5),
    # block C {0,1,2,3}
    (0, 3), (2, 3), (0, 1), (1, 2),
    # cross-chip (high error)
    (9, 10), (3, 4),
}
chip_edges = {tuple(sorted(e)) for e in chip_edges}
bad_edges = {tuple(sorted(e)) for e in [(9, 10), (3, 4)]}

on_graph = [e for e in cz_pairs if e in chip_edges]
off_graph = [e for e in cz_pairs if e not in chip_edges]
on_bad = [e for e in cz_pairs if e in bad_edges]
print("\nUnder identity placement (orbital i -> physical i):")
print("  CZ on a chip edge:", len(on_graph))
print("  CZ NOT on any chip edge (need routing):", len(off_graph),
      Counter(off_graph))
print("  CZ on high-error edges:", len(on_bad), Counter(on_bad))
