# Cloud Export for `main_HF.ipynb`

This folder is self-contained for running the exported VQE workflow on a cloud VM.

Run one molecule from this directory:

```bash
python main_HF.py
python main_Br2.py
python main_Cl2.py
```

Results are saved under `data/<molecule>_bond_<bond>/` as JSON and pickle files. By default the scripts use the same `ogm` measurement setting as the notebook. OGM basis files are read from the local `June_main/OGM_measurement_basis/` folder inside this export directory. No external `shadowgrouping` checkout is required for OGM runs.

`main_Br2.py` will save VQE data even if H^2/H^3 files are unavailable for CMX. Existing Hamiltonian and OGM files from the repo are copied into `Pauli_Ham/` and `June_main/OGM_measurement_basis/`.

For a quick smoke test, reduce the expensive knobs without changing the script defaults:

```bash
VQE_ITERS=1 GLOBAL_NUM_SHOTS=128 CDR_NUM_TRAINING_CIRCUITS=2 python main_HF.py
```
