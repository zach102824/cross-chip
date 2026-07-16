from __future__ import annotations

import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _run_dir(
    data_dir: str | Path,
    molecule: str,
    bond_length: float,
    num_shots: int | None = None,
) -> Path:
    root = Path(data_dir)
    bond_dir = f"{molecule}_bond_{float(bond_length):.1f}"
    if num_shots is not None:
        return root / f"shots_{int(num_shots)}" / bond_dir
    return root / bond_dir


def _atomic_pickle(path: Path, payload: Any) -> None:
    """Write pickle via a same-directory temp file, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_vqe_progress(
    *,
    data_dir: str | Path,
    molecule: str,
    bond_length: float,
    payload: dict[str, Any],
    num_shots: int | None = None,
) -> Path:
    """Persist mid-VQE state so a stopped run can resume from the next iteration."""
    run_dir = _run_dir(data_dir, molecule, bond_length, num_shots=num_shots)
    path = run_dir / "vqe_progress.pkl"
    body = dict(payload)
    body["saved_utc"] = datetime.now(timezone.utc).isoformat()
    body["molecule"] = molecule
    body["bond_length"] = float(bond_length)
    if num_shots is not None:
        body["num_shots"] = int(num_shots)
    _atomic_pickle(path, body)
    # Small JSON sidecar for quick inspection (thetas / timings only).
    sidecar = {
        "saved_utc": body["saved_utc"],
        "molecule": molecule,
        "bond_length": float(bond_length),
        "completed_iters": int(body.get("completed_iters", 0)),
        "max_iters": int(body.get("max_iters", 0)),
        "theta": _jsonable(body.get("theta")),
        "best_theta": _jsonable(body.get("best_theta")),
        "best_E": _jsonable(body.get("best_E")),
        "prev_E": _jsonable(body.get("prev_E")),
        "iteration_timings": _jsonable(body.get("iteration_timings", [])),
    }
    _dump_json(run_dir / "vqe_progress.json", sidecar)
    print(
        f"[cloud-results] saved VQE progress "
        f"(completed_iters={sidecar['completed_iters']}/{sidecar['max_iters']}) under {run_dir}"
    )
    return path


def load_vqe_progress(
    *,
    data_dir: str | Path,
    molecule: str,
    bond_length: float,
    num_shots: int | None = None,
) -> dict[str, Any] | None:
    """Load mid-VQE checkpoint if present; otherwise return None."""
    path = _run_dir(data_dir, molecule, bond_length, num_shots=num_shots) / "vqe_progress.pkl"
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_checkpoint(
    *,
    data_dir: str | Path,
    molecule: str,
    bond_length: float,
    stage: str,
    vqe_results: Any | None = None,
    cme_results: Any | None = None,
    cme_results_by_multiplier: Any | None = None,
    metadata: dict[str, Any] | None = None,
    num_shots: int | None = None,
) -> dict[str, str]:
    """Save cloud-run artifacts in JSON plus pickle form.

    JSON is convenient for plotting and quick inspection; pickle preserves numpy
    arrays and any nested objects that JSON conversion simplifies.

    Layout when ``num_shots`` is set (preferred for new runs)::

        data/shots_<N>/<molecule>_bond_<R>/

    Without ``num_shots``, keeps the legacy layout ``data/<molecule>_bond_<R>/``.
    """
    root = Path(data_dir)
    bond_dir = f"{molecule}_bond_{float(bond_length):.1f}"
    if num_shots is not None:
        run_dir = root / f"shots_{int(num_shots)}" / bond_dir
    else:
        run_dir = root / bond_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "molecule": molecule,
        "bond_length": float(bond_length),
        "num_shots": int(num_shots) if num_shots is not None else None,
        "stage": stage,
        "metadata": metadata or {},
        "has_vqe_results": vqe_results is not None,
        "has_cme_results": cme_results is not None,
        "has_cme_results_by_multiplier": cme_results_by_multiplier is not None,
    }
    _dump_json(run_dir / "run_summary.json", summary)

    written: dict[str, str] = {"run_summary": str(run_dir / "run_summary.json")}
    if vqe_results is not None:
        _dump_json(run_dir / "vqe_results.json", vqe_results)
        _dump_pickle(run_dir / "vqe_results.pkl", vqe_results)
        written["vqe_json"] = str(run_dir / "vqe_results.json")
        written["vqe_pickle"] = str(run_dir / "vqe_results.pkl")
    if cme_results_by_multiplier is not None:
        _dump_json(run_dir / "cme_results_by_multiplier.json", cme_results_by_multiplier)
        _dump_pickle(run_dir / "cme_results_by_multiplier.pkl", cme_results_by_multiplier)
        written["cme_by_multiplier_json"] = str(run_dir / "cme_results_by_multiplier.json")
        written["cme_by_multiplier_pickle"] = str(run_dir / "cme_results_by_multiplier.pkl")
    if cme_results is not None:
        _dump_json(run_dir / "cme_results.json", cme_results)
        _dump_pickle(run_dir / "cme_results.pkl", cme_results)
        written["cme_json"] = str(run_dir / "cme_results.json")
        written["cme_pickle"] = str(run_dir / "cme_results.pkl")

    _dump_pickle(
        run_dir / "all_results.pkl",
        {
            "summary": summary,
            "vqe_results": vqe_results,
            "cme_results": cme_results,
            "cme_results_by_multiplier": cme_results_by_multiplier,
        },
    )
    written["all_pickle"] = str(run_dir / "all_results.pkl")
    print(f"[cloud-results] saved {stage} artifacts under {run_dir}")
    return written
