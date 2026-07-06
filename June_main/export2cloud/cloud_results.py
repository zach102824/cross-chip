from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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
) -> dict[str, str]:
    """Save cloud-run artifacts in JSON plus pickle form.

    JSON is convenient for plotting and quick inspection; pickle preserves numpy
    arrays and any nested objects that JSON conversion simplifies.
    """
    root = Path(data_dir)
    run_dir = root / f"{molecule}_bond_{float(bond_length):.1f}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "molecule": molecule,
        "bond_length": float(bond_length),
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
