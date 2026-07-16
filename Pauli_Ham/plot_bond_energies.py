#!/usr/bin/env python3
"""Plot HF and ground-state energies vs bond length for HF, Cl2, Br2.

Reads ``*_bond_scan_summary.txt`` in this folder and writes a three-panel
figure matching the style of a typical PES comparison (HF vs exact GS).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# (summary file, panel label, title with qubit count)
CASES = (
    ("HF_bond_scan_summary.txt", "a", r"HF 8Q"),
    ("Cl2_bond_scan_summary.txt", "b", r"Cl$_2$ 10Q"),
    ("Br2_bond_scan_summary.txt", "c", r"Br$_2$ 12Q"),
)

HF_COLOR = "#2ca02c"
GS_COLOR = "#7fdbda"
FACE = "#dce8d4"


def load_summary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bonds, e_hf, e_gs = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("bond_"):
                continue
            parts = line.split("\t")
            if len(parts) < 3 or not parts[2]:
                continue
            bonds.append(float(parts[0]))
            e_hf.append(float(parts[1]))
            e_gs.append(float(parts[2]))
    return np.asarray(bonds), np.asarray(e_hf), np.asarray(e_gs)


def annotate_curve(ax, x, y, text: str, color: str, *, x_frac: float, y_off: float) -> None:
    """Place an in-axes label near a point on the curve."""
    i = int(np.clip(round(x_frac * (len(x) - 1)), 0, len(x) - 1))
    ax.text(
        x[i],
        y[i] + y_off,
        text,
        color=color,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
    )


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)

    # Per-panel label placement: (HF x_frac, HF y_off_frac, GS x_frac, GS y_off_frac)
    label_pos = {
        "a": (0.75, 0.12, 0.40, -0.12),
        "b": (0.85, 0.08, 0.55, -0.10),
        "c": (0.80, 0.10, 0.45, -0.12),
    }

    for ax, (fname, letter, title) in zip(axes, CASES):
        bonds, e_hf, e_gs = load_summary(HERE / fname)

        ax.set_facecolor(FACE)
        ax.plot(bonds, e_hf, color=HF_COLOR, lw=2.0, zorder=2)
        ax.plot(bonds, e_gs, color=GS_COLOR, lw=2.0, zorder=2)
        ax.plot(bonds, e_hf, "o", color=HF_COLOR, ms=4, zorder=3)
        ax.plot(bonds, e_gs, "o", color=GS_COLOR, ms=4, zorder=3)

        y_all = np.concatenate([e_hf, e_gs])
        y_span = float(np.ptp(y_all))
        pad = 0.08 * y_span if y_span > 0 else 0.01
        hf_xf, hf_yf, gs_xf, gs_yf = label_pos[letter]
        annotate_curve(ax, bonds, e_hf, "HF", HF_COLOR, x_frac=hf_xf, y_off=hf_yf * y_span)
        annotate_curve(ax, bonds, e_gs, "Ground", GS_COLOR, x_frac=gs_xf, y_off=gs_yf * y_span)

        ax.set_title(f"{letter}   {title}", loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel(r"Bond length ($\mathrm{\AA}$)")
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(False)
        # Avoid ugly offset notation on large absolute energies (e.g. Br2).
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        # Small padding so markers / labels are not clipped.
        ax.set_ylim(y_all.min() - pad, y_all.max() + 1.5 * pad)

    axes[0].set_ylabel("Energy (Hartree)")

    out = HERE / "bond_energy_HF_GS.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
