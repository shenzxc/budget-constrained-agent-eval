"""Cross-benchmark generalization analysis for tau2-bench and BFCL.

Outputs:
  - output/cross_benchmark.json
  - output/figures/fig5_cross_benchmark.pdf
  - output/figures/fig5_cross_benchmark.png
  - output/figures/fig5_note.txt

Usage:
  experiments/.venv/bin/python experiments/analysis/cross_benchmark.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, rankdata

try:
    from adjustText import adjust_text

    HAVE_ADJUST_TEXT = True
except ImportError:
    HAVE_ADJUST_TEXT = False

ANALYSIS_DIR = Path(__file__).resolve().parent
OUT_DIR = ANALYSIS_DIR / "output"
FIG_DIR = OUT_DIR / "figures"
TAU3_SUMMARY_JSON = OUT_DIR / "summary.json"
BFCL_SUMMARY_JSON = OUT_DIR / "bfcl_summary.json"
CROSS_JSON = OUT_DIR / "cross_benchmark.json"
FIG_PDF = FIG_DIR / "fig5_cross_benchmark.pdf"
FIG_PNG = FIG_DIR / "fig5_cross_benchmark.png"
NOTE_TXT = FIG_DIR / "fig5_note.txt"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_tau3(summary: dict) -> dict[str, dict]:
    by_model: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"aubc": [], "pass1": []}
    )
    for key, row in summary.items():
        model, _domain = key.split("|", 1)
        by_model[model]["aubc"].append(float(row["aubc"]))
        by_model[model]["pass1"].append(float(row["pass1"]))

    return {
        model: {
            "aubc": float(np.mean(vals["aubc"])),
            "pass1": float(np.mean(vals["pass1"])),
            "n_domains": len(vals["aubc"]),
        }
        for model, vals in by_model.items()
        if vals["aubc"] and vals["pass1"]
    }


def descending_ranks(values: dict[str, float]) -> dict[str, float]:
    models = list(values)
    scores = np.asarray([values[m] for m in models], dtype=float)
    ranks = rankdata(-scores, method="average")
    return {m: float(r) for m, r in zip(models, ranks)}


def kendall_for_ranks(
    common_models: list[str], left_ranks: dict[str, float], right_ranks: dict[str, float]
) -> tuple[float | None, float | None]:
    if len(common_models) < 2:
        return None, None
    result = kendalltau(
        [left_ranks[m] for m in common_models],
        [right_ranks[m] for m in common_models],
    )
    tau = None if math.isnan(result.statistic) else float(result.statistic)
    p_value = None if math.isnan(result.pvalue) else float(result.pvalue)
    return tau, p_value


def build_cross_summary(tau3: dict[str, dict], bfcl: dict[str, dict]) -> dict:
    common = sorted(set(tau3) & set(bfcl))
    tau3_aubc = {m: tau3[m]["aubc"] for m in common}
    bfcl_aubc = {m: float(bfcl[m]["aubc"]) for m in common}
    tau3_pass1 = {m: tau3[m]["pass1"] for m in common}
    bfcl_pass1 = {m: float(bfcl[m]["pass1"]) for m in common}

    tau3_aubc_rank = descending_ranks(tau3_aubc)
    bfcl_aubc_rank = descending_ranks(bfcl_aubc)
    tau3_pass1_rank = descending_ranks(tau3_pass1)
    bfcl_pass1_rank = descending_ranks(bfcl_pass1)

    aubc_tau, aubc_p = kendall_for_ranks(common, tau3_aubc_rank, bfcl_aubc_rank)
    pass1_tau, pass1_p = kendall_for_ranks(common, tau3_pass1_rank, bfcl_pass1_rank)

    rankings = []
    for model in sorted(common, key=lambda m: tau3_aubc_rank[m]):
        rankings.append(
            {
                "model": model,
                "tau3_aubc": tau3_aubc[model],
                "bfcl_aubc": bfcl_aubc[model],
                "tau3_aubc_rank": tau3_aubc_rank[model],
                "bfcl_aubc_rank": bfcl_aubc_rank[model],
                "tau3_pass1": tau3_pass1[model],
                "bfcl_pass1": bfcl_pass1[model],
                "tau3_pass1_rank": tau3_pass1_rank[model],
                "bfcl_pass1_rank": bfcl_pass1_rank[model],
            }
        )

    return {
        "n_shared": len(common),
        "shared_models": common,
        "aubc_kendall_tau": aubc_tau,
        "aubc_kendall_p": aubc_p,
        "pass1_kendall_tau": pass1_tau,
        "pass1_kendall_p": pass1_p,
        "rankings": rankings,
    }


def axis_limits(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    lo = float(min(xs.min(), ys.min()))
    hi = float(max(xs.max(), ys.max()))
    if math.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 0.01)
    else:
        pad = (hi - lo) * 0.08
    return max(0.0, lo - pad), min(1.0, hi + pad)


def plot_cross_benchmark(cross: dict) -> None:
    rows = cross["rankings"]
    if not rows:
        raise ValueError("No shared tau2/BFCL models to plot")

    xs = np.asarray([r["tau3_aubc"] for r in rows], dtype=float)
    ys = np.asarray([r["bfcl_aubc"] for r in rows], dtype=float)
    models = [r["model"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.scatter(xs, ys, s=28, color="#276fbf", edgecolor="white", linewidth=0.5, zorder=3)
    texts = []
    for x, y, model in zip(xs, ys, models):
        texts.append(ax.text(x, y, model, fontsize=6.5, ha="left", va="bottom"))

    lo, hi = axis_limits(xs, ys)
    ax.plot([lo, hi], [lo, hi], color="0.35", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("tau3-bench mean AUBC")
    ax.set_ylabel("BFCL AUBC")
    ax.grid(True, which="major", linewidth=0.3, alpha=0.4)
    if HAVE_ADJUST_TEXT:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.3, color="0.5"))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG)
    plt.close(fig)


def fmt_stat(value: float | None) -> str:
    return "nan" if value is None else f"{value:.2f}"


def write_note(cross: dict) -> None:
    note = (
        f"Kendall \u03c4={fmt_stat(cross['aubc_kendall_tau'])} "
        f"(p={fmt_stat(cross['aubc_kendall_p'])}) between \u03c4\u00b3 "
        f"and BFCL AUBC rankings over {cross['n_shared']} shared models"
    )
    NOTE_TXT.write_text(note + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tau3 = aggregate_tau3(load_json(TAU3_SUMMARY_JSON))
    bfcl = load_json(BFCL_SUMMARY_JSON)
    cross = build_cross_summary(tau3, bfcl)
    CROSS_JSON.write_text(json.dumps(cross, indent=2), encoding="utf-8")
    plot_cross_benchmark(cross)
    write_note(cross)
    print(f"Wrote {CROSS_JSON}")
    print(f"Wrote {FIG_PDF}")
    print(f"Wrote {FIG_PNG}")
    print(f"Wrote {NOTE_TXT}")


if __name__ == "__main__":
    main()
