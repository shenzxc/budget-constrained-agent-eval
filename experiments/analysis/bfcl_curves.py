"""BFCL budget-conditioned metrics for the second benchmark.

This script converts the BFCL multi_turn_base long table into the same
budget-curve summary schema used for tau2-bench in budget_curves.py.
Each BFCL row is a single completed task, so its task cost is represented
as one trajectory step.

Usage:
  experiments/.venv/bin/python experiments/analysis/bfcl_curves.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from budget_curves import (
    Trajectory,
    aubc,
    bootstrap_aubc_ci,
    budget_at_tau,
    load_price_table,
    s_of_b,
)

ANALYSIS_DIR = Path(__file__).resolve().parent
OUT_DIR = ANALYSIS_DIR / "output"
INPUT_CSV = OUT_DIR / "bfcl_multi_turn_base.csv"
SUMMARY_JSON = OUT_DIR / "bfcl_summary.json"
SUMMARY_TABLE = OUT_DIR / "bfcl_summary_table.csv"
GRID_POINTS = 200
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0

SPECIAL_MODEL_IDS = {
    "qwen3-235b-a22b-instruct-2507": "qwen3-235b-instruct",
    "qwen3-235b-a22b-thinking-2507": "qwen3-235b-thinking",
}


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_int(value: str) -> int:
    if value is None or str(value).strip() == "":
        return 0
    return int(float(value))


def bfcl_name_to_model_id(name: str) -> str:
    model = str(name or "").strip()
    if "/" in model:
        model = model.rsplit("/", 1)[-1]
    if model.endswith("-FC"):
        model = model[:-3]
    return SPECIAL_MODEL_IDS.get(model, model)


def normalize_model_id(row: dict[str, str], price_table: dict) -> str:
    candidates = [row.get("model", ""), row.get("bfcl_model", "")]
    for candidate in candidates:
        model_id = bfcl_name_to_model_id(candidate)
        if model_id in price_table:
            return model_id
    tried = ", ".join(repr(c) for c in candidates if c)
    raise KeyError(f"No price entry found for BFCL model fields: {tried}")


def load_bfcl_trajs(csv_path: Path = INPUT_CSV) -> list[Trajectory]:
    price_table = load_price_table()
    trajs: list[Trajectory] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            model_id = normalize_model_id(row, price_table)
            price = price_table[model_id]
            prompt_tokens = parse_int(row.get("total_input_tokens", "0"))
            completion_tokens = parse_int(row.get("total_output_tokens", "0"))
            cost = (
                prompt_tokens / 1e6 * price.input_per_m
                + completion_tokens / 1e6 * price.output_per_m
            )
            trajs.append(
                Trajectory(
                    model=model_id,
                    domain="bfcl_multi_turn_base",
                    task_id=str(row.get("task_id", "")),
                    trial=0,
                    success=parse_bool(row.get("success", "False")),
                    valid=True,
                    steps_cost=[float(cost)],
                    steps_ctok=[completion_tokens],
                    steps_ptok=[prompt_tokens],
                )
            )
    return trajs


def bfcl_budget_grid(trajs: list[Trajectory], n: int = GRID_POINTS) -> np.ndarray:
    costs = [t.total_cost for t in trajs if t.valid and t.total_cost > 0]
    if not costs:
        raise ValueError("BFCL input has no positive task costs; cannot build log grid")
    lo, hi = min(costs), max(costs)
    return np.logspace(math.log10(lo * 0.5), math.log10(hi * 1.05), n)


def summarize_model(trajs: list[Trajectory], grid: np.ndarray) -> dict:
    valid = [t for t in trajs if t.valid]
    if not valid:
        raise ValueError("Cannot summarize an empty BFCL model group")

    s = s_of_b(valid, grid)
    if np.any(np.diff(s) < -1e-12):
        raise AssertionError("S(B) must be monotonically non-decreasing")

    pass1 = float(np.mean([t.success for t in valid]))
    if not np.isclose(s[-1], pass1, atol=1e-12):
        raise AssertionError(f"S(B_max)={s[-1]} does not match pass1={pass1}")

    lo, hi = bootstrap_aubc_ci(
        valid, grid, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED
    )
    return {
        "n": len(valid),
        "pass1": pass1,
        "mean_cost_usd": float(np.mean([t.total_cost for t in valid])),
        "aubc": aubc(grid, s),
        "aubc_ci95": [lo, hi],
        "b_at_50": budget_at_tau(grid, s, 0.5),
        "b_at_80": budget_at_tau(grid, s, 0.8),
        "grid": grid.tolist(),
        "s_of_b": s.tolist(),
    }


def fmt_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def write_summary_table(summary: dict[str, dict], path: Path) -> None:
    fieldnames = [
        "model",
        "n",
        "pass1",
        "mean_cost_usd",
        "aubc",
        "aubc_lo",
        "aubc_hi",
        "b_at_50",
        "b_at_80",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for model, row in sorted(summary.items()):
            lo, hi = row["aubc_ci95"]
            writer.writerow(
                {
                    "model": model,
                    "n": row["n"],
                    "pass1": f"{row['pass1']:.10g}",
                    "mean_cost_usd": f"{row['mean_cost_usd']:.10g}",
                    "aubc": f"{row['aubc']:.10g}",
                    "aubc_lo": f"{lo:.10g}",
                    "aubc_hi": f"{hi:.10g}",
                    "b_at_50": fmt_optional(row["b_at_50"]),
                    "b_at_80": fmt_optional(row["b_at_80"]),
                }
            )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajs = load_bfcl_trajs()
    grid = bfcl_budget_grid(trajs)

    by_model: dict[str, list[Trajectory]] = defaultdict(list)
    for traj in trajs:
        by_model[traj.model].append(traj)

    summary = {
        model: summarize_model(model_trajs, grid)
        for model, model_trajs in sorted(by_model.items())
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_table(summary, SUMMARY_TABLE)
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_TABLE}")


if __name__ == "__main__":
    main()
