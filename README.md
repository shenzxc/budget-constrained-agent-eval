# Budget-Constrained Evaluation of Open-Weight LLM Agents

Replication package for the paper **"Budget-Constrained Evaluation of Open-Weight
LLM Agents: Success–Budget Curves, Price Reversal, and Thinking-Budget Saturation"**
(Weiming Shen, Independent Researcher, Suqian, Jiangsu, China).

We propose a budget-constrained evaluation protocol that treats a hard per-task
inference budget `B` as the independent variable and reports the success–budget
curve `S(B)` together with a family of derived metrics (AUBC, `B@τ`, elasticity,
crossover points). This repository contains the protocol implementation, the
analysis pipeline, all derived evaluation outputs, per-token cost records, and
the manuscript source.

## Repository layout

```
experiments/analysis/          protocol + analysis code (MIT)
  budget_curves.py             S(B)/AUBC/B@τ/bootstrap core
  run_analysis.py              τ³-bench metrics
  bfcl_curves.py               BFCL metrics
  cross_benchmark.py           cross-benchmark generalization
  make_figures.py              figures 1–5
  robust_subset.py             subset-sensitivity robustness (Appendix B)
  robust_cost.py               cost-accounting robustness (Appendix B)
  robust_power.py              statistical power / MDE (Appendix B)
  models.yaml                  model matrix + price table (with source URLs)
  output/                      summaries, tables, figures (CC BY 4.0)
稿件/latex/                     manuscript (LaTeX, elsarticle)
```

Third-party benchmark checkouts (`experiments/tau2-bench/`,
`experiments/gorilla/`), virtual environments, raw simulation logs, and private
billing files are intentionally excluded (see `.gitignore`).

## Reproducibility asset table

| Item | Value / source |
|---|---|
| Primary benchmark | τ³-bench (tau2-bench), domains: airline (50), retail (114), telecom (114); `base` split |
| Second benchmark | Berkeley Function-Calling Leaderboard (BFCL), `multi_turn_base`, fixed first 100 task IDs |
| Models (12) | DeepSeek-V4 flash/pro; Qwen3.7 max/plus; Qwen3.6-flash; Qwen3.5 397B-A17B/122B-A10B/35B-A3B/27B; Qwen3-32B; Qwen3-235B-A22B instruct/thinking |
| User simulator | DeepSeek-V4-Flash (fixed across all runs) |
| Judge | DeepSeek-V4-Pro (τ³ NL-assertion + env-interface components patched from OpenAI default; patch released) |
| Temperature | 0.0 (agent, user, judge) |
| Seed | 300 (τ³ simulations); 0 (bootstrap resampling) |
| Cost accounting | token count × official list price (cache-miss rate), verified 2026-07-04; sources in `models.yaml` |
| Exchange rate | 1 USD = 6.8067 CNY (CFETS, 2026-07-01) |
| Empty-response handling | deterministic empty agent responses counted as model task failures (see `run_analysis.py` `PHANTOM_FAILURES`); infrastructure errors excluded and reported |
| Retry / recovery | low-concurrency re-run with error-strip loop (`heal_loop.py`, `bfcl_finish.py`) |
| Qwen3-32B note | run with `enable_thinking=false` (DashScope non-streaming restriction); billed at non-thinking rate |
| Realized vs list cost | realized/list ratios recovered from provider bills (Appendix B); primary metric is list price |

## Running the analysis

Analysis is offline and reads the released outputs; no API calls or keys are
required to reproduce the figures and tables:

```bash
python experiments/analysis/run_analysis.py       # τ³ summaries
python experiments/analysis/bfcl_curves.py        # BFCL summaries
python experiments/analysis/cross_benchmark.py    # cross-benchmark τ + fig5
python experiments/analysis/make_figures.py       # figures 1–5
python experiments/analysis/robust_subset.py      # Appendix B
python experiments/analysis/robust_cost.py
python experiments/analysis/robust_power.py
```

Regenerating trajectories from scratch requires the two benchmark repositories,
API keys for the providers, and the run scripts (`run_matrix.py`, `bfcl_finish.py`).

## Data availability

All derived evaluation data (trajectory-level success and token/cost records,
summary tables, figures) are in `experiments/analysis/output/`. Raw per-task
simulation logs (~1 GB) are archived separately due to size. This repository is archived on Zenodo with a persistent DOI:
[10.5281/zenodo.21215799](https://doi.org/10.5281/zenodo.21215799).

## License

Code: MIT. Derived data: CC BY 4.0. See `LICENSE`.

## Citation

```bibtex
@misc{shen2026budget,
  title  = {Budget-Constrained Evaluation of Open-Weight LLM Agents:
            Success--Budget Curves, Price Reversal, and Thinking-Budget Saturation},
  author = {Shen, Weiming},
  year   = {2026},
  doi    = {10.5281/zenodo.21215799},
  note   = {Replication package, Zenodo}
}
```
