"""分析驱动脚本:扫描全部模拟结果,产出每(模型,域)的预算条件化指标汇总。

输出:
  analysis/output/summary.json   全部指标(含S(B)曲线数据)
  analysis/output/summary.csv    表格摘要(不含曲线)
用法:
  .venv/bin/python analysis/run_analysis.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

import budget_curves as bc

EXP = Path(__file__).resolve().parent.parent
SIMS = EXP / "tau2-bench" / "data" / "simulations"
OUT = Path(__file__).resolve().parent / "output"
DOMAINS = ("airline", "retail", "telecom")

# 确定性空回复失败(agent侧AssistantMessage为空,4次重试×6轮补跑均复现,2026-07-05核实):
# 计为被测模型的任务失败,注入零成本失败轨迹进入分母。
# 注意:qwen3.7-plus在telecom另有2条UserMessage(用户模拟器)空回复,属评测基础设施问题,
# 不注入、直接从分母排除(340/342),论文如实报告。
PHANTOM_FAILURES = {
    ("qwen3.5-35b-a3b", "airline"): 14,
    ("qwen3.5-35b-a3b", "retail"): 1,
}


def main():
    cfg = yaml.safe_load((Path(__file__).parent / "models.yaml").read_text())
    prices = bc.load_price_table()
    id_by_litellm = {m["litellm"]: m["id"] for m in cfg["models"]}

    # 收集全部轨迹: {(model_id, domain): [Trajectory]}
    groups: dict[tuple[str, str], list[bc.Trajectory]] = {}
    for d in sorted(SIMS.iterdir()):
        if d.name[0].isdigit() or not (d / "results.json").exists():
            continue
        domain = d.name.split("_")[0]
        if domain not in DOMAINS:
            continue
        litellm_name = d.name[len(domain) + 1 :].replace("_", "/", 1)
        model_id = id_by_litellm.get(litellm_name)
        if model_id is None or model_id not in prices:
            print(f"[跳过] {d.name}: 未知模型或缺价格 ({litellm_name})")
            continue
        trajs = bc.load_tau2_dir(d, model_id, prices[model_id])
        groups.setdefault((model_id, domain), []).extend(trajs)

    # 注入确定性失败(见PHANTOM_FAILURES说明)
    for (model_id, domain), n in PHANTOM_FAILURES.items():
        if (model_id, domain) in groups:
            for i in range(n):
                groups[(model_id, domain)].append(
                    bc.Trajectory(model=model_id, domain=domain,
                                  task_id=f"phantom_{i}", trial=0,
                                  success=False, valid=True)
                )

    # 全域统一预算网格(每个域一个网格,便于域内模型比较)
    grids = {
        dom: bc.budget_grid([t for (m, dd), ts in groups.items() if dd == dom for t in ts])
        for dom in DOMAINS
    }

    summary = {}
    for (model_id, domain), trajs in sorted(groups.items()):
        s = bc.summarize(trajs, grids[domain])
        summary[f"{model_id}|{domain}"] = s
        print(
            f"{model_id:24s} {domain:8s} n={s['n_valid']:4d} "
            f"pass1={s['pass1']:.3f} cost=${s['mean_cost']:.4f} "
            f"AUBC={s['aubc']:.3f} B@50%={s['b_at_50'] and round(s['b_at_50'],4)}"
        )

    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "domain", "n_valid", "pass1", "mean_cost_usd", "aubc",
                    "aubc_lo", "aubc_hi", "b_at_50", "b_at_80"])
        for key, s in summary.items():
            m, dom = key.split("|")
            w.writerow([m, dom, s["n_valid"], round(s["pass1"], 4),
                        round(s["mean_cost"], 6), round(s["aubc"], 4),
                        round(s["aubc_ci95"][0], 4), round(s["aubc_ci95"][1], 4),
                        s["b_at_50"], s["b_at_80"]])
    print(f"\n已写出 {OUT/'summary.json'} 与 summary.csv")


if __name__ == "__main__":
    main()
