#!/usr/bin/env python3
"""BFCL multi_turn_base 结果提取脚本(第二基准, 固定前100任务)。

从 gorilla/berkeley-function-call-leaderboard 的 result/ 和 score/ 目录提取:
1. 长表 CSV (analysis/output/bfcl_multi_turn_base.csv), 每行 = 一个 (model, task_id):
   model(models.yaml id), bfcl_model(BFCL条目名), task_id, success(score判分),
   n_requests(模型API调用步数, 即 turn x step 叶子数),
   total_input_tokens, total_output_tokens, total_latency_s
2. 汇总表 (analysis/output/bfcl_multi_turn_base_summary.csv), 每模型:
   n_tasks, accuracy, avg_input_tokens_per_task, avg_output_tokens_per_task,
   total_cost_usd, total_cost_cny (单价来自 analysis/models.yaml, CNY按其汇率字段折算)

只统计 multi_turn_base 前100个任务ID (multi_turn_base_0 .. multi_turn_base_99),
与本次全量跑批的固定前缀口径一致。

用法: .venv/bin/python analysis/bfcl_extract.py
"""
import csv
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parent.parent
BFCL_ROOT = EXP / "gorilla/berkeley-function-call-leaderboard"
OUT_DIR = EXP / "analysis/output"
LONG_CSV = OUT_DIR / "bfcl_multi_turn_base.csv"
SUMMARY_CSV = OUT_DIR / "bfcl_multi_turn_base_summary.csv"

N_TASKS = 100
FIRST100 = [f"multi_turn_base_{i}" for i in range(N_TASKS)]
FIRST100_SET = set(FIRST100)

# BFCL模型条目名(-FC后缀前) -> models.yaml 的 id
YAML_ID = {
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "qwen3.7-max": "qwen3.7-max",
    "qwen3.7-plus": "qwen3.7-plus",
    "qwen3.6-flash": "qwen3.6-flash",
    "qwen3.5-397b-a17b": "qwen3.5-397b-a17b",
    "qwen3.5-122b-a10b": "qwen3.5-122b-a10b",
    "qwen3.5-35b-a3b": "qwen3.5-35b-a3b",
    "qwen3.5-27b": "qwen3.5-27b",
    "qwen3-32b": "qwen3-32b",
    "qwen3-235b-a22b-instruct-2507": "qwen3-235b-instruct",
    "qwen3-235b-a22b-thinking-2507": "qwen3-235b-thinking",
}

BFCL_MODELS = [f"{base}-FC" for base in YAML_ID]


def flat_sum(x):
    """list[list] (turn x step) 或标量 -> 总和。"""
    if isinstance(x, list):
        return sum(flat_sum(e) for e in x)
    return x if isinstance(x, (int, float)) else 0


def leaf_count(x):
    """list[list] 叶子个数 = 模型API调用步数。"""
    if isinstance(x, list):
        return sum(leaf_count(e) for e in x)
    return 1


def load_prices():
    with open(EXP / "analysis/models.yaml") as f:
        cfg = yaml.safe_load(f)
    rate = cfg["exchange_rate_usd_cny"]
    prices = {m["id"]: m for m in cfg["models"]}
    return prices, rate


def load_results(bfcl_model):
    """result文件 -> {task_id: entry}, 只保留前100任务。"""
    rf = (BFCL_ROOT / "result" / bfcl_model / "multi_turn"
          / "BFCL_v4_multi_turn_base_result.json")
    out = {}
    if not rf.exists():
        return out
    with open(rf) as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in FIRST100_SET:
                out[d["id"]] = d
    return out


def load_failures(bfcl_model):
    """score文件 -> (failed_id_set, summary_dict or None)。首行是汇总, 其余行是失败条目。"""
    sf = (BFCL_ROOT / "score" / bfcl_model / "multi_turn"
          / "BFCL_v4_multi_turn_base_score.json")
    if not sf.exists():
        return None, None
    failed = set()
    summary = None
    with open(sf) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i == 0:
                summary = d
            else:
                failed.add(d["id"])
    return failed, summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices, rate = load_prices()

    long_rows = []
    summary_rows = []
    problems = []

    for bfcl_model in BFCL_MODELS:
        base = bfcl_model[: -len("-FC")]
        yid = YAML_ID[base]
        results = load_results(bfcl_model)
        failed, score_summary = load_failures(bfcl_model)

        if not results:
            problems.append(f"{bfcl_model}: result文件缺失或前100任务无结果, 跳过")
            continue
        if failed is None:
            problems.append(f"{bfcl_model}: score文件缺失, success留空")

        tin_total = tout_total = 0
        n_success = n_scored = 0
        for tid in FIRST100:
            if tid not in results:
                problems.append(f"{bfcl_model}: 缺任务 {tid}")
                continue
            d = results[tid]
            tin = flat_sum(d.get("input_token_count"))
            tout = flat_sum(d.get("output_token_count"))
            lat = flat_sum(d.get("latency"))
            nreq = leaf_count(d.get("input_token_count"))
            if failed is None:
                success = ""
            else:
                success = tid not in failed
                n_scored += 1
                n_success += int(success)
            tin_total += tin
            tout_total += tout
            long_rows.append({
                "model": yid,
                "bfcl_model": bfcl_model,
                "task_id": tid,
                "success": success,
                "n_requests": nreq,
                "total_input_tokens": tin,
                "total_output_tokens": tout,
                "total_latency_s": round(lat, 3),
            })

        n = len(results)
        p = prices[yid]
        ip, op, cur = p["input_price"], p["output_price"], p["currency"]
        cost_native = (tin_total / 1e6) * ip + (tout_total / 1e6) * op
        if cur == "USD":
            cost_usd, cost_cny = cost_native, cost_native * rate
        else:
            cost_usd, cost_cny = cost_native / rate, cost_native
        summary_rows.append({
            "model": yid,
            "bfcl_model": bfcl_model,
            "n_tasks": n,
            "accuracy": round(n_success / n_scored, 4) if n_scored else "",
            "avg_input_tokens_per_task": round(tin_total / n, 1),
            "avg_output_tokens_per_task": round(tout_total / n, 1),
            "total_input_tokens": tin_total,
            "total_output_tokens": tout_total,
            "total_cost_usd": round(cost_usd, 4),
            "total_cost_cny": round(cost_cny, 2),
        })

    with open(LONG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "bfcl_model", "task_id", "success", "n_requests",
            "total_input_tokens", "total_output_tokens", "total_latency_s"])
        w.writeheader()
        w.writerows(long_rows)

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "bfcl_model", "n_tasks", "accuracy",
            "avg_input_tokens_per_task", "avg_output_tokens_per_task",
            "total_input_tokens", "total_output_tokens",
            "total_cost_usd", "total_cost_cny"])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"长表: {LONG_CSV} ({len(long_rows)} 行)")
    print(f"汇总: {SUMMARY_CSV} ({len(summary_rows)} 模型)")
    total_cny = sum(r["total_cost_cny"] for r in summary_rows)
    print(f"总成本: {total_cny:.2f} CNY")
    if problems:
        print(f"\n异常({len(problems)}):")
        for p_ in problems[:40]:
            print(" -", p_)


if __name__ == "__main__":
    main()
