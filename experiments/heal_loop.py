"""自愈循环:清理infrastructure_error→低并发补跑→重复,直到数据完整或轮次耗尽。"""
import json
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
SIMS = EXP / "tau2-bench" / "data" / "simulations"
EXPECTED = {"airline": 50, "retail": 114, "telecom": 114}
ANCHORS = {"deepseek_deepseek-v4-flash", "dashscope_qwen3.7-plus"}


def clean_and_count():
    """删除错误模拟,返回(移除数, 剩余缺口)。"""
    removed = missing = 0
    for d in sorted(SIMS.iterdir()):
        if d.name[0].isdigit():
            continue
        f = d / "results.json"
        if not f.exists():
            continue
        domain = d.name.split("_")[0]
        if domain not in EXPECTED:
            continue
        model = d.name[len(domain) + 1:]
        exp = EXPECTED[domain] * (3 if model in ANCHORS else 1)
        data = json.load(open(f))
        good = [s for s in data["simulations"] if s["termination_reason"] != "infrastructure_error"]
        if len(good) < len(data["simulations"]):
            removed += len(data["simulations"]) - len(good)
            data["simulations"] = good
            json.dump(data, open(f, "w"), ensure_ascii=False)
        missing += max(0, exp - len(good))
    return removed, missing


for rnd, conc in [(1, 4), (2, 3), (3, 2), (4, 2), (5, 2)]:
    removed, missing = clean_and_count()
    print(f"[round {rnd}] 清理{removed}条错误, 缺口{missing}条", flush=True)
    if missing == 0:
        print("数据已完整,自愈完成。", flush=True)
        sys.exit(0)
    subprocess.run(
        [sys.executable, str(EXP / "run_matrix.py"), "--max-concurrency", str(conc)],
        cwd=EXP, stdout=open(EXP / "logs" / f"heal_round{rnd}.log", "w"),
        stderr=subprocess.STDOUT,
    )

removed, missing = clean_and_count()
print(f"[final] 缺口{missing}条", flush=True)
sys.exit(0 if missing == 0 else 3)
