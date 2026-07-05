#!/usr/bin/env python3
"""
run_matrix.py -- 全量τ²-bench评测调度脚本(预算受控LLM智能体评测项目)

用途:
  读取 analysis/models.yaml 中的模型矩阵,对每个模型 × 每个域(airline/retail/telecom)
  组织并串行执行 `uv run tau2 run ...` 命令(tau2-bench自带独立uv环境,与本脚本运行的
  Python解释器无关,本脚本只用标准库 + PyYAML 做纯subprocess调度)。

各域任务数(已用 tau2 CLI 源码 + data/tau2/domains/*/split_tasks.json 核实清楚,2026-07-04):
  - airline: 50 个任务(tasks.json 全量50个 == "base" split,--domain airline 默认即用此split)
  - retail:  114 个任务(tasks.json 全量114个 == "base" split)
  - telecom: 114 个任务  <-- 注意:不是2285!
      telecom域的任务注册比较特殊:
        - registry.py 里 "--domain telecom" 实际绑定的是 telecom_domain_get_tasks(不带
          _full/_small后缀的版本),其默认 task_split_name="base"。
        - data/tau2/domains/telecom/split_tasks.json 里 "base" split 有114个任务ID。
        - tasks_full.json 有2285个任务、tasks_small.json 有20个,但这两个只有在显式传
          --task-set-name telecom_full 或 telecom_small 时才会被使用,--domain telecom
          默认并不会跑2285个任务的"full"集合。
      本脚本按 --domain telecom 的默认行为组织命令,即 114 个任务,不额外传
      --task-set-name/--task-split-name。若未来需要跑 telecom_full(2285任务),
      需要新增命令行开关,此脚本目前不支持(全量成本预估也是按114算的,见
      experiments/全量实验预估.md)。

anchor_models(models.yaml里的 benchmarks.tau2.anchor_models: deepseek-v4-flash, qwen3.7-max)
跑 --num-trials 3(用于估计跑次间方差),其余模型跑 --num-trials 1。

断点续跑机制(两层):
  1. tau2 CLI 自带 --auto-resume:调研结论——它只能在"同一个 --save-to 路径"下生效
     (对比 src/tau2/runner/checkpoint.py 的 try_resume 实现,resume是通过读取
     data/simulations/<save_to>/results.json 已完成的 (trial, task_id, seed) 集合来跳过已
     完成的任务,继续把剩余任务补齐)。如果不传 --save-to,tau2 每次都会生成新的带时间戳的
     目录名,--auto-resume 就没有目标可续。
     因此本脚本给每个(域,模型)组合构造一个**去时间戳的固定 save_to 名称**
     (形如 "<domain>_<model_sanitized>",模型名里的 "/" 替换成 "_"),并总是带上
     --auto-resume,这样:
       - 若该目录不存在:创建新的,正常跑。
       - 若该目录存在但未跑完(比如脚本上次被中断):tau2会自动续跑剩余任务,不会重新问
         y/n,也不会重复计费已完成的任务。
       - 若该目录已跑完(见下面第2层判断):脚本会直接跳过,连tau2进程都不启动。
  2. 脚本自身在下发每条命令前,先检查 data/simulations/<save_to>/results.json 是否存在且
     "完整"——完整的定义是 len(simulations) >= len(tasks) * num_trials(即该域任务数 ×
     该模型的trial数)。用 info.agent_info.llm 和 info.environment_info.domain_name 字段
     核对文件内容确实对应目标(域,模型),而不是单纯依赖目录名字符串,更稳健。
     完整则跳过、不调用tau2;不完整(部分完成或全新)则调用tau2 + --auto-resume。

用法:
  python3 run_matrix.py --dry-run          # 只打印将要执行的命令清单,不运行任何东西
  python3 run_matrix.py                    # 实际串行执行全部命令(会产生真实API调用!!)
  python3 run_matrix.py --dry-run --domain airline   # 只看某个域
  python3 run_matrix.py --dry-run --model deepseek-v4-flash  # 只看某个模型

约束:
  - 一次只跑一个tau2进程(脚本串行调度),每个tau2进程内部并发 --max-concurrency 8。
  - 失败(非0退出码)不中断整体调度,记录进CLI最后的汇总报错列表并继续下一条。
  - 每条命令的stdout/stderr都写入 experiments/logs/<模型>_<域>.log(追加模式,方便resume后
    查看历史)。
  - 本脚本本身不会被本次任务自动执行全量;仅在 --dry-run 下验证过。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print(
        "需要 PyYAML。请用系统python3或项目.venv运行本脚本(两者都已装yaml)。",
        file=sys.stderr,
    )
    raise

EXPERIMENTS_DIR = Path(__file__).resolve().parent
TAU2_DIR = EXPERIMENTS_DIR / "tau2-bench"
MODELS_YAML = EXPERIMENTS_DIR / "analysis" / "models.yaml"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
SIMULATIONS_DIR = TAU2_DIR / "data" / "simulations"

# 各域任务数(见文件头注释的核实过程),用于判断"是否已跑完"。
DOMAIN_TASK_COUNTS = {
    "airline": 50,
    "retail": 114,
    "telecom": 114,  # --domain telecom 默认 base split,不是tasks_full.json的2285
}

DEFAULT_MAX_CONCURRENCY = 8


def sanitize_model_name(litellm_name: str) -> str:
    """把litellm模型名(如 'dashscope/qwen3.7-max')变成文件系统/save-to安全的字符串。"""
    return litellm_name.replace("/", "_")


@dataclass
class RunSpec:
    domain: str
    model_id: str  # models.yaml里的id,比如 "deepseek-v4-flash"
    litellm_name: str  # 实际传给--agent-llm的名字,比如 "deepseek/deepseek-v4-flash"
    num_trials: int
    save_to: str  # data/simulations/<save_to>/
    extra_agent_llm_args: Optional[dict] = None

    @property
    def log_path(self) -> Path:
        return LOGS_DIR / f"{self.model_id}_{self.domain}.log"

    def build_command(self, user_llm: str, max_concurrency: int) -> list[str]:
        agent_llm_args = {"temperature": 0.0}
        if self.extra_agent_llm_args:
            agent_llm_args.update(self.extra_agent_llm_args)
        cmd = [
            "uv",
            "run",
            "tau2",
            "run",
            "--domain",
            self.domain,
            "--agent-llm",
            self.litellm_name,
            "--agent-llm-args",
            json.dumps(agent_llm_args),
            "--user-llm",
            user_llm,
            "--num-trials",
            str(self.num_trials),
            "--max-concurrency",
            str(max_concurrency),
            "--save-to",
            self.save_to,
            "--auto-resume",
        ]
        return cmd


# 已知需要在--agent-llm-args里显式关闭思考模式的模型,否则DashScope会报:
#   "parameter.enable_thinking must be set to false for non-streaming calls"
# 这是任务2逐模型验证时发现的真实报错(qwen3-32b复现过),原因是该模型默认按"混合思考"模式
# 响应,但tau2/litellm默认走非流式调用,DashScope的非流式接口不允许开着思考模式。
# qwen3.7-max/qwen3.7-plus/qwen3.6-flash/qwen3.5系列在验证时未触发此问题(可能因为这些模型的
# hosted接口把enable_thinking的默认值处理得不同,或版本差异),qwen3-235b-a22b-thinking-2507
# 是"仅思考模式"模型,不接受enable_thinking参数,也未触发。为稳妥起见,只对已确认会报错的
# qwen3-32b加此参数;若全量跑批时其他模型也报同样的错,照此模式加 "enable_thinking": false
# 到对应模型的 extra_agent_llm_args 即可。
MODELS_NEEDING_NO_THINKING = {
    "qwen3-32b": {"enable_thinking": False},
}


def load_config() -> dict:
    with open(MODELS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_all_specs(cfg: dict) -> list[RunSpec]:
    domains = cfg["benchmarks"]["tau2"]["domains"]
    anchor_models = set(cfg["benchmarks"]["tau2"]["anchor_models"])
    anchor_trials = cfg["benchmarks"]["tau2"]["anchor_trials"]
    default_trials = cfg["benchmarks"]["tau2"]["num_trials"]

    specs = []
    for model in cfg["models"]:
        model_id = model["id"]
        litellm_name = model["litellm"]
        num_trials = anchor_trials if model_id in anchor_models else default_trials
        for domain in domains:
            save_to = f"{domain}_{sanitize_model_name(litellm_name)}"
            specs.append(
                RunSpec(
                    domain=domain,
                    model_id=model_id,
                    litellm_name=litellm_name,
                    num_trials=num_trials,
                    save_to=save_to,
                    extra_agent_llm_args=MODELS_NEEDING_NO_THINKING.get(model_id),
                )
            )
    return specs


def is_run_complete(spec: RunSpec) -> tuple[bool, str]:
    """检查 data/simulations/<save_to>/results.json 是否存在且已完整覆盖该(域,模型)的全部任务。

    完整 = len(simulations) >= 该域任务数 × spec.num_trials,且文件内容的
    agent_info.llm / environment_info.domain_name 与目标一致(防止save_to名称冲突导致误判)。

    返回 (是否完整, 原因说明)。
    """
    results_path = SIMULATIONS_DIR / spec.save_to / "results.json"
    if not results_path.exists():
        return False, "结果文件不存在"

    try:
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"结果文件损坏或无法读取: {e}"

    try:
        actual_domain = data["info"]["environment_info"]["domain_name"]
        actual_llm = data["info"]["agent_info"]["llm"]
        num_sims = len(data.get("simulations", []))
    except (KeyError, TypeError) as e:
        return False, f"结果文件结构异常: {e}"

    if actual_domain != spec.domain or actual_llm != spec.litellm_name:
        return False, (
            f"目录内容与预期不符(实际 domain={actual_domain}, llm={actual_llm}; "
            f"预期 domain={spec.domain}, llm={spec.litellm_name}),不跳过,让tau2自行处理"
        )

    expected = DOMAIN_TASK_COUNTS[spec.domain] * spec.num_trials
    if num_sims >= expected:
        return True, f"已完整: {num_sims}/{expected} simulations"
    else:
        return False, f"未跑完: {num_sims}/{expected} simulations,将用--auto-resume续跑"


def run_one(spec: RunSpec, user_llm: str, max_concurrency: int, dry_run: bool) -> dict:
    cmd = spec.build_command(user_llm=user_llm, max_concurrency=max_concurrency)
    cmd_str = " ".join(cmd)

    complete, reason = is_run_complete(spec)
    result = {
        "domain": spec.domain,
        "model_id": spec.model_id,
        "num_trials": spec.num_trials,
        "save_to": spec.save_to,
        "command": cmd_str,
        "skipped": False,
        "skip_reason": None,
        "returncode": None,
        "log_path": str(spec.log_path),
    }

    if complete:
        result["skipped"] = True
        result["skip_reason"] = reason
        print(f"[SKIP] {spec.domain}/{spec.model_id}: {reason}")
        return result

    print(f"[{'DRY-RUN' if dry_run else 'RUN'}] {spec.domain}/{spec.model_id} "
          f"(trials={spec.num_trials}, save_to={spec.save_to}) :: {reason}")
    print(f"    cwd={TAU2_DIR}")
    print(f"    cmd={cmd_str}")
    print(f"    log={spec.log_path}")

    if dry_run:
        return result

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec.log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n\n===== run_matrix.py invocation: {cmd_str} =====\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(TAU2_DIR),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    result["returncode"] = proc.returncode
    if proc.returncode != 0:
        print(f"    !! 失败,退出码={proc.returncode},详见 {spec.log_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令清单,不运行")
    parser.add_argument("--domain", choices=list(DOMAIN_TASK_COUNTS.keys()), default=None,
                        help="只跑/只打印某个域(默认全部三个域)")
    parser.add_argument("--model", default=None,
                        help="只跑/只打印某个模型的id(models.yaml里的id字段,默认全部模型)")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY,
                        help=f"tau2进程内部并发数,默认{DEFAULT_MAX_CONCURRENCY}")
    args = parser.parse_args()

    cfg = load_config()
    user_llm = cfg["user_simulator"]["litellm"]
    specs = build_all_specs(cfg)

    if args.domain:
        specs = [s for s in specs if s.domain == args.domain]
    if args.model:
        specs = [s for s in specs if s.model_id == args.model]

    if not specs:
        print("没有匹配的(域,模型)组合,请检查 --domain / --model 参数。", file=sys.stderr)
        sys.exit(1)

    print(f"共 {len(specs)} 条(域,模型)任务待处理。user_llm={user_llm}, "
          f"max_concurrency={args.max_concurrency}, dry_run={args.dry_run}")
    print(f"域任务数: {DOMAIN_TASK_COUNTS}")
    print("-" * 80)

    results = []
    failures = []
    for spec in specs:
        r = run_one(spec, user_llm=user_llm, max_concurrency=args.max_concurrency, dry_run=args.dry_run)
        results.append(r)
        if not args.dry_run and not r["skipped"] and r["returncode"] != 0:
            failures.append(r)

    print("-" * 80)
    n_skipped = sum(1 for r in results if r["skipped"])
    n_ran = len(results) - n_skipped
    print(f"汇总: 共{len(results)}条, 跳过{n_skipped}条(已完整), "
          f"{'将执行' if args.dry_run else '已执行'}{n_ran}条。")

    if not args.dry_run and failures:
        print(f"\n失败 {len(failures)} 条(已继续跑完其余任务,未中断整体调度):")
        for f in failures:
            print(f"  - {f['domain']}/{f['model_id']}: 退出码={f['returncode']}, 日志={f['log_path']}")
        sys.exit(2)


if __name__ == "__main__":
    main()
