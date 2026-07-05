"""预算受控智能体评测:核心分析模块。

把τ³-bench轨迹变成预算条件化指标:
  S(B)      预算-成功率曲线(离线截断:budget-unaware智能体在预算B下的结果
            = 最大预算轨迹截断到累计成本首次超过B之前;若轨迹自然结束且
            成功且总成本<=B,记为成功,否则失败)
  AUBC      log预算轴上的归一化曲线下面积
  B@tau     达到成功率tau所需的最小预算
  弹性       dlogS/dlogB(有限差分)
  反转点     两模型S(B)曲线的交叉预算区间

成本口径:仅agent侧(assistant消息的usage),token数 x models.yaml牌价,
用户模拟器与判分器成本单独记录、不计入预算。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_YAML = ANALYSIS_DIR / "models.yaml"


# ---------------------------------------------------------------------------
# 计价
# ---------------------------------------------------------------------------

@dataclass
class Price:
    input_per_m: float   # USD / 1M prompt tokens
    output_per_m: float  # USD / 1M completion tokens


def load_price_table(models_yaml: Path = DEFAULT_MODELS_YAML) -> dict[str, Price]:
    cfg = yaml.safe_load(models_yaml.read_text())
    rate = float(cfg["exchange_rate_usd_cny"])
    table: dict[str, Price] = {}
    for m in cfg["models"]:
        if m["input_price"] is None or m["output_price"] is None:
            continue  # 价格未填的模型跳过,分析时会显式报缺
        k = 1.0 if m["currency"] == "USD" else 1.0 / rate
        table[m["id"]] = Price(m["input_price"] * k, m["output_price"] * k)
        table[m["litellm"]] = table[m["id"]]  # litellm名也可查
    return table


# ---------------------------------------------------------------------------
# 轨迹加载(τ³-bench results.json)
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    model: str
    domain: str
    task_id: str
    trial: int
    success: bool                 # reward == 1.0 且正常结束
    valid: bool                   # 非infrastructure_error
    steps_cost: list[float] = field(default_factory=list)   # 每次agent调用的增量成本(USD)
    steps_ctok: list[int] = field(default_factory=list)     # 每次agent调用的completion tokens
    steps_ptok: list[int] = field(default_factory=list)     # 每次agent调用的prompt tokens

    @property
    def total_cost(self) -> float:
        return float(sum(self.steps_cost))

    @property
    def total_ctok(self) -> int:
        return int(sum(self.steps_ctok))

    def cost_at_completion(self) -> float:
        """轨迹自然完成所需的agent总成本。"""
        return self.total_cost


def load_tau2_dir(sim_dir: Path, model_id: str, price: Price) -> list[Trajectory]:
    """解析一个tau2模拟目录(需含results.json)。"""
    r = json.loads((sim_dir / "results.json").read_text())
    domain = r["info"]["environment_info"]["domain_name"] if "info" in r else "unknown"
    out = []
    for s in r["simulations"]:
        valid = s["termination_reason"] != "infrastructure_error"
        reward = (s.get("reward_info") or {}).get("reward", 0.0)
        t = Trajectory(
            model=model_id,
            domain=domain,
            task_id=str(s["task_id"]),
            trial=int(s.get("trial", 0)),
            success=bool(valid and reward is not None and reward >= 1.0),
            valid=valid,
        )
        for m in s["messages"]:
            if m.get("role") == "assistant" and m.get("usage"):
                p = int(m["usage"].get("prompt_tokens") or 0)
                c = int(m["usage"].get("completion_tokens") or 0)
                t.steps_ptok.append(p)
                t.steps_ctok.append(c)
                t.steps_cost.append(
                    p / 1e6 * price.input_per_m + c / 1e6 * price.output_per_m
                )
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# S(B) 与派生指标
# ---------------------------------------------------------------------------

def budget_grid(trajs: list[Trajectory], n: int = 200) -> np.ndarray:
    """全体轨迹总成本范围上的log等距预算网格。"""
    costs = [t.total_cost for t in trajs if t.valid and t.total_cost > 0]
    lo, hi = min(costs), max(costs)
    return np.logspace(math.log10(lo * 0.5), math.log10(hi * 1.05), n)


def s_of_b(trajs: list[Trajectory], grid: np.ndarray) -> np.ndarray:
    """S(B):截断语义下预算B的成功率。仅计valid轨迹。"""
    ts = [t for t in trajs if t.valid]
    if not ts:
        return np.full_like(grid, np.nan)
    costs = np.array([t.total_cost for t in ts])
    succ = np.array([t.success for t in ts])
    # 成功条件:轨迹成功 且 完成成本 <= B
    return np.array([(succ & (costs <= b)).mean() for b in grid])


def aubc(grid: np.ndarray, s: np.ndarray) -> float:
    """log预算轴上归一化的曲线下面积,范围[0,1]。"""
    x = np.log10(grid)
    return float(np.trapz(s, x) / (x[-1] - x[0]))


def budget_at_tau(grid: np.ndarray, s: np.ndarray, tau: float) -> float | None:
    """B@tau:最小预算使S(B)>=tau;达不到返回None。"""
    idx = np.nonzero(s >= tau)[0]
    return float(grid[idx[0]]) if len(idx) else None


def elasticity(grid: np.ndarray, s: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """dlogS/dlogB(中心差分,S平滑后)。"""
    logs = np.log(np.maximum(s, eps))
    logb = np.log(grid)
    return np.gradient(logs, logb)


def crossovers(
    grid: np.ndarray, s1: np.ndarray, s2: np.ndarray, min_gap: float = 0.02
) -> list[float]:
    """两条S(B)曲线的交叉预算点(差值符号翻转且翻转前后差距超过min_gap)。"""
    d = s1 - s2
    pts = []
    for i in range(1, len(grid)):
        if d[i - 1] * d[i] < 0 and (abs(d[i - 1]) > min_gap or abs(d[i]) > min_gap):
            # 线性插值求交点
            w = abs(d[i - 1]) / (abs(d[i - 1]) + abs(d[i]))
            pts.append(float(grid[i - 1] ** (1 - w) * grid[i] ** w))
    return pts


# ---------------------------------------------------------------------------
# 任务级配对bootstrap(同域同任务集的模型间比较)
# ---------------------------------------------------------------------------

def bootstrap_aubc_ci(
    trajs: list[Trajectory], grid: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """AUBC的95%bootstrap置信区间(按任务重采样)。"""
    rng = np.random.default_rng(seed)
    ts = [t for t in trajs if t.valid]
    vals = []
    for _ in range(n_boot):
        sample = [ts[i] for i in rng.integers(0, len(ts), len(ts))]
        vals.append(aubc(grid, s_of_b(sample, grid)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(trajs: list[Trajectory], grid: np.ndarray | None = None) -> dict:
    """单(模型,域)组合的指标汇总。"""
    ts = [t for t in trajs if t.valid]
    if grid is None:
        grid = budget_grid(ts)
    s = s_of_b(ts, grid)
    lo, hi = bootstrap_aubc_ci(ts, grid)
    completed = [t for t in ts if t.steps_cost]  # 排除零成本幻影失败,只算真实轨迹的均值成本
    return {
        "n_valid": len(ts),
        "n_invalid": sum(1 for t in trajs if not t.valid),
        "pass1": float(np.mean([t.success for t in ts])) if ts else float("nan"),
        "mean_cost": float(np.mean([t.total_cost for t in completed])) if completed else float("nan"),
        "aubc": aubc(grid, s),
        "aubc_ci95": [lo, hi],
        "b_at_50": budget_at_tau(grid, s, 0.5),
        "b_at_80": budget_at_tau(grid, s, 0.8),
        "grid": grid.tolist(),
        "s_of_b": s.tolist(),
    }
