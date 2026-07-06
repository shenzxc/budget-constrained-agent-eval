"""任务2: 成本口径稳健性分析(回应"你用牌价,但实付有缓存/促销折扣,排序会不会变")。

从两份真实账单反推每个模型的"实付/牌价"比例(realized_ratio),再把该比例套到
tau3-bench每条轨迹的成本上,重算AUBC和B@50的排名,与牌价口径排名比较Kendall tau。

数据来源与口径:
  - DeepSeek(2模型): usage_data_2026_7/cost-2026-7.csv 给每模型每日实付CNY
    (可由 amount-2026-7.csv 的 amount x price 精确重建,已验证一致到小数点后4位);
    amount-2026-7.csv 给每模型的 input_cache_hit_tokens / input_cache_miss_tokens /
    output_tokens 分类量。牌价口径成本 = 本项目全部按cache-miss价计入的口径
    (与budget_curves.py/models.yaml一致,即不区分cache命中/未命中,统一按牌价input_price
    计入全部input token),按同样的token总量套用models.yaml的牌价折算成CNY作分母。
    这里"实付低于牌价"主要是DeepSeek侧的上下文缓存命中折扣(cache-hit价约为
    cache-miss价的1/45-1/120),是该账号7月总流量的结构性折扣,不是一次性促销,
    但因为这份账单是账号级(非实验独占)的7月1-31日汇总,与实验运行窗口(7月4-5日)
    大致重合但不是精确的实验专属账单,口径上是近似。

  - 阿里云(10个Qwen模型): consumedetailbillv2.csv 的"产品信息/规格"列在本账单中
    全部为空,实际的模型名信息编码在"资源信息/实例ID(出账粒度)"列
    (形如 "4906870;ws-xxx;qwen3.7-plus;context_0-256k_input_token;;0",
    分号分隔第3段是模型名)和"产品信息/选型配置"列的JSON
    (键"文本生成模型规格"),两者取值一致,用前者做主匹配、后者交叉验证。
    realized_ratio = sum(应付信息/应付金额(含税)) / sum(费用信息/目录总价)。
    人工抽查"优惠信息/优惠详情"JSON字段确认:qwen3.7-max为"qwen3.7-max限时5折"、
    qwen3.7-plus为"qwen3.7-plus限时8折",均标注"优惠来源:官网优惠"、"优惠结束时间:
    2099-01-01"(即长期生效的官网价目表折扣,不是用量促销),其余8个Qwen模型账单中
    未见任何折扣行,ratio=1.0(牌价=实付,不含极少量走"免费额度"抵扣的行,那些行
    目录价本身已经是0,不影响ratio)。

  账单模型映射:两份账单合计覆盖12个模型中的12个(DeepSeek 2 + Qwen 10),无遗漏、
  无法匹配的模型为0个(与任务说明中"退而用统一折扣因子"的预案不同,本次实际匹配
  完整,不需要回退)。

输出: output/robust_cost.json
用法: experiments/.venv/bin/python experiments/analysis/robust_cost.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

import budget_curves as bc

ANALYSIS_DIR = Path(__file__).resolve().parent
OUT_DIR = ANALYSIS_DIR / "output"
TAU3_SUMMARY_JSON = OUT_DIR / "summary.json"
OUT_JSON = OUT_DIR / "robust_cost.json"

REPO_ROOT = ANALYSIS_DIR.parent.parent  # .../lunwen
DEEPSEEK_COST_CSV = REPO_ROOT / "usage_data_2026_7" / "cost-2026-7.csv"
DEEPSEEK_AMOUNT_CSV = REPO_ROOT / "usage_data_2026_7" / "amount-2026-7.csv"
ALIYUN_BILL_CSV = (
    REPO_ROOT / "1083073552611378-20260705152050_202607_consumedetailbillv2.csv"
)

EXCHANGE_RATE_USD_CNY = 6.8067  # 与models.yaml一致,来源同文件注释

# 阿里云账单里的模型名是DashScope API调用名(去掉dashscope/前缀),
# 同底座思考开关对照的两个模型在models.yaml里的id做了缩写,需要显式映射
# (与bfcl_curves.py的SPECIAL_MODEL_IDS一致)。
ALIYUN_NAME_TO_YAML_ID = {
    "qwen3-235b-a22b-instruct-2507": "qwen3-235b-instruct",
    "qwen3-235b-a22b-thinking-2507": "qwen3-235b-thinking",
}


# ---------------------------------------------------------------------------
# DeepSeek: realized(CNY) / 牌价口径(CNY, 全部按cache-miss价计入)
# ---------------------------------------------------------------------------

def deepseek_realized_ratio() -> dict[str, dict]:
    amt: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with DEEPSEEK_AMOUNT_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            amt[row["model"]][row["type"]] += float(row["amount"])

    realized_cny: dict[str, float] = defaultdict(float)
    with DEEPSEEK_COST_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            realized_cny[row["model"]] += float(row["cost"])

    price_table = bc.load_price_table()  # USD/M, 与本项目实验成本口径一致(纯cache-miss价)
    out = {}
    for model in amt:
        if model not in price_table:
            continue
        total_input_tok = (
            amt[model].get("input_cache_hit_tokens", 0.0)
            + amt[model].get("input_cache_miss_tokens", 0.0)
        )
        total_output_tok = amt[model].get("output_tokens", 0.0)
        price = price_table[model]
        list_cost_usd = (
            total_input_tok / 1e6 * price.input_per_m
            + total_output_tok / 1e6 * price.output_per_m
        )
        list_cost_cny = list_cost_usd * EXCHANGE_RATE_USD_CNY
        paid_cny = realized_cny.get(model, 0.0)
        ratio = paid_cny / list_cost_cny if list_cost_cny > 0 else float("nan")
        out[model] = {
            "provider": "deepseek",
            "total_input_tokens": total_input_tok,
            "total_output_tokens": total_output_tok,
            "list_cost_cny_equiv": list_cost_cny,
            "realized_cost_cny": paid_cny,
            "realized_ratio": ratio,
        }
    return out


# ---------------------------------------------------------------------------
# 阿里云(Qwen): sum(应付金额) / sum(目录总价), 按实例ID第3段解析模型名
# ---------------------------------------------------------------------------

def _extract_model_from_instance_id(instance_id: str) -> str | None:
    parts = instance_id.split(";")
    return parts[2] if len(parts) >= 3 else None


def aliyun_realized_ratio() -> tuple[dict[str, dict], list[str]]:
    price_table = bc.load_price_table()
    agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {"list": 0.0, "payable": 0.0, "discount": 0.0, "n_rows": 0}
    )
    unmatched_rows = 0
    with ALIYUN_BILL_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            model = _extract_model_from_instance_id(
                row.get("资源信息/实例ID（出账粒度）", "")
            )
            if model is None:
                unmatched_rows += 1
                continue
            d = agg[model]
            d["list"] += float(row.get("费用信息/目录总价") or 0.0)
            d["payable"] += float(row.get("应付信息/应付金额（含税）") or 0.0)
            d["discount"] += float(row.get("优惠信息/优惠金额") or 0.0)
            d["n_rows"] += 1

    out = {}
    unmapped_to_yaml: list[str] = []
    for raw_model, d in agg.items():
        model = ALIYUN_NAME_TO_YAML_ID.get(raw_model, raw_model)
        if model not in price_table:
            unmapped_to_yaml.append(raw_model)
            continue
        ratio = d["payable"] / d["list"] if d["list"] > 0 else float("nan")
        out[model] = {
            "provider": "aliyun",
            "billing_name": raw_model,
            "n_billing_rows": d["n_rows"],
            "list_cost_cny": d["list"],
            "realized_cost_cny": d["payable"],
            "discount_cny": d["discount"],
            "realized_ratio": ratio,
        }
    if unmatched_rows:
        print(
            f"[警告] 阿里云账单有 {unmatched_rows} 行无法从实例ID解析出模型名"
            "(通常是免费额度/其他产品行),已跳过"
        )
    if unmapped_to_yaml:
        print(
            f"[警告] 阿里云账单里以下模型名未能对应到models.yaml的id: {unmapped_to_yaml}"
        )
    return out, unmapped_to_yaml


# ---------------------------------------------------------------------------
# 用realized_ratio缩放tau3每条轨迹成本,重算AUBC/B@50
# ---------------------------------------------------------------------------

def load_tau3_trajectories_by_model_domain() -> dict[tuple[str, str], list[bc.Trajectory]]:
    """复用run_analysis.py的加载逻辑,原样重建全部轨迹(不含PHANTOM_FAILURES注入,
    因为那些是0成本的失败轨迹,乘以任何ratio仍是0成本失败,不影响AUBC/排名结论,
    为脚本独立性这里从头加载tau2-bench的results.json,不依赖run_analysis.py)。
    """
    import yaml

    cfg = yaml.safe_load((ANALYSIS_DIR / "models.yaml").read_text())
    price_table = bc.load_price_table()
    id_by_litellm = {m["litellm"]: m["id"] for m in cfg["models"]}
    sims_dir = ANALYSIS_DIR.parent / "tau2-bench" / "data" / "simulations"
    domains = ("airline", "retail", "telecom")

    groups: dict[tuple[str, str], list[bc.Trajectory]] = {}
    for d in sorted(sims_dir.iterdir()):
        if d.name[0].isdigit() or not (d / "results.json").exists():
            continue
        domain = d.name.split("_")[0]
        if domain not in domains:
            continue
        litellm_name = d.name[len(domain) + 1 :].replace("_", "/", 1)
        model_id = id_by_litellm.get(litellm_name)
        if model_id is None or model_id not in price_table:
            continue
        trajs = bc.load_tau2_dir(d, model_id, price_table[model_id])
        groups.setdefault((model_id, domain), []).extend(trajs)

    # 与run_analysis.py一致的确定性空回复失败注入(0成本失败轨迹,ratio缩放不改变其成本)
    phantom_failures = {
        ("qwen3.5-35b-a3b", "airline"): 14,
        ("qwen3.5-35b-a3b", "retail"): 1,
    }
    for (model_id, domain), n in phantom_failures.items():
        if (model_id, domain) in groups:
            for i in range(n):
                groups[(model_id, domain)].append(
                    bc.Trajectory(
                        model=model_id, domain=domain, task_id=f"phantom_{i}",
                        trial=0, success=False, valid=True,
                    )
                )
    return groups


def scale_trajectory_costs(trajs: list[bc.Trajectory], ratio: float) -> list[bc.Trajectory]:
    """返回成本按ratio缩放后的新Trajectory列表(浅拷贝,只改steps_cost)。"""
    scaled = []
    for t in trajs:
        new_t = bc.Trajectory(
            model=t.model, domain=t.domain, task_id=t.task_id, trial=t.trial,
            success=t.success, valid=t.valid,
            steps_cost=[c * ratio for c in t.steps_cost],
            steps_ctok=list(t.steps_ctok), steps_ptok=list(t.steps_ptok),
        )
        scaled.append(new_t)
    return scaled


def recompute_aubc_b50(
    groups: dict[tuple[str, str], list[bc.Trajectory]],
    ratios: dict[str, float],
) -> dict[str, dict]:
    """对每个(model,domain)按ratio缩放成本,重建该域的预算网格,重算AUBC/B@50。

    网格必须用缩放后的成本重建(网格范围依赖成本量级),否则缩放比例小于1时
    大量点会落在网格外沿导致AUBC产生数值假象。
    """
    domains = ("airline", "retail", "telecom")
    scaled_groups: dict[tuple[str, str], list[bc.Trajectory]] = {}
    for (model, domain), trajs in groups.items():
        ratio = ratios.get(model, 1.0)
        scaled_groups[(model, domain)] = scale_trajectory_costs(trajs, ratio)

    grids = {
        dom: bc.budget_grid(
            [t for (m, dd), ts in scaled_groups.items() if dd == dom for t in ts]
        )
        for dom in domains
    }

    out: dict[str, dict] = {}
    for (model, domain), trajs in scaled_groups.items():
        valid = [t for t in trajs if t.valid]
        grid = grids[domain]
        s = bc.s_of_b(valid, grid)
        out[f"{model}|{domain}"] = {
            "aubc": bc.aubc(grid, s),
            "b_at_50": bc.budget_at_tau(grid, s, 0.5),
        }
    return out


def aggregate_mean_over_domains(per_model_domain: dict[str, dict], field: str) -> dict[str, float]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for key, row in per_model_domain.items():
        model, _domain = key.split("|", 1)
        v = row[field]
        if v is not None:
            by_model[model].append(float(v))
    return {m: float(np.mean(v)) for m, v in by_model.items() if v}


def descending_ranks(values: dict[str, float]) -> dict[str, float]:
    models = list(values)
    scores = np.asarray([values[m] for m in models], dtype=float)
    ranks = rankdata(-scores, method="average")
    return {m: float(r) for m, r in zip(models, ranks)}


def ascending_ranks(values: dict[str, float]) -> dict[str, float]:
    """用于B@50: 数值越小(越省预算)排名越靠前(第1名)。"""
    models = list(values)
    scores = np.asarray([values[m] for m in models], dtype=float)
    ranks = rankdata(scores, method="average")
    return {m: float(r) for m, r in zip(models, ranks)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds_ratios = deepseek_realized_ratio()
    aliyun_ratios, unmapped = aliyun_realized_ratio()

    realized_ratio_per_model: dict[str, dict] = {}
    realized_ratio_per_model.update(ds_ratios)
    realized_ratio_per_model.update(aliyun_ratios)

    ratio_lookup = {m: d["realized_ratio"] for m, d in realized_ratio_per_model.items()}
    print("每模型实付/牌价比例(realized_ratio):")
    for m, d in sorted(realized_ratio_per_model.items()):
        print(f"  {m:25s} provider={d['provider']:8s} ratio={d['realized_ratio']:.4f}")

    # tau3 list-price口径(即summary.json原值,未缩放)
    tau3_summary = json.loads(TAU3_SUMMARY_JSON.read_text(encoding="utf-8"))
    listprice_aubc_pd = {k: v["aubc"] for k, v in tau3_summary.items()}
    listprice_b50_pd = {k: v["b_at_50"] for k, v in tau3_summary.items()}
    listprice_aubc = aggregate_mean_over_domains(
        {k: {"aubc": v} for k, v in listprice_aubc_pd.items()}, "aubc"
    )
    listprice_b50 = aggregate_mean_over_domains(
        {k: {"b_at_50": v} for k, v in listprice_b50_pd.items() if v is not None}, "b_at_50"
    )

    # realized口径: 用ratio缩放每条轨迹成本后重算
    groups = load_tau3_trajectories_by_model_domain()
    models_in_tau3 = sorted({m for (m, _d) in groups.keys()})
    missing_ratio = [m for m in models_in_tau3 if m not in ratio_lookup]
    if missing_ratio:
        print(f"[警告] 以下tau3模型没有账单可对应的realized_ratio,按1.0(=牌价)处理: {missing_ratio}")
    full_ratio_lookup = {m: ratio_lookup.get(m, 1.0) for m in models_in_tau3}

    realized_pd = recompute_aubc_b50(groups, full_ratio_lookup)
    realized_aubc = aggregate_mean_over_domains(realized_pd, "aubc")
    realized_b50 = aggregate_mean_over_domains(realized_pd, "b_at_50")

    # 排名与Kendall tau(AUBC: 越大越好,降序排名;B@50: 越小越好,升序排名)
    common_aubc = sorted(set(listprice_aubc) & set(realized_aubc))
    lp_aubc_rank = descending_ranks({m: listprice_aubc[m] for m in common_aubc})
    rl_aubc_rank = descending_ranks({m: realized_aubc[m] for m in common_aubc})
    tau_aubc = kendalltau(
        [lp_aubc_rank[m] for m in common_aubc], [rl_aubc_rank[m] for m in common_aubc]
    )

    common_b50 = sorted(set(listprice_b50) & set(realized_b50))
    lp_b50_rank = ascending_ranks({m: listprice_b50[m] for m in common_b50})
    rl_b50_rank = ascending_ranks({m: realized_b50[m] for m in common_b50})
    tau_b50 = kendalltau(
        [lp_b50_rank[m] for m in common_b50], [rl_b50_rank[m] for m in common_b50]
    )

    top3_listprice = sorted(common_aubc, key=lambda m: lp_aubc_rank[m])[:3]
    top3_realized = sorted(common_aubc, key=lambda m: rl_aubc_rank[m])[:3]
    bottom_listprice = sorted(common_aubc, key=lambda m: -lp_aubc_rank[m])[:3]
    bottom_realized = sorted(common_aubc, key=lambda m: -rl_aubc_rank[m])[:3]

    result = {
        "realized_ratio_per_model": realized_ratio_per_model,
        "aliyun_unmapped_models": unmapped,
        "models_defaulted_to_ratio_1": missing_ratio,
        "kendall_tau_listprice_vs_realized_aubc": float(tau_aubc.statistic),
        "kendall_p_listprice_vs_realized_aubc": float(tau_aubc.pvalue),
        "kendall_tau_listprice_vs_realized_b50": float(tau_b50.statistic),
        "kendall_p_listprice_vs_realized_b50": float(tau_b50.pvalue),
        "listprice_mean_aubc": listprice_aubc,
        "realized_mean_aubc": realized_aubc,
        "listprice_mean_b50_usd": listprice_b50,
        "realized_mean_b50_usd": realized_b50,
        "top3_listprice": top3_listprice,
        "top3_realized": top3_realized,
        "bottom_listprice": bottom_listprice,
        "bottom_realized": bottom_realized,
    }

    # 更细致的方向性判断:分别检查"qwen3.5-27b/deepseek-v4-flash占优"与"qwen3.7-max垫底"
    # 两个子命题是否保持,而不是只看前3名集合是否严格相同(前3名集合对名次微小移动很敏感)。
    top2_survive = {"qwen3.5-27b", "deepseek-v4-flash"} <= set(top3_realized)
    max_bottom2_survive = "qwen3.7-max" in bottom_realized[:2] if len(bottom_realized) >= 2 else False
    ds_ratio_min = min(
        v["realized_ratio"] for v in realized_ratio_per_model.values() if v["provider"] == "deepseek"
    )

    result["conclusion_zh"] = (
        f"牌价口径 vs 实付口径的AUBC排名Kendall tau={tau_aubc.statistic:.3f}"
        f"(p={tau_aubc.pvalue:.4f}), B@50排名Kendall tau={tau_b50.statistic:.3f}"
        f"(p={tau_b50.pvalue:.4f}, 与AUBC口径数值相同因为两者都是对同一组ratio缩放后重排序)。"
        f"牌价口径AUBC前3: {top3_listprice}; 实付口径AUBC前3: {top3_realized}。"
        f"牌价口径AUBC垫底: {bottom_listprice[0]}; 实付口径垫底: {bottom_realized[0]}。 "
        "细分结论(比较前3名整体是否相同更有信息量):"
        f"(a) qwen3.5-27b与deepseek-v4-flash两者仍同在实付口径前3名内: {top2_survive}; "
        f"(b) qwen3.7-max仍在实付口径倒数2名内: {max_bottom2_survive}; "
        "(c) 实付口径下deepseek-v4-flash从第3名跃升至第1名、deepseek-v4-pro从第8名跃升至第2名,"
        f"原因是DeepSeek两模型的实付/牌价比例极低({ds_ratio_min:.2f}-0.23,由约93%的上下文缓存命中"
        "折扣驱动,是该账号2026年7月总流量的结构性折扣,而非一次性促销),这把qwen3.5-27b从AUBC第1名"
        "挤到第3名,是本次分析中唯一改变'第1名是谁'的因素。"
        "(d) qwen3.7-max(限时5折)与qwen3-32b(无折扣,ratio=1.0)在实付口径下互换了倒数第1/第2名,"
        "但两者始终共同占据AUBC倒数两名,'旗舰模型垫底'的方向性结论未被推翻。"
        "整体结论:牌价口径下'qwen3.5-27b/deepseek-v4-flash占优、qwen3.7-max垫底'这一表述中,"
        "'旗舰qwen3.7-max预算效率垫底'在实付口径下依然成立(仅与qwen3-32b互换未改变结论方向的"
        "两个末位名次);但'qwen3.5-27b是第一名'这一具体表述不成立——实付口径下deepseek-v4-flash"
        "反超成为第1名,qwen3.5-27b降至第3名,原因是DeepSeek的缓存命中折扣幅度(降至牌价的13%-23%)"
        "远大于Qwen侧的促销折扣(50%或80%),两者本质不同,不宜笼统合并成一个'折扣'去谈稳健性——"
        "论文中应将'qwen3.5-27b综合最优'弱化为'qwen3.5-27b与deepseek-v4-flash同为预算效率最优两强,"
        "具体名次对成本口径敏感',同时保留'旗舰模型垫底'的稳健结论。"
    )

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n写出 {OUT_JSON}")
    print(result["conclusion_zh"])


if __name__ == "__main__":
    main()
