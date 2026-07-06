"""图表生成脚本(预算受控智能体评测,SCI论文用图)。

数据源(运行时读取,不硬编码任何数字):
  analysis/output/summary.json   -- 键 "模型id|域",值含 S(B) 曲线与派生指标
  analysis/models.yaml           -- 模型元数据与官方牌价(含CNY->USD汇率折算)

产出(analysis/output/figures/):
  fig1_sb_curves.pdf/.png        全部模型的预算-成功率曲线,按域分3子图
  fig2_aubc_vs_price.pdf/.png    AUBC(三域平均) vs 输出牌价,含帕累托前沿
  fig3_rank_table.csv            不限预算排名 vs 三档预算下的排名
  fig3_kendall.txt               各预算档排名与pass1排名的Kendall tau
  fig4_price_vs_realized.pdf/.png 牌价综合单价排名 vs 实测成本排名
  fig4_note.txt                  价格反转模型对数量占比
  table1_main.csv                主结果表:模型x域 + 三域平均行

用法:
  .venv/bin/python analysis/make_figures.py
"""

from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import kendalltau

try:
    from adjustText import adjust_text
    HAVE_ADJUST_TEXT = True
except ImportError:
    HAVE_ADJUST_TEXT = False

ANALYSIS_DIR = Path(__file__).resolve().parent
OUT_DIR = ANALYSIS_DIR / "output"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_JSON = OUT_DIR / "summary.json"
MODELS_YAML = ANALYSIS_DIR / "models.yaml"
DOMAINS = ("airline", "retail", "telecom")

# ---------------------------------------------------------------------------
# 期刊风格
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # 嵌入TrueType,矢量文字可编辑/可搜索
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

# family -> 色系(colormap),同family内不同tier用色系内的不同深浅
FAMILY_CMAPS = {
    "deepseek": "Blues",
    "qwen-commercial": "Oranges",
    "qwen-open": "Greens",
}


# ---------------------------------------------------------------------------
# 数据加载(每次调用时读文件,不缓存硬编码数字)
# ---------------------------------------------------------------------------

def load_summary() -> dict:
    return json.loads(SUMMARY_JSON.read_text())


def load_models_cfg() -> dict:
    return yaml.safe_load(MODELS_YAML.read_text())


def load_price_table_usd() -> dict[str, dict]:
    """返回 {model_id: {input_price_usd, output_price_usd, family, tier, open_weight}}。"""
    cfg = load_models_cfg()
    rate = float(cfg["exchange_rate_usd_cny"])
    table = {}
    for m in cfg["models"]:
        if m.get("input_price") is None or m.get("output_price") is None:
            continue
        k = 1.0 if m["currency"] == "USD" else 1.0 / rate
        table[m["id"]] = {
            "input_price_usd": m["input_price"] * k,
            "output_price_usd": m["output_price"] * k,
            "family": m["family"],
            "tier": m.get("tier"),
            "open_weight": m.get("open_weight"),
        }
    return table


def model_ids_in_summary(summary: dict) -> list[str]:
    return sorted({k.split("|")[0] for k in summary})


def get_family_order(price_table: dict[str, dict], model_ids: list[str]) -> list[str]:
    """按family分组、组内按输出价从低到高排序,便于配色由浅到深单调。"""
    order = []
    families = sorted({price_table[m]["family"] for m in model_ids if m in price_table})
    # 固定期望顺序优先,若都存在则用之
    preferred = ["deepseek", "qwen-commercial", "qwen-open"]
    families = [f for f in preferred if f in families] + [f for f in families if f not in preferred]
    for fam in families:
        members = sorted(
            [m for m in model_ids if price_table.get(m, {}).get("family") == fam],
            key=lambda m: price_table[m]["output_price_usd"],
        )
        order.extend(members)
    return order


def build_color_map(price_table: dict[str, dict], model_ids: list[str]) -> dict[str, tuple]:
    """按family分色系,系内深浅按输出价排序分配。"""
    colors = {}
    families = sorted({price_table[m]["family"] for m in model_ids if m in price_table})
    for fam in families:
        members = sorted(
            [m for m in model_ids if price_table.get(m, {}).get("family") == fam],
            key=lambda m: price_table[m]["output_price_usd"],
        )
        cmap = plt.get_cmap(FAMILY_CMAPS.get(fam, "Greys"))
        n = len(members)
        # 避开colormap两端过浅/过深的部分
        shades = np.linspace(0.4, 0.9, n) if n > 1 else np.array([0.65])
        for m, s in zip(members, shades):
            colors[m] = cmap(s)
    # 未在price_table中的模型给灰色兜底
    for m in model_ids:
        colors.setdefault(m, (0.5, 0.5, 0.5, 1.0))
    return colors


# ---------------------------------------------------------------------------
# fig1: S(B) 曲线,1x3子图(按域)
# ---------------------------------------------------------------------------

def fig1_sb_curves():
    summary = load_summary()
    price_table = load_price_table_usd()
    model_ids = model_ids_in_summary(summary)
    order = get_family_order(price_table, model_ids)
    colors = build_color_map(price_table, model_ids)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.9), sharey=True)

    lines_for_legend = []
    labels_for_legend = []
    for ax, dom in zip(axes, DOMAINS):
        for m in order:
            key = f"{m}|{dom}"
            if key not in summary:
                continue
            s = summary[key]
            grid = np.asarray(s["grid"])
            sb = np.asarray(s["s_of_b"])
            (line,) = ax.plot(
                grid, sb, color=colors[m], linewidth=1.2, label=m, solid_capstyle="round"
            )
            if dom == DOMAINS[0]:
                lines_for_legend.append(line)
                labels_for_legend.append(m)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Budget B (USD/task)")
        ax.set_title(dom.capitalize())
        ax.grid(True, which="major", linewidth=0.3, alpha=0.4)

    axes[0].set_ylabel("Success rate S(B)")

    # 若第一子图缺某些模型(该域无数据),补齐图例项:按order遍历summary任意域取得颜色
    present = set(labels_for_legend)
    missing = [m for m in order if m not in present]
    for m in missing:
        line = plt.Line2D([0], [0], color=colors[m], linewidth=1.2, label=m)
        lines_for_legend.append(line)
        labels_for_legend.append(m)

    fig.legend(
        lines_for_legend,
        labels_for_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        title="Model",
        columnspacing=1.2,
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig1_sb_curves.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_sb_curves.png", bbox_inches="tight")
    plt.close(fig)
    return FIG_DIR / "fig1_sb_curves.pdf"


# ---------------------------------------------------------------------------
# fig2: AUBC vs 输出价(三域平均),含帕累托前沿
# ---------------------------------------------------------------------------

def fig2_aubc_vs_price():
    summary = load_summary()
    price_table = load_price_table_usd()
    model_ids = model_ids_in_summary(summary)
    colors = build_color_map(price_table, model_ids)

    pts = []  # (model_id, output_price, aubc_mean, ci_halfwidth_mean)
    for m in model_ids:
        if m not in price_table:
            continue
        aubcs = []
        ci_halfwidths = []
        for dom in DOMAINS:
            key = f"{m}|{dom}"
            if key not in summary:
                continue
            s = summary[key]
            aubcs.append(s["aubc"])
            lo, hi = s["aubc_ci95"]
            ci_halfwidths.append((hi - lo) / 2.0)
        if not aubcs:
            continue
        pts.append((
            m,
            price_table[m]["output_price_usd"],
            float(np.mean(aubcs)),
            float(np.mean(ci_halfwidths)) if ci_halfwidths else 0.0,
        ))

    fig, ax = plt.subplots(figsize=(4.6, 3.6))

    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    errs = np.array([p[3] for p in pts])

    for (m, x, y, e) in pts:
        ax.errorbar(
            x, y, yerr=e, fmt="o", color=colors[m], markersize=4.5,
            capsize=2, elinewidth=0.8, markeredgecolor="black", markeredgewidth=0.3,
            zorder=3,
        )

    # 帕累托前沿:价低且AUBC高者。按价格升序扫描,保留AUBC严格超过当前最大值的点。
    order_idx = np.argsort(xs)
    frontier_idx = []
    best_y = -np.inf
    for i in order_idx:
        if ys[i] > best_y:
            frontier_idx.append(i)
            best_y = ys[i]
    if len(frontier_idx) >= 2:
        fx = xs[frontier_idx]
        fy = ys[frontier_idx]
        ax.plot(fx, fy, linestyle="--", color="grey", linewidth=1.0, zorder=1,
                 label="Pareto frontier")

    # 标注模型名,避让重叠
    texts = []
    for (m, x, y, e) in pts:
        texts.append(ax.text(x, y, m, fontsize=6))
    if HAVE_ADJUST_TEXT and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.4),
        )
    else:
        # 手动小幅偏移兜底(不使用adjustText时)
        for t in texts:
            x, y = t.get_position()
            t.set_position((x, y))
            t.set_ha("left")
            t.set_va("bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Output price (USD / 1M tokens, list price)")
    ax.set_ylabel("AUBC (mean over 3 domains)")
    ax.grid(True, which="major", linewidth=0.3, alpha=0.4)
    if len(frontier_idx) >= 2:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig2_aubc_vs_price.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_aubc_vs_price.png", bbox_inches="tight")
    plt.close(fig)
    return FIG_DIR / "fig2_aubc_vs_price.pdf"


# ---------------------------------------------------------------------------
# fig3: 排名表(不限预算 vs 三档预算) + Kendall tau
# ---------------------------------------------------------------------------

def _interp_s_at_b(grid: list[float], s_of_b: list[float], b: float) -> float:
    """在log(B)空间线性插值S(B);b落在网格范围外则clip到端点。"""
    grid_arr = np.asarray(grid)
    s_arr = np.asarray(s_of_b)
    log_grid = np.log10(grid_arr)
    log_b = np.log10(b)
    if log_b <= log_grid[0]:
        return float(s_arr[0])
    if log_b >= log_grid[-1]:
        return float(s_arr[-1])
    return float(np.interp(log_b, log_grid, s_arr))


def fig3_rank_table():
    summary = load_summary()
    model_ids = model_ids_in_summary(summary)

    rows = []  # 用于CSV
    kendall_lines = []

    for dom in DOMAINS:
        dom_models = [m for m in model_ids if f"{m}|{dom}" in summary]
        if not dom_models:
            continue

        pass1 = {m: summary[f"{m}|{dom}"]["pass1"] for m in dom_models}
        # 不限预算排名(按pass1降序,1=最好)
        rank_unlimited = _rank_desc(pass1)

        # 预算档位:该域全体模型 b_at_50 的中位数 * {0.3, 1, 3}
        b50_vals = [
            summary[f"{m}|{dom}"]["b_at_50"]
            for m in dom_models
            if summary[f"{m}|{dom}"]["b_at_50"] is not None
        ]
        if not b50_vals:
            print(f"[fig3] 域 {dom}: 所有模型 b_at_50 均为 None,跳过预算档排名")
            continue
        median_b50 = float(np.median(b50_vals))
        tiers = {"0.3x": median_b50 * 0.3, "1x": median_b50 * 1.0, "3x": median_b50 * 3.0}

        tier_ranks = {}
        for tier_name, b in tiers.items():
            s_at_b = {}
            for m in dom_models:
                s = summary[f"{m}|{dom}"]
                s_at_b[m] = _interp_s_at_b(s["grid"], s["s_of_b"], b)
            tier_ranks[tier_name] = _rank_desc(s_at_b)

        for m in dom_models:
            rows.append({
                "domain": dom,
                "model": m,
                "pass1": round(pass1[m], 4),
                "rank_unlimited": rank_unlimited[m],
                "budget_0.3x_usd": round(tiers["0.3x"], 6),
                "rank_0.3x": tier_ranks["0.3x"][m],
                "budget_1x_usd": round(tiers["1x"], 6),
                "rank_1x": tier_ranks["1x"][m],
                "budget_3x_usd": round(tiers["3x"], 6),
                "rank_3x": tier_ranks["3x"][m],
            })

        kendall_lines.append(f"# Domain: {dom} (median b_at_50 = {median_b50:.6f} USD, n_models={len(dom_models)})")
        for tier_name in ["0.3x", "1x", "3x"]:
            r1 = [rank_unlimited[m] for m in dom_models]
            r2 = [tier_ranks[tier_name][m] for m in dom_models]
            tau, p = kendalltau(r1, r2)
            kendall_lines.append(
                f"  budget={tier_name} (B={tiers[tier_name]:.6f} USD): "
                f"kendall_tau={tau:.4f}, p_value={p:.4g}"
            )
        kendall_lines.append("")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIG_DIR / "fig3_rank_table.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "domain", "model", "pass1", "rank_unlimited",
            "budget_0.3x_usd", "rank_0.3x",
            "budget_1x_usd", "rank_1x",
            "budget_3x_usd", "rank_3x",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    txt_path = FIG_DIR / "fig3_kendall.txt"
    header = [
        "Kendall's tau: rank-under-budget vs rank-by-unconstrained-pass1",
        "(tau near 1 => budget-constrained ranking agrees with unconstrained pass1 ranking;",
        " tau near 0 or negative => budget materially reshuffles the leaderboard)",
        "",
    ]
    txt_path.write_text("\n".join(header + kendall_lines))

    return csv_path, txt_path


def _rank_desc(values: dict[str, float]) -> dict[str, float]:
    """按value降序排名(1=最大值)。并列取平均名次(标准竞赛排名的平均法)。"""
    items = sorted(values.items(), key=lambda kv: -kv[1])
    ranks = {}
    i = 0
    n = len(items)
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# fig4: 牌价综合单价排名 vs 实测成本排名
# ---------------------------------------------------------------------------

def fig4_price_vs_realized():
    summary = load_summary()
    price_table = load_price_table_usd()
    model_ids = model_ids_in_summary(summary)

    list_price = {}
    realized_cost = {}
    for m in model_ids:
        if m not in price_table:
            continue
        p = price_table[m]
        list_price[m] = p["input_price_usd"] * 0.9 + p["output_price_usd"] * 0.1
        costs = [
            summary[f"{m}|{dom}"]["mean_cost"]
            for dom in DOMAINS
            if f"{m}|{dom}" in summary
        ]
        if costs:
            realized_cost[m] = float(np.mean(costs))

    common = sorted(set(list_price) & set(realized_cost))
    rank_list_price = _rank_desc({m: -list_price[m] for m in common})   # 排名1=最便宜(价最低)
    rank_realized = _rank_desc({m: -realized_cost[m] for m in common})  # 排名1=实测最便宜

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    xs = [rank_list_price[m] for m in common]
    ys = [rank_realized[m] for m in common]
    ax.scatter(xs, ys, s=22, color="tab:blue", edgecolor="black", linewidth=0.3, zorder=3)

    n = len(common)
    lims = [0.5, n + 0.5]
    ax.plot(lims, lims, linestyle="--", color="grey", linewidth=1.0, zorder=1,
             label="y = x (no reversal)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Rank by list price (input*0.9 + output*0.1, USD)")
    ax.set_ylabel("Rank by realized cost / task")
    ax.grid(True, which="major", linewidth=0.3, alpha=0.4)
    ax.legend(frameon=False, loc="upper left")

    texts = []
    for m in common:
        texts.append(ax.text(rank_list_price[m], rank_realized[m], m, fontsize=6))
    if HAVE_ADJUST_TEXT and texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.4))

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig4_price_vs_realized.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_price_vs_realized.png", bbox_inches="tight")
    plt.close(fig)

    # 价格反转:两两模型对比,牌价排名顺序与实测排名顺序不一致
    n_pairs = 0
    n_reversed = 0
    reversed_pairs = []
    for a, b in combinations(common, 2):
        n_pairs += 1
        list_order = np.sign(rank_list_price[a] - rank_list_price[b])
        realized_order = np.sign(rank_realized[a] - rank_realized[b])
        if list_order != 0 and realized_order != 0 and list_order != realized_order:
            n_reversed += 1
            reversed_pairs.append((a, b))

    frac = n_reversed / n_pairs if n_pairs else float("nan")
    note_lines = [
        "Price-reversal analysis: pairs where list-price rank order and",
        "realized-cost rank order disagree (i.e. model A is listed cheaper than",
        "model B by list price, but ends up more expensive per task in practice).",
        "",
        f"Models compared: {len(common)}",
        f"Total pairs: {n_pairs}",
        f"Reversed pairs: {n_reversed}",
        f"Reversed fraction: {frac:.4f} ({frac*100:.2f}%)",
        "",
        "Reversed pairs (model_a, model_b):",
    ]
    for a, b in reversed_pairs:
        note_lines.append(f"  {a} <-> {b}")
    (FIG_DIR / "fig4_note.txt").write_text("\n".join(note_lines) + "\n")

    return FIG_DIR / "fig4_price_vs_realized.pdf", FIG_DIR / "fig4_note.txt"


# ---------------------------------------------------------------------------
# table1: 主结果表
# ---------------------------------------------------------------------------

def table1_main():
    summary = load_summary()
    model_ids = model_ids_in_summary(summary)

    def fmt_ci(s):
        return f"{s['aubc']:.4f} [{s['aubc_ci95'][0]:.4f}, {s['aubc_ci95'][1]:.4f}]"

    rows = []
    for m in model_ids:
        for dom in DOMAINS:
            key = f"{m}|{dom}"
            if key not in summary:
                continue
            s = summary[key]
            rows.append({
                "model": m,
                "domain": dom,
                "n_valid": s["n_valid"],
                "pass1": round(s["pass1"], 4),
                "mean_cost_usd": round(s["mean_cost"], 6),
                "aubc_with_ci95": fmt_ci(s),
                "b_at_50_usd": (round(s["b_at_50"], 6) if s["b_at_50"] is not None else ""),
                "b_at_80_usd": (round(s["b_at_80"], 6) if s["b_at_80"] is not None else ""),
            })

        # 三域平均行
        doms_present = [dom for dom in DOMAINS if f"{m}|{dom}" in summary]
        if not doms_present:
            continue
        n_valid_vals = [summary[f"{m}|{dom}"]["n_valid"] for dom in doms_present]
        pass1_vals = [summary[f"{m}|{dom}"]["pass1"] for dom in doms_present]
        cost_vals = [summary[f"{m}|{dom}"]["mean_cost"] for dom in doms_present]
        aubc_vals = [summary[f"{m}|{dom}"]["aubc"] for dom in doms_present]
        ci_lo_vals = [summary[f"{m}|{dom}"]["aubc_ci95"][0] for dom in doms_present]
        ci_hi_vals = [summary[f"{m}|{dom}"]["aubc_ci95"][1] for dom in doms_present]
        b50_vals = [summary[f"{m}|{dom}"]["b_at_50"] for dom in doms_present
                    if summary[f"{m}|{dom}"]["b_at_50"] is not None]
        b80_vals = [summary[f"{m}|{dom}"]["b_at_80"] for dom in doms_present
                    if summary[f"{m}|{dom}"]["b_at_80"] is not None]

        rows.append({
            "model": m,
            "domain": "AVERAGE",
            "n_valid": round(float(np.mean(n_valid_vals)), 1),
            "pass1": round(float(np.mean(pass1_vals)), 4),
            "mean_cost_usd": round(float(np.mean(cost_vals)), 6),
            "aubc_with_ci95": (
                f"{np.mean(aubc_vals):.4f} "
                f"[{np.mean(ci_lo_vals):.4f}, {np.mean(ci_hi_vals):.4f}]"
            ),
            "b_at_50_usd": (round(float(np.mean(b50_vals)), 6) if b50_vals else ""),
            "b_at_80_usd": (round(float(np.mean(b80_vals)), 6) if b80_vals else ""),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "table1_main.csv"
    fieldnames = [
        "model", "domain", "n_valid", "pass1", "mean_cost_usd",
        "aubc_with_ci95", "b_at_50_usd", "b_at_80_usd",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    return csv_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if not SUMMARY_JSON.exists():
        raise SystemExit(f"未找到 {SUMMARY_JSON},请先运行 analysis/run_analysis.py")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] fig1_sb_curves ...")
    p1 = fig1_sb_curves()
    print(f"      -> {p1}")

    print("[2/5] fig2_aubc_vs_price ...")
    p2 = fig2_aubc_vs_price()
    print(f"      -> {p2}")

    print("[3/5] fig3_rank_table + kendall ...")
    p3_csv, p3_txt = fig3_rank_table()
    print(f"      -> {p3_csv}")
    print(f"      -> {p3_txt}")

    print("[4/5] fig4_price_vs_realized ...")
    p4_pdf, p4_note = fig4_price_vs_realized()
    print(f"      -> {p4_pdf}")
    print(f"      -> {p4_note}")

    print("[5/5] table1_main ...")
    t1 = table1_main()
    print(f"      -> {t1}")

    print("\n全部图表已生成于:", FIG_DIR)


if __name__ == "__main__":
    main()
