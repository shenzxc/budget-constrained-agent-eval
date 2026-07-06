# Codex 任务清单:BFCL 第二基准的分析与论文整合

> 架构与验收:主模型(Claude)。执行:Codex。协作方式:Codex 在分支上工作并开 PR,主模型在 GitHub 上 `gh pr diff` 审核。
> 本文件是**自包含**的:所有输入都是仓库里已提交的派生数据(CSV/JSON),**不需要原始轨迹数据、不需要调用任何 API、不产生任何费用**。

## 背景(读这段就够,不必翻全项目)

论文《Budget-Constrained Evaluation of Open-Weight LLM Agents》已在 τ³-bench 三个客服域完成主实验(12 模型 × 3 域,结论见 `experiments/analysis/output/summary.json`)。外部评审要求补一个**不同任务形态的第二基准**以证明结论可外推。我们选了 BFCL 多轮函数调用(multi_turn_base,固定前 100 任务 × 同样 12 个模型),数据采集在主模型机器上进行,产物为逐(模型,任务)长表 CSV。

**你的总目标**:把 BFCL 数据也变成同一套预算受控指标,并回答关键科学问题——**τ³-bench 上得到的"预算重排排行榜、便宜模型占优"结论,在 BFCL 上是否同样成立?** 这是论文能否冲更好期刊的核心增量。

## 输入文件(均已提交在仓库)

- `experiments/analysis/output/bfcl_multi_turn_base.csv` —— BFCL 长表。列:`model, task_id, success(bool), n_requests, total_input_tokens, total_output_tokens, total_latency_s`。
  - ⚠️ 当前提交的是**部分模型**的数据(主实验还在跑最后几个模型)。**你的代码必须数据无关**:读到几个模型就处理几个,不要硬编码模型数或任务数。主模型会在全量跑完后提交完整 CSV,届时你的同一份代码应能直接产出最终数字。
- `experiments/analysis/output/summary.json` —— τ³-bench 的结果。键为 `"模型id|域"`,值含 `pass1, mean_cost, aubc, aubc_ci95, b_at_50, b_at_80, grid, s_of_b`。
- `experiments/analysis/models.yaml` —— 价格表。每个模型有 `id, litellm, input_price, output_price, currency`;`exchange_rate_usd_cny` 字段用于把 CNY 折算成 USD。BFCL 里模型名带 `-FC` 后缀,去掉后缀即对应 models.yaml 的某个模型(注意 `qwen3-235b-a22b-instruct-2507-FC`→id `qwen3-235b-instruct`,`...thinking-2507-FC`→`qwen3-235b-thinking`;`qwen3-32b-FC`→`qwen3-32b` 按非思考价)。
- `experiments/analysis/budget_curves.py` —— **参考实现**。里面的 `aubc()`, `budget_at_tau()`, `s_of_b()` 逻辑可直接复用;但它的 `load_tau2_dir` 是给 τ³-bench 格式的,BFCL 需要你自己按长表 CSV 构造等价的成本序列(BFCL 只有每任务的 total tokens,没有逐步序列,所以每个任务视作单点:`cost = (in_tok*α_in + out_tok*α_out)/1e6`,S(B) 用"成功且成本≤B"的比例,与 τ³ 口径一致)。

## 环境

- Python 用仓库根目录下 `experiments/.venv`(已装 numpy/scipy/matplotlib/pyyaml)。运行:`experiments/.venv/bin/python <脚本>`。
- 若在自己的克隆环境无此 venv,自建 venv 装 `numpy scipy matplotlib pyyaml` 即可,这些任务无需项目的重型依赖。
- 图表风格对齐已有图:serif 字体、font size 9、dpi 300、PDF 为主 PNG 为辅,全英文标签。参考 `experiments/analysis/make_figures.py` 的 rcParams。

---

## 任务 C1:BFCL 的预算受控指标

**产出** `experiments/analysis/bfcl_curves.py`,运行后生成:
- `experiments/analysis/output/bfcl_summary.json`:每个模型一条,含 `n, pass1, mean_cost_usd, aubc, aubc_ci95, b_at_50, b_at_80, grid, s_of_b`。口径与 τ³ 的 `summary.json` **完全一致**(便于跨基准对比),预算网格用全体 BFCL 模型成本范围的 log 等距 200 点。
- `experiments/analysis/output/bfcl_summary_table.csv`:`model, n, pass1, mean_cost_usd, aubc, aubc_lo, aubc_hi, b_at_50, b_at_80`。
- AUBC 的置信区间用任务级 bootstrap(2000 次,固定 seed=0),与 τ³ 一致。

**验收**:S(B) 单调不减;S(B_max)==pass1;成本用 models.yaml 牌价、CNY 经汇率折 USD;脚本对部分数据不报错。

## 任务 C2:跨基准泛化分析(**最重要**)

**产出** `experiments/analysis/cross_benchmark.py`,运行后生成:
- `experiments/analysis/output/cross_benchmark.json`:对 τ³-bench 和 BFCL **共有**的模型,计算:
  - τ³ 的 AUBC(三域平均)排名 vs BFCL 的 AUBC 排名 的 Kendall τ 与 p 值(scipy.stats.kendalltau);
  - 同理 pass1 排名的 Kendall τ;
  - 一份两基准并排的排名表(每模型:τ³ AUBC 排名、BFCL AUBC 排名)。
- `experiments/analysis/output/figures/fig5_cross_benchmark.pdf/.png`:x=τ³ 三域平均 AUBC,y=BFCL AUBC,每模型一点、标注模型名,画对角参考线;标题区留空(论文用 caption)。
- 一个 `fig5_note.txt`:一句话结论,形如 "Kendall τ=0.__ (p=0.__) between τ³ and BFCL AUBC rankings over N shared models",供主模型写正文引用。

**科学意义提示**(帮助你自检结果是否合理):我们**预期** τ 为正且中高(说明"便宜模型在两个形态迥异的基准上都占优"→ 结论可外推)。若 τ 很低甚至为负,不要修饰,如实输出——主模型会据实讨论。

## 任务 C3:LaTeX 表格数据填充

`稿件/latex/manuscript.tex` 里 Table 1(`\label{tab:table1-main}`)当前是 `%TODO` 占位。
- 读 `experiments/analysis/output/table1_main.csv`,把 36 行(12 模型 × 3 域)数据填进 tabular,并按模型加三域平均行(或用 `\multirow` 让模型名跨行,择一即可,保持可读)。
- 数值格式:pass1 三位小数;mean_cost 用美元、`\$0.0000` 四位;AUBC 三位 + 方括号 CI;B@50/B@80 四位美元,`None` 显示为 `--`。
- **只改 Table 1 的数据行区域**,不要动 `manuscript.tex` 其他任何内容(正文由主模型写)。
- 若装了 tectonic,跑一次 `tectonic 稿件/latex/manuscript.tex` 确认能编译出 PDF(缺 BFCL 表不影响);没有则跳过编译,保证 tabular 语法自查配对正确。

---

## 交付与审核流程(重要)

1. 在分支 `codex/bfcl-analysis` 上工作(不要直接改 main)。
2. 每个任务一个或多个 commit,commit message 说明改了什么。
3. 完成后 `git push` 并开 PR(标题 "BFCL second-benchmark analysis (C1-C3)"),PR 描述里贴上 C2 的 Kendall τ 结果数字和你对结果是否符合预期的判断。
4. 主模型会用 `gh pr diff` / `gh pr view` 审核;有问题在 PR 上留意见,你据此修改。
5. **不要**:改 `.env`、改 `experiments/analysis/` 下的现有文件(`budget_curves.py`/`run_analysis.py`/`make_figures.py`/`models.yaml`——新建文件可以)、改 `manuscript.tex` 的正文区、动 τ³ 的原始数据或 summary.json。

## 硬约束
- 全程离线、零 API 调用、零费用。
- 结果如实——负面/不显著结果照样输出,不修饰。
- 数据无关编码——读到多少模型处理多少,不硬编码。
