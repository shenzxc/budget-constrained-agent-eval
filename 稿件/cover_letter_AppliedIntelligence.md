# Cover Letter — Applied Intelligence

Dear Editor-in-Chief of *Applied Intelligence*,

I am pleased to submit my manuscript, **"Budget-Constrained Evaluation of Open-Weight LLM Agents: Success–Budget Curves, Price Reversal, and Thinking-Budget Saturation,"** as an original research article for consideration by *Applied Intelligence*.

**Fit with the journal.** The paper addresses a real-life, complex decision problem at the core of *Applied Intelligence*'s scope: how to select and deploy intelligent agents under operational budget constraints. Large language model (LLM) agents are now deployed for multi-turn, tool-using tasks whose inference cost accumulates over many steps, yet practitioners lack a principled way to decide *which agent to run under a given per-task budget*. The paper turns this deployment decision into a measurable, table-lookup object, providing a decision-support instrument for cost-aware model selection.

**Why this is a methodological contribution, not another benchmark.** I want to be explicit about the novelty, because the contribution is conceptual before it is empirical:

1. **A new evaluation semantics.** Prior cost-aware work (e.g., the Holistic Agent Leaderboard; Cost-of-Pass; single-turn "price reversal" studies) *records or aggregates realized spending*. This paper instead imposes a *prospective per-task budget constraint* and measures how success responds to it—answering "what can be accomplished under budget B," which is the form a deployment decision actually takes.

2. **A theorem that makes the protocol practical.** I prove a *truncation-equivalence* property: for a budget-unaware agent, the entire success–budget curve can be reconstructed offline from a single unconstrained run, so evaluation cost is independent of the number of budget levels. This is what turns an otherwise expensive multi-budget sweep into a tractable protocol.

3. **A decision-support output.** The protocol yields a *budget-interval-to-best-model lookup table* via pairwise curve crossovers, directly usable for model selection under a service-level budget.

A positioning matrix in the paper (Table 1) contrasts these properties against the closest prior art; to my knowledge no existing method combines a prospective budget constraint, offline whole-curve derivation, and a deployment-ready selection rule.

**Empirical scope.** The protocol is validated at scale—12 open-weight and public-API model configurations (DeepSeek-V4 and Qwen3 families, ~100× price range), 4,448 multi-turn trajectories on τ³-bench, plus a cross-benchmark generalization test on BFCL (Kendall τ = 0.758, p < 0.001)—yielding four deployment-relevant findings on leaderboard reshuffling, the absence of price reversal in multi-step tasks, thinking-budget saturation, and hosted-endpoint reliability. Statistical rigor (paired bootstrap CIs, McNemar tests, Holm correction, minimum-detectable-effect analysis) and full open-science release (code, all trajectories, per-token cost records; Zenodo DOI 10.5281/zenodo.21215799) support reproducibility.

This manuscript is original, has not been published elsewhere, and is not under consideration by any other journal. As an independent researcher, I do not hold an institutional e-mail address; my personal e-mail (19708110566@139.com) serves as the corresponding contact. The author declares no competing interests, received no specific funding, and has provided ethics, data-availability, and generative-AI-use statements in the manuscript.

Thank you for considering my work.

Sincerely,

Weiming Shen
Independent Researcher, Suqian, Jiangsu, China
ORCID: 0009-0006-8222-1668
19708110566@139.com
