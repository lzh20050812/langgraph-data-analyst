# Reproducible Benchmarks

This document describes the evaluation boundary for AI Data Analyst. All metrics are generated from fixed local test sets and are intended for regression tracking, architecture comparison, and failure analysis. They are not claims about unrestricted open-domain performance.

## Benchmark matrix

| Area | Entry point | Primary metrics |
|---|---|---|
| Data governance | `evaluation/experiments/governance_experiment.py` | missing-value repair, duplicate repair, record retention, repeat runtime |
| Schema retrieval | `evaluation/experiments/rag_retrieval_formal.py` | Field/Table Recall@K, MRR, paired significance |
| Text2SQL | `evaluation/experiments/text2sql_ablation_formal.py` | execution accuracy, first-pass accuracy, retry recovery, McNemar test |
| Multi-agent orchestration | `evaluation/experiments/run_formal_comparison.py` | task completion score, SQL accuracy, report quality, route efficiency |
| Analytics and prediction | `evaluation/experiments/model_experiment_formal.py` | silhouette, ARI, ROC-AUC, MAPE, baseline deltas |
| Runtime and API | `evaluation/experiments/system_experiment_formal.py` | contract compliance, latency, throughput, concurrency success rate |
| Security and scope | `evaluation/experiments/robustness_experiment.py` | OOD fail-closed rate, prompt-injection isolation, database invariants |
| Analysis memory | `evaluation/experiments/memory_reuse_experiment.py` | Recall@K, MRR, prompt integration, random baseline |

## Test data

- `evaluation/data/text2sql/`: fixed Text2SQL questions and audited references.
- `evaluation/data/rag_schema/`: Schema retrieval queries and field/table labels.
- `evaluation/data/multi_agent/`: multi-intent tasks for orchestration comparison.

## Reproduction principles

1. Freeze the code, test-set snapshot, dependency versions, model name, and random seed.
2. Compare paired systems on the same samples and preserve per-sample outcomes.
3. Report strong deterministic or statistical baselines, including negative results.
4. Separate syntax validity, execution success, and business-semantic correctness.
5. Keep LLM-judged metrics secondary to executable and deterministic evidence.
6. Do not commit API keys, runtime logs, local model caches, or large raw outputs.

## Current validated results

- Data governance repaired 1,176 missing cells and 604 duplicate issues while retaining 8,000 unique customers.
- Text2SQL content execution accuracy reached 87.76% on 49 scorable questions.
- The deterministic business-semantics layer improved a 15-case high-risk subset from 46.67% to 100% (`p=0.0078`).
- XGBoost reached ROC-AUC 0.6777 versus 0.6453 for logistic regression; the confidence interval for the delta crossed zero.
- Prophet reached MAPE 15.65% and did not beat the last-value baseline at 14.36%.
- OOD requests failed closed in 6/6 cases and prompt-injection attempts were isolated in 4/4 cases.
- API success rate remained 100% at 1, 2, and 4 concurrent workers.

The repository intentionally keeps the benchmark harness and fixed test sets under version control while excluding bulky local run artifacts. This keeps clones small and makes new results independently reproducible.
