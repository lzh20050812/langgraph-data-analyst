# Evaluation Protocol V2

## Scope and dataset boundary

The versioned catalog is `evaluation/data/benchmark_catalog_v2026_09.json`. It contains 179 repository-curated and deterministically audited samples. Text dataset SHA-256 values are computed after normalizing line endings to UTF-8 LF so that Windows and Linux checkouts have the same identity. This count is a coverage inventory, not a claim that every label received a new independent human review in September 2026.

| Usage | Purpose | Rule |
|---|---|---|
| `development_regression` | Fast implementation feedback | May be inspected and rerun during development; never reported as held-out generalization. |
| `validation_and_frozen_test` | Retrieval parameter selection plus held-out reporting | The 12 validation IDs select RRF parameters; the disjoint 28 IDs are reported once as held-out test results. |
| `frozen_test` | Final comparison | Must not be used to select prompts, thresholds, or retries. New results go to a new run directory and never overwrite `evaluation/final_results/`. |
| `transfer_test` | Different-Schema migration check | Report separately from the e-commerce suite; eight support cases are too small for broad generalization. |

The catalog hashes every source file and audits unique IDs, required labels, SQL parsing, known tables/fields, known semantic recipes, split disjointness, and linked failure regressions. The 15 semantic paraphrases cover unfamiliar wording, eight support cases cover a second Schema, and eight multi-turn cases cover inheritance, replacement, clearing, clarification, and incompatible inherited conditions.

## Independent metrics

Metrics are calculated only when a sample has the corresponding label. Missing labels have denominator `n=0` and value `null`; they are never converted to failures or successes.

| Capability | Automated metrics | Required human review |
|---|---|---|
| Retrieval | table recall, field recall, relationship-path coverage, P50/P95 latency | Whether labels contain every field actually necessary to answer the question. |
| Query | business-answer accuracy, condition-retention rate, execution success, repair success | Ambiguous questions, acceptable equivalent grains, and extra-but-valid result columns. |
| Multi-turn | task completion, inheritance correctness, clarification correctness | Whether the follow-up interpretation matches normal user expectations. |
| Report | numeric consistency, citation correctness, unsupported-claim-free rate | Causal wording, usefulness, limitation clarity, and actionability; reviewers must be blind to architecture. |
| Engineering | P50/P95 latency, LLM/tool calls, prompt/completion/total tokens, estimated cost, recovery success | Operational acceptability under the declared budget and environment. |

### Human scoring rubric

Each report receives independent 0–2 scores for evidence correctness, question coverage, limitation disclosure, and actionable usefulness. A score of 0 means materially wrong or unsupported, 1 means partially correct/incomplete, and 2 means correct and sufficient. Reviewers record cited fact IDs and an error category. Disagreements are retained, then adjudicated by a third review; the original scores are not deleted. LLM-as-Judge can be reported as secondary analysis only.

## Per-sample artifact contract

Every architecture run preserves sample/dataset/split IDs, architecture, success, duration, LLM/tool calls, token counts, estimated cost, error and failure category, bounded output, full configuration, and code/protocol versions. Failed samples must have a failure category. Prompts and secrets are not stored.

Failure categories use stable families: `scope_rejection`, `retrieval_miss`, `sql_generation`, `sql_safety`, `execution_failure`, `answer_mismatch`, `condition_loss`, `report_grounding`, `budget_exhausted`, and `runtime_error`. New categories require a protocol-version change or an explicit mapping note.

## Architecture comparison

`python -m evaluation.experiments.architecture_comparison_v2` is dry-run only by default. The V2 comparison fixes the same model, temperature, 50-task snapshot, public Schema catalog, table permissions, read-only database, SQL timeout, row cap, and global call/token/cost budget across:

1. `direct_sql`: one-shot SQL with the full authorized Schema;
2. `single_agent_tools`: one agent using scoped Schema retrieval and database execution;
3. `controlled_workflow`: the current planned, validated workflow.

The difference in how Schema is delivered is part of the architecture under test; the underlying catalog and data authorization are the same. A paid run requires `--execute`, the exact confirmation `CONFIRM_PAID_EVALUATION`, and positive sample, call, token, and USD caps. Input/output prices must match the selected provider and run date. The command refuses an uncapped paid run.

The 2026-08-11 comparison remains historical evidence only. Its single-LLM lower bound did not receive private Schema context, so it is not reused as a current fair architecture comparison.

## Reproduction and CI

Fast deterministic validation runs with:

```bash
python -m evaluation.experiments.stage5_deterministic_regression --check
```

It performs no network or LLM calls and is suitable for CI. Omitting `--check` writes a new post-freeze artifact under `evaluation/optimization_results/stage5_20260918/`. Historical frozen metrics and manifests remain unchanged.
