# Evaluation Matrix

| ID | Capability | Comparison | Output |
|---|---|---|---|
| E1 | Data governance | raw vs governed data | repair, retention, and runtime metrics |
| E2 | Schema grounding | BGE vs character TF-IDF vs random | Recall@K, MRR, significance |
| E3 | Text2SQL | RAG context vs full schema | execution accuracy and paired outcomes |
| E4 | SQL correction | first attempt vs final attempt | recovery rate and retry trace |
| E5 | Multi-agent orchestration | single LLM vs RAG LLM vs full workflow | task completion, SQL, report quality |
| E6 | Analytics models | candidate cluster counts and predictive baselines | silhouette, ARI, AUC, MAPE |
| E7 | Runtime system | serial and concurrent requests | health, contract, latency, throughput |
| E8 | Safety and memory | OOD/injection cases and retrieval baselines | fail-closed rate, invariants, Recall@K |

Every comparison uses a fixed test-set snapshot and records per-sample outcomes before aggregate metrics are computed.
