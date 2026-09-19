# Stage 5 deterministic regression

- Protocol: `stage5-deterministic-v1`
- Dataset audit: **passed** (179 samples)
- Per-sample deterministic runs: **31**
- Overall status: **passed**
- Paid LLM calls: **0**

## Metrics

- Semantic answer accuracy: `1.0` (`n=15`)
- Support transfer table recall pass rate: `1.0`
- Support transfer field recall pass rate: `1.0`
- Relationship coverage: `1.0`
- Multi-turn completion: `1.0`
- Multi-turn inheritance correctness: `1.0`
- Latency P50/P95: `0.161` / `0.397` ms

## Boundary

This fast run validates labels, deterministic semantic routing, transfer retrieval, and multi-turn state. It is not a substitute for the paid, same-model architecture comparison and makes no claim about current LLM accuracy.
