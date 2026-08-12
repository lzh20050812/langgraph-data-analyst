"""Formal persistent-memory retrieval and reuse ablation."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.prompts.report_prompt import build_report_prompt
from storage.chromadb.analysis_memory import get_analysis_memory


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"evaluation/results/formal_comparison/20260812_005036/raw/full_multi_agent.json"
OUT = ROOT/"evaluation/results/memory_20260812"
PARAPHRASES = {
    41: "如何针对容易流失的用户设计分层召回与留存动作",
    42: "请规划下一季度促销活动并给出预算投放方向",
    43: "当前商品组合有哪些问题，新品和品类结构应该怎样调整",
    44: "怎样升级高价值会员权益来增强忠诚度",
    45: "退单偏高可能由什么造成，应如何系统降低退货",
    46: "生成本月电商经营复盘，覆盖客户、销售、预测和措施",
    47: "做一次全局经营体检，重点关注流失预警、销售走势与品类",
    48: "根据复购表现判断高价值用户后续消费并设计精准触达",
    49: "需要一份用户分层、营收预估与渠道改进的综合报告",
    50: "从客户价值、销售、商品和渠道维度评价公司并提出季度战略",
}


def summarize(details, method):
    rows=[r for r in details if r["method"]==method]
    return {"n":len(rows), "recall_at_1":round(np.mean([r["hit_at_1"] for r in rows]),6),
            "recall_at_3":round(np.mean([r["hit_at_3"] for r in rows]),6),
            "mrr_at_3":round(np.mean([r["reciprocal_rank"] for r in rows]),6),
            "median_latency_ms":round(float(np.median([r["latency_ms"] for r in rows])),3),
            "p95_latency_ms":round(float(np.quantile([r["latency_ms"] for r in rows],.95)),3)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    payload=json.loads(SOURCE.read_text(encoding="utf-8"))
    source={int(row["id"]):row for row in payload["details"] if row.get("report")}
    memory=get_analysis_memory(); memory.clear()
    for qid,row in source.items():
        memory.remember(query=row["query"],report=row["report"],category=row["category"],source_id=str(qid))
    details=[]
    ids=sorted(PARAPHRASES)
    for qid in ids:
        query=PARAPHRASES[qid]
        tick=time.perf_counter(); recalled=memory.recall(query,top_k=3); latency=(time.perf_counter()-tick)*1000
        ranking=[int(r["memory_id"]) for r in recalled]
        rank=ranking.index(qid)+1 if qid in ranking else 0
        details.append({"method":"vector_memory","source_id":qid,"query":query,"ranking":ranking,
                        "hit_at_1":rank==1,"hit_at_3":rank>0,"reciprocal_rank":1/rank if rank else 0,
                        "latency_ms":round(latency,3)})
        rng=random.Random(42+qid); random_rank=rng.sample(ids,3); rr=random_rank.index(qid)+1 if qid in random_rank else 0
        details.append({"method":"random","source_id":qid,"query":query,"ranking":random_rank,
                        "hit_at_1":rr==1,"hit_at_3":rr>0,"reciprocal_rank":1/rr if rr else 0,"latency_ms":0})

    # Verify that retrieved context is materially inserted into the downstream Report prompt.
    reuse_checks=[]
    for qid in ids[:5]:
        memories=memory.recall(PARAPHRASES[qid],top_k=3)
        without=build_report_prompt(PARAPHRASES[qid],{}, {}, {}, {}, historical_memory=[])
        with_memory=build_report_prompt(PARAPHRASES[qid],{}, {}, {}, {}, historical_memory=memories)
        reuse_checks.append({"source_id":qid,"retrieved":len(memories),
            "prompt_chars_without":len(without),"prompt_chars_with":len(with_memory),
            "top_memory_present":memories[0]["query"] in with_memory,
            "guard_present":"不得把历史数字当作本次事实" in with_memory})

    result={"protocol":{"seed_memories":len(source),"paraphrase_queries":len(ids),
              "collection":"analysis_memory","embedding":"BAAI/bge-small-zh",
              "ablation":"vector memory vs seeded random retrieval"},
            "summaries":{"vector_memory":summarize(details,"vector_memory"),"random":summarize(details,"random")},
            "downstream_reuse":{"checks":len(reuse_checks),
              "context_injection_rate":round(np.mean([r["top_memory_present"] for r in reuse_checks]),6),
              "fact_guard_rate":round(np.mean([r["guard_present"] for r in reuse_checks]),6)},
            "details":details,"reuse_checks":reuse_checks}
    (OUT/"memory_metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(details).to_csv(OUT/"memory_retrieval_details.csv",index=False,encoding="utf-8-sig")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
