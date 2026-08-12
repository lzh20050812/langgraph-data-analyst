"""Deterministic capability-boundary checks for concepts absent from the schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UnsupportedRequest:
    capability: str
    reason: str


_UNSUPPORTED_GROUPS = (
    (("身份证", "护照号", "社会保障号"), "sensitive_identity",
     "当前 Schema 不含身份证、护照等敏感身份字段，customer_id 不能作为其替代。"),
    (("信用卡号", "银行卡号", "发卡银行", "卡号"), "payment_card_pii",
     "当前 Schema 仅含支付方式，不含卡号或发卡银行字段。"),
    (("利润", "毛利", "净利", "成本", "进货价"), "profit_cost",
     "当前 Schema 不含商品成本或进货价，无法可靠计算利润。"),
    (("库存", "仓库", "缺货", "存货"), "inventory",
     "当前 Schema 不含库存量、仓库或补货字段，销量不能替代库存。"),
    (("员工", "工资", "薪资", "部门人数", "人事"), "human_resources",
     "当前 Schema 不含员工、薪酬或组织部门数据。"),
)


def detect_unsupported_request(query: str) -> Optional[UnsupportedRequest]:
    text = (query or "").strip().lower()
    for terms, capability, reason in _UNSUPPORTED_GROUPS:
        if any(term in text for term in terms):
            return UnsupportedRequest(capability, reason)
    if "流失" in text and any(term in text for term in ("每月", "月度", "按月", "各月")):
        return UnsupportedRequest(
            "churn_timeline",
            "customers.churned 只有静态标签，缺少 churn_date，无法将流失归因到具体月份。",
        )
    return None
