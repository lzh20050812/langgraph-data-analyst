"""
Schema Metadata —— 结构化表/字段元信息（人类可读的业务术语映射表）。

这是 Schema Agent 做语义检索的基础数据源：
- 每个字段包含 table_name, column_name, dtype, business_term, aliases
- 业务术语（business_term）是 embedding 的输入文本
- 同义词（aliases）覆盖常见口语表达，提升检索召回率

设计原则：
- 不把元数据写死在代码里是无法维护的，集中在一个模块管理
- 新增表/字段时只需在此处补充
"""

from hashlib import sha256
import json
from typing import List, Dict


SCHEMA_CATALOG_VERSION = "2026.09.1"

# ============================================================
# 字段元数据定义
# ============================================================

# 每条记录对应数据库中的一个字段
SCHEMA_FIELDS: List[Dict] = [
    # ---- customers 表 ----
    {
        "table_name": "customers",
        "column_name": "customer_id",
        "dtype": "VARCHAR",
        "business_term": "客户唯一标识ID",
        "aliases": ["客户ID", "用户ID", "客户编号", "customer"],
    },
    {
        "table_name": "customers",
        "column_name": "country",
        "dtype": "VARCHAR",
        "business_term": "客户所在国家",
        "aliases": ["国家", "地区", "市场", "country", "nation"],
    },
    {
        "table_name": "customers",
        "column_name": "age",
        "dtype": "INT",
        "business_term": "客户年龄",
        "aliases": ["年龄", "岁数", "age"],
    },
    {
        "table_name": "customers",
        "column_name": "gender",
        "dtype": "VARCHAR",
        "business_term": "客户性别（Male/Female/Other）",
        "aliases": ["性别", "男女", "gender", "sex"],
    },
    {
        "table_name": "customers",
        "column_name": "membership_tier",
        "dtype": "VARCHAR",
        "business_term": "会员等级（Free/Silver/Gold/Platinum）",
        "aliases": ["会员", "等级", "会员级别", "VIP等级", "membership", "tier", "membership_level"],
    },
    {
        "table_name": "customers",
        "column_name": "registration_date",
        "dtype": "DATE",
        "business_term": "客户注册日期",
        "aliases": ["注册时间", "加入日期", "注册日", "signup date"],
    },
    {
        "table_name": "customers",
        "column_name": "total_orders",
        "dtype": "INT",
        "business_term": "客户历史累计订单数",
        "aliases": ["订单数", "下单次数", "购买次数", "总订单", "order count"],
    },
    {
        "table_name": "customers",
        "column_name": "total_spend_usd",
        "dtype": "DECIMAL",
        "business_term": "客户累计消费总额（美元）",
        "aliases": ["消费总额", "累计消费", "GMV", "营收", "消费金额", "spend", "revenue", "总花费", "交易额"],
    },
    {
        "table_name": "customers",
        "column_name": "avg_order_value_usd",
        "dtype": "DECIMAL",
        "business_term": "客户平均客单价（美元）",
        "aliases": ["客单价", "平均订单金额", "笔单价", "AOV", "average order value"],
    },
    {
        "table_name": "customers",
        "column_name": "days_since_last_purchase",
        "dtype": "INT",
        "business_term": "距离上次购买的天数",
        "aliases": ["上次购买距今", "最近一次购买", "最后购买时间", "recency", "最近购买间隔"],
    },
    {
        "table_name": "customers",
        "column_name": "preferred_category",
        "dtype": "VARCHAR",
        "business_term": "客户偏好的商品品类",
        "aliases": ["偏好品类", "喜欢类别", "常购品类", "品类偏好", "category preference"],
    },
    {
        "table_name": "customers",
        "column_name": "preferred_device",
        "dtype": "VARCHAR",
        "business_term": "客户偏好的购物设备（Mobile/Desktop/Tablet）",
        "aliases": ["设备偏好", "使用设备", "购物终端", "device"],
    },
    {
        "table_name": "customers",
        "column_name": "preferred_payment_method",
        "dtype": "VARCHAR",
        "business_term": "客户偏好的支付方式",
        "aliases": ["支付方式", "付款方式", "支付偏好", "payment method"],
    },
    {
        "table_name": "customers",
        "column_name": "acquisition_channel",
        "dtype": "VARCHAR",
        "business_term": "客户获客渠道",
        "aliases": ["获客来源", "引流渠道", "注册渠道", "来源渠道", "channel", "acquisition"],
    },
    {
        "table_name": "customers",
        "column_name": "reviews_given",
        "dtype": "INT",
        "business_term": "客户给出的评价数量",
        "aliases": ["评价数", "评论数", "点评次数", "review count"],
    },
    {
        "table_name": "customers",
        "column_name": "avg_review_score",
        "dtype": "DECIMAL",
        "business_term": "客户给出的平均评价分数（1-5）",
        "aliases": ["平均评分", "评价分数", "打分", "rating", "review score"],
    },
    {
        "table_name": "customers",
        "column_name": "returns_made",
        "dtype": "INT",
        "business_term": "客户退货次数",
        "aliases": ["退货数", "退货次数", "退单量", "return count"],
    },
    {
        "table_name": "customers",
        "column_name": "wishlist_items",
        "dtype": "INT",
        "business_term": "客户心愿单商品数量",
        "aliases": ["心愿单", "收藏商品数", "wishlist count"],
    },
    {
        "table_name": "customers",
        "column_name": "newsletter_subscribed",
        "dtype": "TINYINT",
        "business_term": "是否订阅营销邮件（1=是, 0=否）",
        "aliases": ["订阅", "邮件订阅", "newsletter", "营销订阅"],
    },
    {
        "table_name": "customers",
        "column_name": "churned",
        "dtype": "TINYINT",
        "business_term": "是否流失客户（1=已流失, 0=未流失）",
        "aliases": ["流失", "客户流失", "churn", "已流失", "流失标记"],
    },

    # ---- orders 表 ----
    {
        "table_name": "orders",
        "column_name": "order_id",
        "dtype": "VARCHAR",
        "business_term": "订单唯一标识ID",
        "aliases": ["订单ID", "订单号", "order", "订单编号"],
    },
    {
        "table_name": "orders",
        "column_name": "customer_id",
        "dtype": "VARCHAR",
        "business_term": "订单所属客户ID，关联 customers.customer_id",
        "aliases": ["订单客户ID", "客户关联", "用户关联", "customer id", "join customer"],
    },
    {
        "table_name": "orders",
        "column_name": "order_date",
        "dtype": "DATE",
        "business_term": "订单日期",
        "aliases": ["下单日期", "交易日期", "购买日期", "order date"],
    },
    {
        "table_name": "orders",
        "column_name": "year",
        "dtype": "INT",
        "business_term": "订单年份",
        "aliases": ["年份", "年"],
    },
    {
        "table_name": "orders",
        "column_name": "month",
        "dtype": "INT",
        "business_term": "订单月份（1-12）",
        "aliases": ["月份", "月"],
    },
    {
        "table_name": "orders",
        "column_name": "quarter",
        "dtype": "INT",
        "business_term": "订单季度（1-4）",
        "aliases": ["季度", "Q1", "Q2", "Q3", "Q4", "quarter"],
    },
    {
        "table_name": "orders",
        "column_name": "day_of_week",
        "dtype": "VARCHAR",
        "business_term": "订单星期几",
        "aliases": ["星期", "周几", "day of week"],
    },
    {
        "table_name": "orders",
        "column_name": "product_name",
        "dtype": "VARCHAR",
        "business_term": "商品名称",
        "aliases": ["产品名", "商品", "product"],
    },
    {
        "table_name": "orders",
        "column_name": "category",
        "dtype": "VARCHAR",
        "business_term": "商品品类",
        "aliases": ["品类", "类别", "商品类别", "category"],
    },
    {
        "table_name": "orders",
        "column_name": "unit_price_usd",
        "dtype": "DECIMAL",
        "business_term": "商品单价（美元）",
        "aliases": ["单价", "价格", "unit price"],
    },
    {
        "table_name": "orders",
        "column_name": "quantity",
        "dtype": "INT",
        "business_term": "购买数量",
        "aliases": ["数量", "件数", "quantity"],
    },
    {
        "table_name": "orders",
        "column_name": "subtotal_usd",
        "dtype": "DECIMAL",
        "business_term": "订单小计金额（美元）",
        "aliases": ["小计", "金额", "subtotal", "订单金额"],
    },
    {
        "table_name": "orders",
        "column_name": "discount_pct",
        "dtype": "DECIMAL",
        "business_term": "折扣百分比",
        "aliases": ["折扣率", "折扣", "discount", "打折"],
    },
    {
        "table_name": "orders",
        "column_name": "total_amount_usd",
        "dtype": "DECIMAL",
        "business_term": "订单总金额（含税含运费，美元）",
        "aliases": ["总金额", "订单总额", "total amount", "实付金额"],
    },
    {
        "table_name": "orders",
        "column_name": "payment_method",
        "dtype": "VARCHAR",
        "business_term": "支付方式",
        "aliases": ["付款方式", "payment"],
    },
    {
        "table_name": "orders",
        "column_name": "device_used",
        "dtype": "VARCHAR",
        "business_term": "下单设备（Mobile/Desktop/Tablet）",
        "aliases": ["设备", "下单设备", "device"],
    },
    {
        "table_name": "orders",
        "column_name": "delivery_days",
        "dtype": "INT",
        "business_term": "配送天数",
        "aliases": ["配送时间", "物流天数", "delivery time", "运输天数"],
    },
    {
        "table_name": "orders",
        "column_name": "order_status",
        "dtype": "VARCHAR",
        "business_term": "订单状态（Completed/Cancelled/Processing/Shipped）",
        "aliases": ["订单状态", "状态", "status"],
    },
    {
        "table_name": "orders",
        "column_name": "returned",
        "dtype": "TINYINT",
        "business_term": "是否退货（1=退货, 0=未退货）",
        "aliases": ["退货", "退货标记", "returned"],
    },
    {
        "table_name": "orders",
        "column_name": "customer_rating",
        "dtype": "DECIMAL",
        "business_term": "客户对订单的评分（1-5），约63%缺失",
        "aliases": ["订单评分", "客户评分", "rating", "评分"],
    },
    {
        "table_name": "orders",
        "column_name": "is_repeat_customer",
        "dtype": "TINYINT",
        "business_term": "是否为复购客户（1=复购, 0=新客）",
        "aliases": ["复购", "回头客", "repeat customer", "老客"],
    },
    {
        "table_name": "orders", "column_name": "discount_amount_usd", "dtype": "DECIMAL",
        "business_term": "订单折扣金额（美元）", "aliases": ["优惠金额", "折扣金额", "discount amount"],
    },
    {
        "table_name": "orders", "column_name": "shipping_fee_usd", "dtype": "DECIMAL",
        "business_term": "订单运费（美元）", "aliases": ["运费", "配送费", "shipping fee"],
    },
    {
        "table_name": "orders", "column_name": "tax_pct", "dtype": "DECIMAL",
        "business_term": "订单税率", "aliases": ["税率", "tax rate"],
    },
    {
        "table_name": "orders", "column_name": "tax_amount_usd", "dtype": "DECIMAL",
        "business_term": "订单税额（美元）", "aliases": ["税额", "税费", "tax amount"],
    },
    {
        "table_name": "orders", "column_name": "delivery_date", "dtype": "DATE",
        "business_term": "订单送达日期", "aliases": ["送达日期", "收货日期", "delivery date"],
    },
    {
        "table_name": "orders", "column_name": "session_duration_minutes", "dtype": "DECIMAL",
        "business_term": "下单会话时长（分钟）", "aliases": ["会话时长", "浏览时长", "session duration"],
    },
    {
        "table_name": "orders", "column_name": "pages_viewed_before_purchase", "dtype": "INT",
        "business_term": "购买前浏览页面数", "aliases": ["浏览页数", "购买前页面数", "pages viewed"],
    },

    # ---- monthly_revenue 表 ----
    {
        "table_name": "monthly_revenue", "column_name": "year", "dtype": "INT",
        "business_term": "月度营收记录所属年份", "aliases": ["营收年份", "年度", "year"],
    },
    {
        "table_name": "monthly_revenue", "column_name": "month", "dtype": "INT",
        "business_term": "月度营收记录所属月份（1-12）", "aliases": ["营收月份", "月份", "month"],
    },
    {
        "table_name": "monthly_revenue", "column_name": "quarter", "dtype": "INT",
        "business_term": "月度营收记录所属季度（1-4）", "aliases": ["营收季度", "季度", "quarter"],
    },
    {
        "table_name": "monthly_revenue",
        "column_name": "revenue_usd",
        "dtype": "DECIMAL",
        "business_term": "月度总营收（美元）",
        "aliases": ["月营收", "月度收入", "月GMV", "monthly revenue", "月销售额"],
    },
    {
        "table_name": "monthly_revenue",
        "column_name": "orders",
        "dtype": "INT",
        "business_term": "月度订单总数",
        "aliases": ["月订单数", "月度订单", "monthly orders"],
    },
    {
        "table_name": "monthly_revenue",
        "column_name": "avg_order_value",
        "dtype": "DECIMAL",
        "business_term": "月度平均客单价",
        "aliases": ["月均客单价", "月度AOV"],
    },
    {
        "table_name": "monthly_revenue",
        "column_name": "avg_discount_pct",
        "dtype": "DECIMAL",
        "business_term": "月度平均折扣率",
        "aliases": ["月均折扣", "月度折扣率"],
    },
    {
        "table_name": "monthly_revenue",
        "column_name": "return_rate",
        "dtype": "DECIMAL",
        "business_term": "月度退货率",
        "aliases": ["月退货率", "退货比例"],
    },
    {
        "table_name": "monthly_revenue",
        "column_name": "unique_customers",
        "dtype": "INT",
        "business_term": "月度活跃客户数",
        "aliases": ["月活客户", "月活跃用户", "MAU", "monthly active users"],
    },
    {
        "table_name": "monthly_revenue",
        "column_name": "new_customers",
        "dtype": "INT",
        "business_term": "月度新增客户数",
        "aliases": ["新客户", "新增用户", "new users"],
    },

    # ---- product_summary 表 ----
    {
        "table_name": "product_summary", "column_name": "category", "dtype": "VARCHAR",
        "business_term": "商品所属品类", "aliases": ["商品品类", "品类", "类别", "category"],
    },
    {
        "table_name": "product_summary", "column_name": "product_name", "dtype": "VARCHAR",
        "business_term": "商品名称，商品汇总表主键", "aliases": ["商品名称", "产品名称", "商品", "product name"],
    },
    {
        "table_name": "product_summary",
        "column_name": "total_orders",
        "dtype": "INT",
        "business_term": "商品累计订单数",
        "aliases": ["商品订单数", "销量"],
    },
    {
        "table_name": "product_summary",
        "column_name": "total_revenue_usd",
        "dtype": "DECIMAL",
        "business_term": "商品累计营收（美元）",
        "aliases": ["商品营收", "商品销售额"],
    },
    {
        "table_name": "product_summary",
        "column_name": "avg_price",
        "dtype": "DECIMAL",
        "business_term": "商品平均价格",
        "aliases": ["均价", "平均价格"],
    },
    {
        "table_name": "product_summary",
        "column_name": "avg_rating",
        "dtype": "DECIMAL",
        "business_term": "商品平均评分（1-5）",
        "aliases": ["商品评分", "平均评分", "评分"],
    },
    {
        "table_name": "product_summary",
        "column_name": "return_rate",
        "dtype": "DECIMAL",
        "business_term": "商品退货率",
        "aliases": ["退货率", "退货比例"],
    },
    {
        "table_name": "product_summary",
        "column_name": "avg_discount_pct",
        "dtype": "DECIMAL",
        "business_term": "商品平均折扣率",
        "aliases": ["折扣率", "平均折扣"],
    },
    {
        "table_name": "product_summary",
        "column_name": "avg_delivery_days",
        "dtype": "DECIMAL",
        "business_term": "商品平均配送天数",
        "aliases": ["配送天数", "平均物流时间", "delivery days"],
    },
]


# A second business schema hosted in the same MySQL database.  It is kept
# deliberately small so migration and retrieval behavior remain reproducible.
SCHEMA_FIELDS.extend([
    {
        "table_name": "support_agents", "column_name": "agent_id", "dtype": "VARCHAR",
        "business_term": "客服坐席唯一标识", "aliases": ["坐席ID", "客服ID", "agent id"],
        "data_source": "support_ops",
    },
    {
        "table_name": "support_agents", "column_name": "team", "dtype": "VARCHAR",
        "business_term": "客服坐席所属团队", "aliases": ["客服团队", "坐席组", "team"],
        "data_source": "support_ops",
    },
    {
        "table_name": "support_agents", "column_name": "region", "dtype": "VARCHAR",
        "business_term": "客服团队负责区域", "aliases": ["负责区域", "客服区域", "region"],
        "data_source": "support_ops",
    },
    {
        "table_name": "support_tickets", "column_name": "ticket_id", "dtype": "VARCHAR",
        "business_term": "客服工单唯一标识", "aliases": ["工单ID", "工单号", "ticket id"],
        "data_source": "support_ops",
    },
    {
        "table_name": "support_tickets", "column_name": "agent_id", "dtype": "VARCHAR",
        "business_term": "处理工单的客服坐席，关联 support_agents.agent_id",
        "aliases": ["处理坐席", "客服关联", "agent id"], "data_source": "support_ops",
    },
    {
        "table_name": "support_tickets", "column_name": "opened_at", "dtype": "DATETIME",
        "business_term": "工单创建时间", "aliases": ["工单时间", "创建时间", "opened at"],
        "data_source": "support_ops",
    },
    {
        "table_name": "support_tickets", "column_name": "channel", "dtype": "VARCHAR",
        "business_term": "工单进入渠道", "aliases": ["工单渠道", "客服渠道", "channel"],
        "data_source": "support_ops",
    },
    {
        "table_name": "support_tickets", "column_name": "priority", "dtype": "VARCHAR",
        "business_term": "工单优先级", "aliases": ["优先级", "紧急程度", "priority"],
        "data_source": "support_ops",
    },
    {
        "table_name": "support_tickets", "column_name": "status", "dtype": "VARCHAR",
        "business_term": "工单当前状态", "aliases": ["工单状态", "处理状态", "status"],
        "data_source": "support_ops",
    },
    {
        "table_name": "support_tickets", "column_name": "resolution_hours", "dtype": "DECIMAL",
        "business_term": "工单解决耗时（小时）", "aliases": ["解决时长", "处理耗时", "resolution time"],
        "data_source": "support_ops",
    },
    {
        "table_name": "support_tickets", "column_name": "satisfaction_score", "dtype": "DECIMAL",
        "business_term": "工单满意度评分（1-5）", "aliases": ["客服满意度", "满意度评分", "csat"],
        "data_source": "support_ops",
    },
])

for _field in SCHEMA_FIELDS:
    _field.setdefault("data_source", "ai_analytics")
    _field.setdefault("catalog_version", SCHEMA_CATALOG_VERSION)
    _field.setdefault("source_id", "repo:storage/chromadb/schema_metadata.py")
    _field.setdefault("access_scope", "public")
    _field.setdefault("owner_id", "")


SCHEMA_RELATIONSHIPS: List[Dict] = [
    {
        "data_source": "ai_analytics", "version": "1.0.0",
        "left_table": "orders", "left_column": "customer_id",
        "right_table": "customers", "right_column": "customer_id",
        "cardinality": "many-to-one", "source_id": "repo:storage/mysql/schema.sql",
        "access_scope": "public", "owner_id": "",
    },
    {
        "data_source": "support_ops", "version": "1.0.0",
        "left_table": "support_tickets", "left_column": "agent_id",
        "right_table": "support_agents", "right_column": "agent_id",
        "cardinality": "many-to-one", "source_id": "repo:storage/mysql/schema.sql",
        "access_scope": "public", "owner_id": "",
    },
]


# ============================================================
# 表级别元数据
# ============================================================
TABLE_METADATA: List[Dict] = [
    {
        "table_name": "customers",
        "description": "客户信息表：包含客户画像、消费行为、会员等级、流失标记等字段，是RFM分析和流失预测的主要数据源",
        "row_count": "~8000",
        "aliases": ["客户表", "用户表", "customers", "客户信息"],
    },
    {
        "table_name": "orders",
        "description": "订单交易明细表：每行一条订单记录，包含商品、金额、折扣、物流、评价等完整交易链路信息",
        "row_count": "~25000",
        "aliases": ["订单表", "交易表", "orders", "订单明细"],
    },
    {
        "table_name": "monthly_revenue",
        "description": "月度营收汇总表：按年-月聚合的营收、订单、客户等关键指标，用于趋势分析和收入预测",
        "row_count": "75",
        "aliases": ["月度营收表", "月表", "月度汇总", "monthly revenue"],
    },
    {
        "table_name": "product_summary",
        "description": "商品维度汇总表：每行一个商品，包含销量、营收、评分、退货率等商品级别指标",
        "row_count": "140",
        "aliases": ["商品表", "产品表", "商品汇总", "product summary"],
    },
    {
        "table_name": "support_agents", "data_source": "support_ops",
        "description": "客服坐席维表：团队和负责区域，每行一个坐席",
        "row_count": "4", "aliases": ["客服坐席", "客服团队", "agents"],
    },
    {
        "table_name": "support_tickets", "data_source": "support_ops",
        "description": "客服工单事实表：渠道、优先级、状态、解决耗时和满意度",
        "row_count": "12", "aliases": ["客服工单", "服务工单", "tickets"],
    },
]

for _table in TABLE_METADATA:
    _table.setdefault("data_source", "ai_analytics")
    _table.setdefault("catalog_version", SCHEMA_CATALOG_VERSION)
    _table.setdefault("source_id", "repo:storage/chromadb/schema_metadata.py")
    _table.setdefault("access_scope", "public")
    _table.setdefault("owner_id", "")


# ============================================================
# 辅助函数
# ============================================================

def is_accessible(item: Dict, principal_id: str | None = None) -> bool:
    return item.get("access_scope", "public") == "public" or (
        bool(principal_id) and item.get("owner_id") == principal_id
    )


def get_schema_fields(
    data_source: str = "ai_analytics", principal_id: str | None = None
) -> List[Dict]:
    return [
        field for field in SCHEMA_FIELDS
        if field.get("data_source") == data_source and is_accessible(field, principal_id)
    ]


def schema_catalog_fingerprint() -> str:
    payload = {"version": SCHEMA_CATALOG_VERSION, "fields": SCHEMA_FIELDS,
               "relationships": SCHEMA_RELATIONSHIPS}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def get_documents_for_embedding(fields: List[Dict] | None = None) -> List[str]:
    """
    生成所有需要做 embedding 的文本。
    每条记录的文本 = table_name + column_name + business_term + aliases
    """
    docs = []
    for field in fields or SCHEMA_FIELDS:
        text = (
            f"表: {field['table_name']} | "
            f"字段: {field['column_name']} | "
            f"含义: {field['business_term']} | "
            f"同义词: {', '.join(field['aliases'])}"
        )
        docs.append(text)
    return docs


def get_field_by_id(doc_id: int) -> Dict:
    """根据 ChromaDB 返回的 doc_id 获取字段元数据。"""
    return SCHEMA_FIELDS[doc_id]


def get_table_list(data_source: str = "ai_analytics") -> List[str]:
    """返回所有表名。"""
    return [t["table_name"] for t in TABLE_METADATA if t["data_source"] == data_source]


def get_table_description(table_name: str) -> str:
    """返回某张表的描述。"""
    for t in TABLE_METADATA:
        if t["table_name"] == table_name:
            return t["description"]
    return ""
