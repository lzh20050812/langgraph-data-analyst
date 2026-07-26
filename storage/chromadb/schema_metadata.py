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

from typing import List, Dict

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

    # ---- monthly_revenue 表 ----
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
]


# ============================================================
# 辅助函数
# ============================================================

def get_documents_for_embedding() -> List[str]:
    """
    生成所有需要做 embedding 的文本。
    每条记录的文本 = table_name + column_name + business_term + aliases
    """
    docs = []
    for field in SCHEMA_FIELDS:
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


def get_table_list() -> List[str]:
    """返回所有表名。"""
    return [t["table_name"] for t in TABLE_METADATA]


def get_table_description(table_name: str) -> str:
    """返回某张表的描述。"""
    for t in TABLE_METADATA:
        if t["table_name"] == table_name:
            return t["description"]
    return ""
