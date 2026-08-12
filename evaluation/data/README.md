# 实验测试数据目录

## 目录结构

```
data/
├── text2sql/          # Text2SQL 实验测试集
│   ├── test_easy.json       # 简单查询（单表单条件）
│   ├── test_medium.json     # 中等查询（多条件/聚合）
│   └── test_hard.json       # 困难查询（多表JOIN/子查询）
├── schema/            # Schema 检索实验测试集
│   └── schema_test.json     # 自然语言查询 + 期望字段标注
├── prediction/        # 预测模型实验数据
│   └── churn_test.csv       # 流失预测测试集特征
├── report/            # 报告质量评估测试集
│   └── report_prompts.json  # 报告生成测试提示
└── system/            # 系统性能测试数据
    └── load_test_queries.json  # 压力测试查询
```

## 数据格式规范

### Text2SQL 测试集 (test_*.json)

```json
[
  {
    "id": 1,
    "difficulty": "easy",
    "query": "查询所有Gold会员的客户ID和消费总额",
    "expected_sql": "SELECT customer_id, total_spend_usd FROM customers WHERE membership_tier = 'Gold'",
    "expected_tables": ["customers"],
    "expected_result_columns": ["customer_id", "total_spend_usd"],
    "solvable": true
  }
]
```

### Schema 检索测试集 (schema_test.json)

```json
[
  {
    "id": 1,
    "query": "中国区GMV最高的前10个客户是谁",
    "difficulty": "easy",
    "expected_fields": ["customers.total_spend_usd", "customers.country", "customers.customer_id"],
    "expected_tables": ["customers"]
  }
]
```

## 注意事项

- 所有 JSON 文件使用 UTF-8 编码
- 测试数据应可复现（固定随机种子）
- 每条测试样本需包含唯一 ID
- 建议包含难度分级（easy / medium / hard）
