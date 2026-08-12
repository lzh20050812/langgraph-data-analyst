from agents.scope_guard import detect_unsupported_request


def test_rejects_absent_schema_capabilities():
    for query in ["查询客户身份证号码", "统计信用卡发卡银行", "计算订单利润",
                  "查询仓库库存", "员工工资部门人数", "每月流失客户数"]:
        assert detect_unsupported_request(query) is not None


def test_allows_supported_business_queries():
    for query in ["每月营收趋势", "客户流失率", "订单折扣和退货率", "商品销量排名"]:
        assert detect_unsupported_request(query) is None
