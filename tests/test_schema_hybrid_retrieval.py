from agents.schema_retrieval import (
    fuse_schema_candidates,
    lexical_schema_search,
)


def test_lexical_retrieval_maps_explicit_business_term_to_schema_field():
    results = lexical_schema_search("按会员等级统计客户数量", top_k=5)

    assert (results[0]["table_name"], results[0]["column_name"]) == (
        "customers",
        "membership_tier",
    )


def test_weighted_fusion_combines_sources_and_deduplicates_fields():
    vector = [
        {
            "rank": 1,
            "table_name": "customers",
            "column_name": "total_spend_usd",
            "business_term": "客户累计消费总额",
            "dtype": "DECIMAL",
            "score": 0.91,
        },
        {
            "rank": 8,
            "table_name": "customers",
            "column_name": "membership_tier",
            "business_term": "会员等级",
            "dtype": "VARCHAR",
            "score": 0.72,
        },
    ]
    lexical = lexical_schema_search("会员等级", top_k=5)

    fused = fuse_schema_candidates(vector, lexical, top_k=5)
    keys = [(item["table_name"], item["column_name"]) for item in fused]

    assert keys.count(("customers", "membership_tier")) == 1
    membership = next(
        item for item in fused if item["column_name"] == "membership_tier"
    )
    assert membership["retrieval_sources"] == ["lexical", "vector"]
    assert membership["lexical_rank"] == 1
