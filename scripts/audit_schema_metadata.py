"""Compare ChromaDB schema metadata with the live MySQL business tables."""

from storage.chromadb.schema_metadata import SCHEMA_FIELDS
from storage.db_adapter import get_available_adapter


def main() -> None:
    adapter = get_available_adapter()
    rows = adapter.execute_sql(
        "SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name "
        "FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = 'bi_she' "
        "AND TABLE_NAME IN ('customers', 'orders', 'monthly_revenue', 'product_summary') "
        "ORDER BY TABLE_NAME, ORDINAL_POSITION"
    )
    actual = {(row["table_name"], row["column_name"]) for row in rows}
    metadata = {
        (field["table_name"], field["column_name"]) for field in SCHEMA_FIELDS
    }

    print("ACTUAL_COLUMNS", len(actual))
    print("METADATA_FIELDS", len(metadata))
    print("MISSING_METADATA", sorted(actual - metadata))
    print("STALE_METADATA", sorted(metadata - actual))

    if actual != metadata:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
