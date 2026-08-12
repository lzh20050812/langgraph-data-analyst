import pytest

from storage.mysql import client


def test_rejects_write_cte_before_opening_connection():
    with pytest.raises(ValueError, match="只读"):
        client.execute_sql("WITH x AS (SELECT 1) DELETE FROM customers")


def test_rejects_multiple_statements_before_opening_connection():
    with pytest.raises(ValueError, match="只读"):
        client.execute_sql("SELECT 1; DROP TABLE customers")
