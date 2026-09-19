import pytest

from storage.mysql.provision_reader import _quote_mysql_string, _safe_identifier


def test_mysql_account_identifiers_accept_only_a_safe_subset():
    assert _safe_identifier("analytics_reader", "user") == "analytics_reader"
    with pytest.raises(ValueError):
        _safe_identifier("reader'@'localhost", "user")


def test_mysql_password_literal_is_escaped():
    assert _quote_mysql_string("a'b\\c") == "a''b\\\\c"
