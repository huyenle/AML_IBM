"""Shared pytest fixtures for the PySpark unit tests.

Two fixtures are offered to every test (get one by naming it as a test argument):
    spark    - a local Spark session, started once for the whole run.
    make_tx  - builds a transactions DataFrame from short tuples.
"""
from datetime import date

import pytest
from pyspark.sql import SparkSession

# Default day used when a test does not care about the date. All rules group by
# (account, day), so most tests only need "the same day" vs "a different day".
DAY = date(2024, 1, 15)
OTHER_DAY = date(2024, 1, 16)

# Schema of the transaction fact as consumed by the rules (see src/rules.py).
# account_key is a string (sha2 hash substring in the real pipeline).
TX_SCHEMA = (
    "txn_id string, txn_date date, "
    "from_account_key string, to_account_key string, "
    "amount_usd double"
)


@pytest.fixture(scope="session")
def spark():
    """A local single-threaded SparkSession, tuned for tiny test data."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("aml-ibm-tests")
        # Default is 200 near-empty shuffle tasks per groupBy; 1 is much faster here.
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        # Same date/time results on every machine, wherever the tests run.
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def make_tx(spark):
    """Build a transactions DataFrame from (txn_id, from, to, amount_usd[, date]) tuples.

    Account keys are coerced to str and amounts to float, so tests can pass plain
    ints/strings. The date defaults to DAY when omitted.
    """
    def _make(rows):
        norm = []
        for r in rows:
            txn_id, frm, to, amount = r[0], r[1], r[2], r[3]
            day = r[4] if len(r) > 4 else DAY
            norm.append((str(txn_id), day, str(frm), str(to), float(amount)))
        return spark.createDataFrame(norm, TX_SCHEMA)

    return _make
