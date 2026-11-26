"""Unit tests for implied_fx in src/fx.py."""
import pytest

from src.fx import USD, implied_fx

# Schema of the raw fact as consumed by implied_fx (see src/fx.py).
FACT_SCHEMA = (
    "payment_currency string, receiving_currency string, "
    "amount_paid double, amount_received double"
)


def make_fact(spark, rows):
    """Build a fact DataFrame from (pay_ccy, recv_ccy, amount_paid, amount_received) tuples."""
    norm = [(p, r, float(ap), float(ar)) for (p, r, ap, ar) in rows]
    return spark.createDataFrame(norm, FACT_SCHEMA)


def rates(df):
    """Collect to a map: currency -> (usd_per_unit, n_obs)."""
    return {r["currency"]: (r["usd_per_unit"], r["n_obs"]) for r in df.collect()}


def test_usd_is_always_present_at_rate_one(spark):
    out = rates(implied_fx(make_fact(spark, [])))

    assert out[USD] == (1.0, None)


def test_single_observation_gives_received_over_paid(spark):
    # 100 EUR paid -> 110 USD received, so 1.1 USD per EUR.
    out = rates(implied_fx(make_fact(spark, [("EUR", USD, 100, 110)])))

    assert out["EUR"][0] == pytest.approx(1.1)
    assert out["EUR"][1] == 1


def test_rate_is_the_median_so_an_outlier_does_not_move_it(spark):
    # Rates 1.0, 1.1, 50.0 -> median 1.1, the outlier is ignored.
    rows = [
        ("EUR", USD, 100, 100),    # 1.0
        ("EUR", USD, 100, 110),    # 1.1
        ("EUR", USD, 100, 5_000),  # 50.0 outlier
    ]
    out = rates(implied_fx(make_fact(spark, rows)))

    assert out["EUR"][0] == pytest.approx(1.1)
    assert out["EUR"][1] == 3


def test_only_cross_currency_payments_into_usd_are_counted(spark):
    rows = [
        ("EUR", USD, 100, 110),   # valid -> 1.1
        ("EUR", "GBP", 100, 90),  # ignored: receiving currency is not USD
        (USD, USD, 100, 100),     # ignored: payment currency is USD (not cross-currency)
        ("EUR", USD, 0, 50),      # ignored: amount_paid == 0 (would divide by zero)
    ]
    out = rates(implied_fx(make_fact(spark, rows)))

    assert out["EUR"][0] == pytest.approx(1.1)
    assert out["EUR"][1] == 1  # only the one valid observation


def test_a_rate_is_produced_per_payment_currency(spark):
    rows = [("EUR", USD, 100, 110), ("GBP", USD, 100, 125)]
    out = rates(implied_fx(make_fact(spark, rows)))

    assert out["EUR"][0] == pytest.approx(1.1)
    assert out["GBP"][0] == pytest.approx(1.25)
    assert out[USD] == (1.0, None)
