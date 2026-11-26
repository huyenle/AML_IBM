"""Unit tests for the detection rules in src/rules.py.

Each test arranges a tiny transactions DataFrame (via make_tx), runs one rule, and
asserts which (account, day) groups are flagged and the aggregated columns.
"""
from pyspark.testing import assertDataFrameEqual

from conftest import DAY, OTHER_DAY
from src.rules import (
    ALERT_COLS,
    RULES,
    build_alerts,
    rule_fan_in,
    rule_fan_out,
    rule_pass_through,
    rule_structuring,
)

# Rule parameters mirror the production config in RULES, so the tests exercise the
# thresholds the pipeline actually runs with.
STRUCT = RULES["R01_STRUCTURING"]["params"]
FAN_OUT = RULES["R02_FAN_OUT"]["params"]
FAN_IN = RULES["R03_FAN_IN"]["params"]
PASS = RULES["R04_PASS_THROUGH"]["params"]

# Schema of an alert row (the columns _finish emits, see ALERT_COLS).
ALERT_SCHEMA = (
    "rule_id string, account_key string, alert_date date, n_txn long, "
    "amount_usd double, metric double, txn_ids array<string>"
)


def expected(spark, rows):
    return spark.createDataFrame(rows, ALERT_SCHEMA)


# --------------------------------------------------------------------------- #
# R01 structuring
# --------------------------------------------------------------------------- #
def test_structuring_fires_on_three_payments_just_below_threshold(spark, make_tx):
    tx = make_tx([("t1", "A", "X", 9_000), ("t2", "A", "Y", 9_500), ("t3", "A", "Z", 9_999)])

    result = rule_structuring(tx, **STRUCT)

    assertDataFrameEqual(
        result,
        expected(spark, [("R01_STRUCTURING", "A", DAY, 3, 28_499.0, 3.0, ["t1", "t2", "t3"])]),
    )
    assert result.columns == ALERT_COLS


def test_structuring_band_is_half_open_lower_inclusive_upper_exclusive(make_tx):
    # 9000 is in-band (lower inclusive); 10000 is out (threshold, exclusive);
    # 8999.99 is below the band. Only two in-band txns remain -> below min_txn=3.
    tx = make_tx([
        ("lo", "A", "X", 9_000),       # included
        ("hi", "A", "Y", 10_000),      # excluded (== threshold)
        ("below", "A", "Z", 8_999.99), # excluded (below band)
        ("inband", "A", "W", 9_500),   # included
    ])

    assert rule_structuring(tx, **STRUCT).isEmpty()


def test_structuring_does_not_fire_below_min_txn(make_tx):
    tx = make_tx([("t1", "A", "X", 9_000), ("t2", "A", "Y", 9_500)])

    assert rule_structuring(tx, **STRUCT).isEmpty()


def test_structuring_groups_by_account_and_day(make_tx):
    # Same account but spread over two days: neither day reaches min_txn=3.
    tx = make_tx([
        ("t1", "A", "X", 9_000, DAY),
        ("t2", "A", "Y", 9_500, DAY),
        ("t3", "A", "Z", 9_999, OTHER_DAY),
    ])

    assert rule_structuring(tx, **STRUCT).isEmpty()


# --------------------------------------------------------------------------- #
# R02 fan-out / R03 fan-in
# --------------------------------------------------------------------------- #
def test_fan_out_fires_on_enough_distinct_receivers(make_tx):
    rows = [("t1", "A", f"R{i}", 100) for i in range(FAN_OUT["min_counterparties"])]
    tx = make_tx(rows)

    result = rule_fan_out(tx, **FAN_OUT)

    [row] = result.collect()
    assert row["rule_id"] == "R02_FAN_OUT"
    assert row["account_key"] == "A"
    assert row["n_txn"] == FAN_OUT["min_counterparties"]
    assert row["metric"] == float(FAN_OUT["min_counterparties"])


def test_fan_out_counts_distinct_receivers_not_payments(make_tx):
    # Many payments, but all to the same two receivers -> only 2 counterparties.
    tx = make_tx([(f"t{i}", "A", "B" if i % 2 else "C", 100) for i in range(20)])

    assert rule_fan_out(tx, **FAN_OUT).isEmpty()


def test_fan_in_fires_on_enough_distinct_senders(make_tx):
    rows = [("t1", f"S{i}", "B", 100) for i in range(FAN_IN["min_counterparties"])]
    tx = make_tx(rows)

    result = rule_fan_in(tx, **FAN_IN)

    [row] = result.collect()
    assert row["rule_id"] == "R03_FAN_IN"
    assert row["account_key"] == "B"
    assert row["metric"] == float(FAN_IN["min_counterparties"])


def test_fan_in_counts_distinct_senders_not_payments(make_tx):
    tx = make_tx([(f"t{i}", "S1" if i % 2 else "S2", "B", 100) for i in range(20)])

    assert rule_fan_in(tx, **FAN_IN).isEmpty()


# --------------------------------------------------------------------------- #
# R04 pass-through
# --------------------------------------------------------------------------- #
def test_pass_through_fires_on_balanced_in_and_out_same_day(spark, make_tx):
    # B receives 10000 and pays out 9500 the same day; ratio 0.95 in [0.9, 1.1].
    tx = make_tx([("in1", "A", "B", 10_000), ("out1", "B", "C", 9_500)])

    result = rule_pass_through(tx, **PASS)

    assertDataFrameEqual(
        result,
        expected(spark, [("R04_PASS_THROUGH", "B", DAY, 2, 10_000.0, 0.95, ["in1", "out1"])]),
    )


def test_pass_through_requires_both_an_inflow_and_an_outflow(make_tx):
    # B only receives, never pays out -> inner join drops it.
    tx = make_tx([("in1", "A", "B", 10_000)])

    assert rule_pass_through(tx, **PASS).isEmpty()


def test_pass_through_does_not_fire_when_ratio_out_of_band(make_tx):
    # Outflow much smaller than inflow -> ratio 0.2, outside [0.9, 1.1].
    tx = make_tx([("in1", "A", "B", 10_000), ("out1", "B", "C", 2_000)])

    assert rule_pass_through(tx, **PASS).isEmpty()


def test_pass_through_does_not_fire_below_min_inflow(make_tx):
    # Balanced ratio but inflow below the min_inflow_usd floor.
    tx = make_tx([("in1", "A", "B", 1_000), ("out1", "B", "C", 1_000)])

    assert rule_pass_through(tx, **PASS).isEmpty()


# --------------------------------------------------------------------------- #
# build_alerts
# --------------------------------------------------------------------------- #
def test_build_alerts_stacks_every_rule_and_adds_an_alert_id(make_tx):
    tx = make_tx([
        ("s1", "A", "X", 9_000),
        ("s2", "A", "Y", 9_500),
        ("s3", "A", "Z", 9_999),
    ])

    alerts = build_alerts(tx, RULES)

    assert "alert_id" in alerts.columns
    assert set(ALERT_COLS).issubset(set(alerts.columns))
    assert alerts.filter("rule_id = 'R01_STRUCTURING'").count() == 1


def test_build_alerts_id_is_stable_across_runs(make_tx):
    rows = [("s1", "A", "X", 9_000), ("s2", "A", "Y", 9_500), ("s3", "A", "Z", 9_999)]

    first = build_alerts(make_tx(rows), RULES).filter("rule_id = 'R01_STRUCTURING'").collect()
    again = build_alerts(make_tx(rows), RULES).filter("rule_id = 'R01_STRUCTURING'").collect()

    assert first[0]["alert_id"] == again[0]["alert_id"]


def test_build_alerts_on_empty_input_yields_no_alerts(make_tx):
    alerts = build_alerts(make_tx([]), RULES)

    assert alerts.isEmpty()
    assert "alert_id" in alerts.columns
