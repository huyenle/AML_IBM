from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

ALERT_COLS = [
    "rule_id",
    "account_key",
    "alert_date",
    "n_txn",
    "amount_usd",
    "metric",
    "txn_ids",
]

RULES = {
    "R01_STRUCTURING": {
        "name": "Structuring",
        "description": "Several payments on one day just below a reporting threshold",
        "params": {"threshold_usd": 10_000, "band": 0.10, "min_txn": 3},
    },
    "R02_FAN_OUT": {
        "name": "Fan-out",
        "description": "One account pays many different accounts on one day",
        "params": {"min_counterparties": 10},
    },
    "R03_FAN_IN": {
        "name": "Fan-in",
        "description": "One account receives from many different accounts on one day",
        "params": {"min_counterparties": 10},
    },
    "R04_PASS_THROUGH": {
        "name": "Rapid pass-through",
        "description": "Money in and nearly the same amount out on the same day",
        "params": {"min_inflow_usd": 5_000, "min_ratio": 0.9, "max_ratio": 1.1},
    },
}


def _finish(df: DataFrame, rule_id: str) -> DataFrame:
    """Add id, txn_ids is sorted."""
    return (
        df.withColumn("rule_id", F.lit(rule_id))
        .withColumn("txn_ids", F.array_sort("txn_ids"))
        .select(ALERT_COLS)
    )


def rule_structuring(
    tx: DataFrame, threshold_usd: float, band: float, min_txn: int
) -> DataFrame:
    """>= min_txn payments by one sender on one day, each in [threshold*(1-band), threshold)."""
    near = tx.filter(
        (F.col("amount_usd") >= threshold_usd * (1 - band))
        & (F.col("amount_usd") < threshold_usd)
    )
    out = (
        near.groupBy(
            F.col("from_account_key").alias("account_key"),
            F.col("txn_date").alias("alert_date"),
        )
        .agg(
            F.count("*").alias("n_txn"),
            F.sum("amount_usd").alias("amount_usd"),
            F.collect_list("txn_id").alias("txn_ids"),
        )
        .filter(F.col("n_txn") >= min_txn)
        .withColumn("metric", F.col("n_txn").cast("double"))
    )
    return _finish(out, "R01_STRUCTURING")


def _fan(
    tx: DataFrame, side: str, other: str, min_counterparties: int, rule_id: str
) -> DataFrame:
    out = (
        tx.groupBy(
            F.col(side).alias("account_key"), F.col("txn_date").alias("alert_date")
        )
        .agg(
            F.countDistinct(other).alias("n_counterparties"),
            F.count("*").alias("n_txn"),
            F.sum("amount_usd").alias("amount_usd"),
            F.collect_list("txn_id").alias("txn_ids"),
        )
        .filter(F.col("n_counterparties") >= min_counterparties)
        .withColumn("metric", F.col("n_counterparties").cast("double"))
    )
    return _finish(out, rule_id)


def rule_fan_out(tx: DataFrame, min_counterparties: int) -> DataFrame:
    """One sender pays >= min_counterparties distinct receivers on one day."""
    return _fan(
        tx, "from_account_key", "to_account_key", min_counterparties, "R02_FAN_OUT"
    )


def rule_fan_in(tx: DataFrame, min_counterparties: int) -> DataFrame:
    """One receiver gets money from >= min_counterparties distinct senders on one day."""
    return _fan(
        tx, "to_account_key", "from_account_key", min_counterparties, "R03_FAN_IN"
    )


def rule_pass_through(
    tx: DataFrame, min_inflow_usd: float, min_ratio: float, max_ratio: float
) -> DataFrame:
    """Inflow >= min_inflow_usd and outflow/inflow within [min_ratio, max_ratio] on the same day."""
    inflow = tx.groupBy(
        F.col("to_account_key").alias("account_key"),
        F.col("txn_date").alias("alert_date"),
    ).agg(F.sum("amount_usd").alias("in_usd"), F.collect_list("txn_id").alias("in_ids"))
    outflow = tx.groupBy(
        F.col("from_account_key").alias("account_key"),
        F.col("txn_date").alias("alert_date"),
    ).agg(
        F.sum("amount_usd").alias("out_usd"), F.collect_list("txn_id").alias("out_ids")
    )
    out = (
        inflow.join(outflow, ["account_key", "alert_date"])
        .withColumn("metric", F.col("out_usd") / F.col("in_usd"))
        .filter(
            (F.col("in_usd") >= min_inflow_usd)
            & F.col("metric").between(min_ratio, max_ratio)
        )
        .withColumn("txn_ids", F.array_union("in_ids", "out_ids"))
        .withColumn("n_txn", F.size("txn_ids"))
        .withColumn("amount_usd", F.col("in_usd"))
    )
    return _finish(out, "R04_PASS_THROUGH")


RULE_FUNCS = {
    "R01_STRUCTURING": rule_structuring,
    "R02_FAN_OUT": rule_fan_out,
    "R03_FAN_IN": rule_fan_in,
    "R04_PASS_THROUGH": rule_pass_through,
}


def build_alerts(tx: DataFrame, rules: dict = RULES) -> DataFrame:
    """Run every configured rule and stack the results, with a stable alert_id."""
    outputs = [
        RULE_FUNCS[rule_id](tx, **cfg["params"]) for rule_id, cfg in rules.items()
    ]
    alerts = reduce(DataFrame.unionByName, outputs)
    return alerts.withColumn(
        "alert_id", F.xxhash64("rule_id", "account_key", "alert_date")
    )
