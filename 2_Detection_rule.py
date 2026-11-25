# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Rules:

# COMMAND ----------

import json
from functools import reduce
from pyspark.sql import DataFrame, functions as F

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS Ref")

# COMMAND ----------

fact = spark.table('gold.fact_transaction_hi_small')
fact.display()

# COMMAND ----------

fact.groupBy("payment_currency").count().display()

# COMMAND ----------

fact.groupBy("receiving_currency").count().display()

# COMMAND ----------

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

# COMMAND ----------

# DBTITLE 1,implied fx by median of fx
implied_fx = (
    fact.filter(
        "receiving_currency = 'US Dollar' AND payment_currency <> 'US Dollar' AND amount_paid > 0"
    )
    .groupBy(F.col("payment_currency").alias("currency"))
    .agg(
        F.percentile_approx(
            F.col("amount_received").cast("double")
            / F.col("amount_paid").cast("double"),
            0.5,
        ).alias("usd_per_unit"),
        F.count("*").alias("n_obs"),
    )
    .unionByName(
        spark.createDataFrame(
            [("US Dollar", 1.0, None)],
            "currency string, usd_per_unit double, n_obs long",
        )
    )
)

# COMMAND ----------

display(implied_fx.orderBy("currency"))

# COMMAND ----------

implied_fx.write.mode("overwrite").saveAsTable("ref.fx_implied")

# COMMAND ----------

tx_with_fx = (
    fact
    .join(F.broadcast(implied_fx.select(F.col("currency").alias("payment_currency"), "usd_per_unit")),
          "payment_currency", "left")
    .withColumn("amount_usd", F.col("amount_paid").cast("double") * F.col("usd_per_unit"))
)
n_no_fx = tx_with_fx.filter(F.col("amount_usd").isNull()).count()
print(f"Transactions without a USD rate (excluded from amount-based rules): {n_no_fx:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Define functions for rules

# COMMAND ----------

ALERT_COLS = ["rule_id", "account_key", "alert_date", "n_txn", "amount_usd", "metric", "txn_ids"]

# COMMAND ----------

def _finish(df: DataFrame, rule_id: str) -> DataFrame:
    return df.withColumn("rule_id", F.lit(rule_id)).select(ALERT_COLS)

# COMMAND ----------

def rule_structuring(
    tx: DataFrame, threshold_usd: float, band: float, min_txn: int
) -> DataFrame:
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
        .withColumn(
            "metric", F.col("n_txn").cast("double")
        )  # metric = payments in the band
    )
    return _finish(out, "R01_STRUCTURING")

# COMMAND ----------

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
        .withColumn(
            "metric", F.col("n_counterparties").cast("double")
        )  # metric = distinct counterparties
    )
    return _finish(out, rule_id)

# COMMAND ----------

def rule_fan_out(tx: DataFrame, min_counterparties: int) -> DataFrame:
    return _fan(tx, "from_account_key", "to_account_key", min_counterparties, "R02_FAN_OUT")


def rule_fan_in(tx: DataFrame, min_counterparties: int) -> DataFrame:
    return _fan(tx, "to_account_key", "from_account_key", min_counterparties, "R03_FAN_IN")

# COMMAND ----------

def rule_pass_through(
    tx: DataFrame, min_inflow_usd: float, min_ratio: float, max_ratio: float
) -> DataFrame:
    """
    this only account for the date though, 24 rolling windows can be better in this case 
    """
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
        .withColumn("metric", F.col("out_usd") / F.col("in_usd"))  # metric = out ÷ in
        .filter(
            (F.col("in_usd") >= min_inflow_usd)
            & F.col("metric").between(min_ratio, max_ratio)
        )
        .withColumn("txn_ids", F.array_union("in_ids", "out_ids"))
        .withColumn("n_txn", F.size("txn_ids"))
        .withColumn("amount_usd", F.col("in_usd"))
    )
    return _finish(out, "R04_PASS_THROUGH")

# COMMAND ----------

# DBTITLE 1,define function for each rule
RULE_FUNCS = {
    "R01_STRUCTURING": rule_structuring,
    "R02_FAN_OUT": rule_fan_out,
    "R03_FAN_IN": rule_fan_in,
    "R04_PASS_THROUGH": rule_pass_through,
}

# COMMAND ----------

# DBTITLE 1,apply to alerts
alerts = reduce(
    DataFrame.unionByName,
    [RULE_FUNCS[rid](tx_with_fx, **cfg["params"]) for rid, cfg in RULES.items()],
).withColumn("alert_id", F.xxhash64("rule_id", "account_key", "alert_date"))

bridge = alerts.select("alert_id", F.explode("txn_ids").alias("txn_id")).distinct()
bridge.write.mode("overwrite").saveAsTable("gold.bridge_alert_transaction")

# COMMAND ----------

bridge.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Alert in the transaction table 

# COMMAND ----------

truth = (
    spark.table("gold.bridge_alert_transaction")
    .join(fact.select("txn_id", "is_laundering"), "txn_id")
    .groupBy("alert_id").agg(F.max("is_laundering").alias("is_true_positive"))
)

fact_alert = (
    alerts.drop("txn_ids")
    .join(truth, "alert_id", "left")
    .withColumn("date_key", F.date_format("alert_date", "yyyyMMdd").cast("int"))
    .select("alert_id", "rule_id", "account_key", "date_key", "alert_date",
            "n_txn", "amount_usd", "metric", "is_true_positive")
)
fact_alert.write.mode("overwrite").saveAsTable("gold.fact_alert")

# COMMAND ----------

dim_rule = spark.createDataFrame(
    [(rid, c["name"], c["description"], json.dumps(c["params"])) for rid, c in RULES.items()],
    "rule_id string, rule_name string, description string, params_json string",
)
dim_rule.write.mode("overwrite").saveAsTable("gold.dim_rule")

# COMMAND ----------

dim_rule.display()