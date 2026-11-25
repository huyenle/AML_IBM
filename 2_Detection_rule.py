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

from src.fx import implied_fx
from src.rules import RULES, build_alerts

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS Ref")

# COMMAND ----------

fact = spark.table('gold.fact_transaction_hi_small')
fact.display()

# COMMAND ----------

fact.groupBy("payment_currency").count().display()

# COMMAND ----------

fx = implied_fx(fact)

# COMMAND ----------

fx.write.mode("overwrite").saveAsTable("ref.fx_implied")

# COMMAND ----------

tx_with_fx = (
    fact
    .join(F.broadcast(fx.select(F.col("currency").alias("payment_currency"), "usd_per_unit")),
          "payment_currency", "left")
    .withColumn("amount_usd", F.col("amount_paid").cast("double") * F.col("usd_per_unit"))
)
n_no_fx = tx_with_fx.filter(F.col("amount_usd").isNull()).count()
print(f"Transactions without a USD rate (excluded from amount-based rules): {n_no_fx:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Define functions for rules

# COMMAND ----------

# DBTITLE 1,apply to alerts
alerts = build_alerts(tx_with_fx, RULES)

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