# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
from pyspark.sql import functions as F

# COMMAND ----------

!pip install databricks-labs-dqx==0.8.0

# COMMAND ----------

dbutils.fs.ls("/")

# COMMAND ----------

# MAGIC %md
# MAGIC # Download data

# COMMAND ----------

import urllib.request

url = "https://storage.googleapis.com/kagglesdsdata/datasets/2948142/12411329/HI-Small_Trans.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20261120%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20261120T185831Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=40338dcd45eb1423eb667fd0f7b42a6d8cf8584d229f36f3b39e79fbcbe056fe8dff59fa158f1a8c7b1b4f255a63068e89791794ef8c4d1b4010bb1749926835f639cc2fa914caed4876cc3aa58dcd3694791990f8fdd4a9ec01d1024ec0b7a2511597a9e8bd480f30b81a0be1c47ec980cb2340e0e18fc34cb4297f8f61bcc4c0780463f604123f659ec6b6f6cba9f6f17416bd9995157075674ea12b0325abb0b9f51a256a5b6a04ace471f5b8807db828a7f0244f0248c10c36d9bbf571d48bd9e48f2337baf6aece4cd49453c662576dfec3a99d18978bdc23a3091fde4416019e5d0bee04e235fe4bb83d9f2fb4a5c8632530ce65e517675cb890ab3ebc"

local_path = "/Volumes/workspace/aml/raw/HI-Small_Trans.csv"
# urllib.request.urlretrieve(url, local_path)

df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(local_path))

display(df)


# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # Bronze dataset

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")

# COMMAND ----------

BRONZE_COLS = [
    "Timestamp",
    "From Bank",
    "Account2",
    "To Bank",
    "Account4",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]
raw_trans = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(local_path))

# COMMAND ----------

# DBTITLE 1,validation
assert len(raw_trans.columns) == len(BRONZE_COLS), raw_trans.columns

# COMMAND ----------

bronze_trans = (
    raw_trans.toDF(*BRONZE_COLS)
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .withColumn("_ingested_at", F.current_timestamp())
)
bronze_trans.display()

# COMMAND ----------

# DBTITLE 1,write to table
(bronze_trans.write
      .option("delta.columnMapping.mode", "name")
      .mode("overwrite")
      .saveAsTable("bronze.transaction_HI_Small"))

# COMMAND ----------

bronze_trans.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver
# MAGIC - Clean up data

# COMMAND ----------

b = spark.table("bronze.transaction_hi_small")

# COMMAND ----------

from pyspark.sql.types import DecimalType, IntegerType

casted_b = b.select(
    F.try_to_timestamp(F.col("Timestamp"), F.lit("yyyy/MM/dd HH:mm")).alias("txn_ts"),
    F.trim("From Bank").alias("from_bank_id"),
    F.upper(F.trim("Account4")).alias("from_account"),
    F.trim("To Bank").alias("to_bank_id"),
    F.upper(F.trim("Account2")).alias("to_account"),
    F.col("Amount Paid").try_cast(DecimalType(38, 2)).alias("amount_paid"),
    F.trim("Payment Currency").alias("payment_currency"),
    F.col("Amount Received").try_cast(DecimalType(38, 2)).alias("amount_received"),
    F.trim("Receiving Currency").alias("receiving_currency"),
    F.trim("Payment Format").alias("payment_format"),
    F.col("Is Laundering").try_cast(IntegerType()).alias("is_laundering"),
    "_source_file",
    "_ingested_at",
).withColumn("txn_date", F.to_date("txn_ts"))

# COMMAND ----------

# casted_b.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in casted_b.columns]).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define validation rules

# COMMAND ----------

PAYMENT_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
KNOWN_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "NZD", "SEK", "DKK", "NOK", "CNY", "HKD", "SGD", "MXN", "BRL", "INR", "PLN", "RUB", "TRY", "ZAR", "KRW", "THB", "IDR", "PHP", "MYR", "TWD", "CZK", "HUF", "PLN", "ISK", "RON", "HRK", "BGN", "CZK", "DKK", "HUF", "ISK", "NOK", "PLN", "RON", "SEK", "TRY", "ZAR"]
NON_NULL_COLS = [
    "txn_ts",
    "from_bank_id",
    "from_account",
    "to_bank_id",
    "to_account",
    "amount_paid",
    "payment_currency",
    "amount_received",
    "receiving_currency"]


# COMMAND ----------

# define dictionary for DQX
def err(name, function, **arguments):
    return {
        "name": name,
        "criticality": "error",
        "check": {"function": function, "arguments": arguments},
    }


def warn(name, function, **arguments):
    return {
        "name": name,
        "criticality": "warn",
        "check": {"function": function, "arguments": arguments},
    }

# COMMAND ----------

# DBTITLE 1,list DQ rules
DQ_Checks = [
    *[err(name=f"{c}_not_null", function="is_not_null") for c in NON_NULL_COLS],
    err(
        name="non negative amount receved",
        function="is_positive",
        column="amount_received",
    ),
    err(name="non negative amount paid", function="is_positive", column="amount_paid"),
    err(
        name="valid laundering flag",
        function="is_in",
        column="is_laundering",
        values=[0, 1],
    ),
    warn(
        name="if_same_currentcy_then_same_amount",
        function="sql_expression",
        expression="payment_currency <> receiving_currency OR amount_paid == amount_received",
        msg = "same currency but the amounts differ"
    ),
    warn(
        name="valid payment format",
        function="is_in",
        column="payment_format",
        values=PAYMENT_FORMATS,
    ),
    warn(
        name="new payment currency",
        function="is_in",
        column="payment_currency",
        values=KNOWN_CURRENCIES,
        msg = "new currency"
    ),
    warn(
        name="new receiving currency",
        function="is_in",
        column="receiving_currency",
        values=KNOWN_CURRENCIES,
        msg = "new currency"
    )
]

# COMMAND ----------

from databricks.labs.dqx.engine import DQEngine
from databricks.sdk import WorkspaceClient

dq = DQEngine(WorkspaceClient())
checked = dq.run_checks(
    casted_b,
    DQ_Checks,
    "silver",
    quarantine="quarantine",
    quarantine_schema="default",
    quarantine_table="quarantine",
)
checked.display()

# COMMAND ----------

is_valid = (
    F.col("txn_ts").isNotNull()
    & F.col("from_account").isNotNull() & F.col("to_account").isNotNull()
    & F.col("amount_paid").isNotNull() & (F.col("amount_paid") >= 0)
    & F.col("amount_received").isNotNull() & (F.col("amount_received") >= 0)
    & F.col("is_laundering").isin(0, 1)
)
valid_b = casted_b.filter(is_valid)
quarantine = casted_b.filter(~is_valid|is_valid.isNull())

# COMMAND ----------

quarantine.display()

# COMMAND ----------

# DBTITLE 1,save table
valid_b.display()

# COMMAND ----------

from pyspark.sql.window import Window

row_hash = F.sha2(
    F.concat_ws(
        " ",
        *[
            F.coalesce(F.col(c).cast("string"), F.lit("<NULL>"))
            for c in [
                "txn_ts",
                "from_bank_id",
                "from_account",
                "to_bank_id",
                "to_account",
                "amount_paid",
                "payment_currency",
                "amount_received",
                "receiving_currency",
                "payment_format",
                "is_laundering",
            ]
        ],
    ),
    256,
)
w = Window.partitionBy("row_hash").orderBy(F.lit("row_hash"))
silver = (
    valid_b.withColumn("row_hash", row_hash)
    .withColumn("dup_seq", F.row_number().over(w))
    .withColumn(
        "txn_id", F.concat_ws("-", F.substring("row_hash", 1, 16), F.col("dup_seq"))
    )
    .withColumn(
        "is_self_loop",
        (F.col("from_bank_id") == F.col("to_bank_id"))
        & (F.col("from_account") == F.col("to_account")),
    )
    .drop("row_hash")
)

# COMMAND ----------

silver.filter(F.col("dup_seq")>1).display()

# COMMAND ----------

silver.filter(
    F.col("txn_id").isin("fd43324c6d42389f-2", "fd43324c6d42389f-1")
).display()

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
silver.write.mode("overwrite").saveAsTable("silver.transaction_hi_small")
quarantine.write.mode("overwrite").saveAsTable("silver.transaction_hi_small_quarantine")

# COMMAND ----------

# MAGIC %md
# MAGIC # Gold

# COMMAND ----------

# MAGIC %md
# MAGIC grain: 
# MAGIC - one row per transaction for transactions
# MAGIC - One bankID and account for account 
# MAGIC

# COMMAND ----------

# DBTITLE 1,accounts
s = spark.table("silver.transaction_hi_small")


def account_key(bank, acct):
    return F.substring(F.sha2(F.concat_ws(" ", bank, acct), 256), 0, 16)


accounts = (
    s.select(
        F.col("from_bank_id").alias("bank_id"),
        F.col("from_account").alias("account_number"),
    )
    .union(
        s.select(
            F.col("to_bank_id").alias("bank_id"),
            F.col("to_account").alias("account_number"),
        )
    )
    .distinct()
)

# COMMAND ----------

accounts = accounts.withColumn("account_key", account_key("bank_id", "account_number"))
accounts.display()


# COMMAND ----------

# DBTITLE 1,Extract Unique Bank IDs with Hashed Keys
banks = (
    s.select(F.col("from_bank_id").alias("bank_id"))
    .union(s.select(F.col("to_bank_id")))
    .distinct()
)
banks = banks.select(
    F.substring(F.sha2(F.col("bank_id"), 256), 0, 16).alias("bank_key"),
    F.col("bank_id"),
)
banks.display()

# COMMAND ----------

# DBTITLE 1,generate date dimension with weekday and weekend flags
bounds = s.agg(F.min("txn_date").alias("lo"), F.max("txn_date").alias("hi"))

dates = (
    bounds.select(F.explode(F.sequence("lo", "hi")).alias("date"))
    .select(
        F.date_format("date", "yyyyMMdd").cast("int").alias("date_key"),
        "date",
        F.dayofweek("date").alias("day_of_week"),        # 1 = Sunday
        F.date_format("date", "E").alias("day_name"),
        F.weekofyear("date").alias("iso_week"),
        F.month("date").alias("month"),
        F.year("date").alias("year"),
        F.dayofweek("date").isin(1, 7).alias("is_weekend"),
    )
)

dates.display()

# COMMAND ----------

fact = s.select(
    "txn_id",
    F.date_format("txn_date", "yyyyMMdd").cast("int").alias("date_key"),
    "txn_ts",
    "txn_date",
    account_key("from_bank_id", "from_account").alias("from_account_key"),
    account_key("to_bank_id", "to_account").alias("to_account_key"),
    F.substring(F.sha2(F.col("from_bank_id"), 256), 0, 16).alias("from_bank_key"),
    F.substring(F.sha2(F.col("to_bank_id"), 256), 0, 16).alias("to_bank_key"),
    "amount_paid",
    "payment_currency",
    "amount_received",
    "receiving_currency",
    "payment_format",
    "is_laundering",
    "is_self_loop",
)

fact.display()

# COMMAND ----------

# DBTITLE 1,write data
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")

accounts.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("gold.dim_account")
banks.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("gold.dim_bank")
dates.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("gold.dim_date")
fact.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("gold.fact_transaction_hi_small")


# COMMAND ----------

# MAGIC %md
# MAGIC # Check count

# COMMAND ----------

n_bronze = spark.table("bronze.transaction_hi_small").count()
n_silver = spark.table("silver.transaction_hi_small").count()
n_quar = spark.table("silver.transaction_hi_small_quarantine").count()
f = spark.table("gold.fact_transaction_hi_small")
n_fact = f.count()

# COMMAND ----------

checks = {
    "bronze = silver + quarantine": n_bronze == n_silver + n_quar,
    "fact = silver": n_fact == n_silver,
    "txn_id unique": f.select("txn_id").distinct().count() == n_fact,
    "no orphan sender accounts": f.join(
        spark.table("gold.dim_account"),
        f.from_account_key == F.col("account_key"),
        "left_anti",
    ).count()
    == 0,
    "no orphan receiver accounts": f.join(
        spark.table("gold.dim_account"),
        f.to_account_key == F.col("account_key"),
        "left_anti",
    ).count()
    == 0,
    "no orphan dates": f.join(
        spark.table("gold.dim_date"), "date_key", "left_anti"
    ).count()
    == 0,
    "laundering count preserved": (
        spark.table("silver.transaction_hi_small").filter("is_laundering = 1").count()
        == f.filter("is_laundering = 1").count()
    ),
}

# COMMAND ----------

print(f"bronze {n_bronze:,} | silver {n_silver:,} | quarantine {n_quar:,} | fact {n_fact:,}")
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
assert all(checks.values()), "Reconciliation failed"