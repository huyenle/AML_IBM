"""Preparation steps shared by all detection rules. Pure functions: DataFrame in, DataFrame out."""
from pyspark.sql import DataFrame, SparkSession, functions as F

USD = "US Dollar"


def implied_fx(fact: DataFrame) -> DataFrame:
    """Estimate one USD rate per payment currency from cross-currency transactions.

    Rate per transaction = amount_received / amount_paid (received in USD).
    The median is used so that odd transactions (rounding, fees, errors) don't move it.
    Returns columns: currency, usd_per_unit, n_obs. US Dollar is always included at 1.0.
    """
    spark = SparkSession.getActiveSession()
    observed = (
        fact.filter((F.col("receiving_currency") == USD)
                    & (F.col("payment_currency") != USD)
                    & (F.col("amount_paid") > 0))
        .groupBy(F.col("payment_currency").alias("currency"))
        .agg(
            F.percentile_approx(
                F.col("amount_received").cast("double") / F.col("amount_paid").cast("double"), 0.5
            ).alias("usd_per_unit"),
            F.count("*").alias("n_obs"),
        )
    )
    usd = spark.createDataFrame([(USD, 1.0, None)], "currency string, usd_per_unit double, n_obs long")
    return observed.unionByName(usd)

