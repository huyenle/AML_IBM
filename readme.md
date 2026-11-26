# AML transaction monitoring — demo pipeline

A small end-to-end transaction monitoring pipeline on Databricks, built on the synthetic
[IBM AML dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) (HI-Small).

```
bronze (raw CSV)  →  silver (typed, DQX-checked)  →  gold (star schema)  →  detection rules  →  alerts  →  dashboard
```

| Folder | What lives there |
|---|---|
| `notebooks/` | Databricks notebooks: read tables, call functions, write tables. **No business logic.** |
| `aml/` | The logic: data preparation (`transforms.py`) and detection rules (`rules.py`). Pure functions. |
| `tests/` | Unit tests for everything in `aml/`. |
| `.github/workflows/ci.yml`, `azure-pipelines.yml` | CI: runs the tests on every push. |

---

# How we test PySpark code

This part is for anyone joining the project. It explains how the tests work and how to write new ones.
If you only read one section, read **"Adding a new rule"**.

## 1. The one idea that makes testing possible

**Logic goes in functions in `src/`, not in notebooks.**

A test can import `src/rules.py`; it cannot import a notebook. So notebooks only do input and output:

```python
fact = spark.table("fact_transaction")      # notebook: read
alerts = build_alerts(tx, RULES)            # aml/: logic  ← this is what we test
alerts.write.saveAsTable("stg_alert")       # notebook: write
```

Every function in `aml/` takes DataFrames as arguments and returns a DataFrame. It never reads or writes
a table itself. That way a test can hand it five made-up rows and check the answer. This is also what
Databricks recommends: keep functions and their unit tests outside notebooks.

## 2. Running the tests

On your laptop (needs Python 3.10+ and Java 17, because Spark runs on the JVM):

```bash
pip install -r requirements-dev.txt
pytest -v
```

You should see all tests pass in under a minute. Useful variants:

```bash
pytest tests/test_rules.py              # one file
pytest -k structuring                   # only tests with "structuring" in the name
pytest -x                               # stop at the first failure
```

> ⚠️ **Never add `pyspark` to `pyproject.toml`.** That file drives the Databricks environment, and installing
> PySpark on Databricks serverless breaks the session. Test-only packages go in `requirements-dev.txt`.

### 2b. Running the tests on a Databricks cluster (CLI)

>  **TODO — not tested yet.**  Run the suite on real Databricks compute with the `databricks` CLI. It deploys the
`databricks.yml` asset bundle (which runs `tests/run_on_databricks.py` on the cluster):

```bash
brew tap databricks/tap && brew install databricks   # install the CLI
databricks configure                                 # workspace URL + PAT token

databricks bundle validate           # check the yaml + auth
databricks bundle deploy -t dev      # upload the repo + create the job
databricks bundle run pytest -t dev  # run it; exits non-zero if a test fails
```

First fill the placeholders in `databricks.yml` (`host`, `spark_version`, `node_type_id`).

## 3. Anatomy of a test

Every test has three parts: **arrange** a small input, **act** by calling the function, **assert** the result.

```python
def test_structuring_fires_on_three_payments_just_below_threshold(spark, make_tx):
    # arrange: account 1 pays account 9 three times, just under $10,000
    tx = make_tx([("t1", 1, 9, 9_500), ("t2", 1, 9, 9_600), ("t3", 1, 9, 9_100)])

    # act
    result = rule_structuring(tx, **STRUCT)

    # assert
    expected = spark.createDataFrame(
        [("R01_STRUCTURING", 1, DAY, 3, 28_200.0, 3.0, ["t1", "t2", "t3"])],
        "rule_id string, account_key long, alert_date date, n_txn long, amount_usd double, "
        "metric double, txn_ids array<string>",
    )
    assertDataFrameEqual(result, expected)
```

**Name tests as sentences** that state the expected behaviour. When a test fails in CI, the name alone
should tell you what broke: `test_fan_out_counts_distinct_receivers_not_payments` beats `test_fan_out_2`.

## 4. The shared setup: `tests/conftest.py`

Pytest loads `conftest.py` automatically. It gives every test two **fixtures** — you get one simply by
naming it as a test argument:

| Fixture | What it gives you |
|---|---|
| `spark` | A local Spark session, started **once** for the whole run (starting Spark takes seconds). |
| `make_tx` | A helper that builds a transactions DataFrame from short tuples: `(txn_id, from, to, amount_usd[, date])`. |

The Spark session is tuned for tiny test data:

| Setting | Why |
|---|---|
| `local[1]` | One thread on this machine. No cluster needed; deterministic. |
| `spark.sql.shuffle.partitions = 1` | The default (200) creates 200 near-empty tasks per `groupBy`. This is the biggest speed-up. |
| `spark.ui.enabled = false` | Nobody looks at the UI in CI; avoids port clashes. |
| `spark.sql.session.timeZone = UTC` | Same date/time results on every machine, wherever it runs. |

## 5. Spark pitfalls that make tests flaky

| Pitfall | Fix |
|---|---|
| Row order is not guaranteed | `assertDataFrameEqual` ignores order by default. Don't compare `collect()` lists that depend on order. |
| `collect_list` returns items in any order | Our rules sort the array (`F.array_sort` in `_finish`). Do the same in new rules. |
| Float arithmetic (`0.1 + 0.2`) | `pytest.approx`, or the `rtol`/`atol` options of `assertDataFrameEqual`. |
| `9500` vs `9500.0` | A `double` column rejects Python ints. `make_tx` converts amounts with `float()`. |
| Machine time zone | Fixed to UTC in `conftest.py`. |
| NULLs vanish in `concat_ws` and `xxhash64` | Wrap columns in `F.coalesce(col, F.lit("<NULL>"))` before hashing. |

## 6. Adding a new rule (checklist)

1. **Write the function** in `src/rules.py`. It takes `tx` plus keyword parameters and ends with
   `return _finish(out, "R05_YOUR_RULE")`, so it returns the standard columns in `ALERT_COLS`.
2. **Register it** in `RULE_FUNCS`.
3. **Add its parameters** to `RULES` (name, description, params). Thresholds live here, never in the code.
4. **Write tests** in `tests/test_rules.py`: at least one "fires" test and one "does not fire" test per
   boundary from section 6.
5. **Break it on purpose** once to see your tests go red, then fix it.
6. **Run `pytest`** locally, push, and check CI is green.

You never touch `build_alerts` or the notebooks: they pick up new rules automatically.

## 7. CI - still TODO 

On every push, CI starts a clean machine, installs Python 3.12, Java 17 and `requirements-dev.txt`,
runs `pytest`, and publishes the results. A red run means: don't merge.

- GitHub: `.github/workflows/ci.yml` — results under the **Actions** tab.
- Azure DevOps: `azure-pipelines.yml` — results per test under the pipeline's **Tests** tab.

## Further reading

- Databricks — [Unit testing for notebooks](https://docs.databricks.com/aws/notebooks/testing)
- Apache Spark — [Testing PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/testing_pyspark.html)
- Databricks — [Testing with Databricks Connect](https://docs.databricks.com/aws/en/dev-tools/databricks-connect/python/testing) (running the same tests on real Databricks compute)
