# tests/run_on_databricks.py — entry point executed on the Databricks cluster.
#
# The cluster already owns a SparkSession, so conftest.py's getOrCreate() binds to
# it (the local[1] master is ignored) and Java is present — no JAVA_GATEWAY_EXITED.
import os
import sys

import pytest

# When run from a Repo/Git folder, the repo root is the parent of tests/.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)  # rootdir == repo root, so pytest.ini's `pythonpath = .` applies
sys.dont_write_bytecode = True

retcode = pytest.main(["tests", "-v", "-p", "no:cacheprovider"])
if retcode != 0:
    raise SystemExit(f"pytest failed with exit code {retcode}")
print("pytest passed")
