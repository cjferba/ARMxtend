# -*- coding: utf-8 -*-
"""Shared fixtures: a real SparkContext (local[2]) for the integration tests."""
import os
import subprocess
import sys

import pytest


def java_available():
    # shutil.which("java") is not enough: macOS ships a /usr/bin/java that
    # exists even when no JDK is installed and fails when run.
    java = os.path.join(os.environ["JAVA_HOME"], "bin", "java") if os.environ.get("JAVA_HOME") else "java"
    try:
        return subprocess.run([java, "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


@pytest.fixture(scope="session")
def real_sc():
    """Real SparkContext in local[2] mode; the test is skipped if there is no pyspark or no JDK >= 17 (set JAVA_HOME)."""
    pyspark = pytest.importorskip("pyspark")
    if not java_available():
        pytest.skip("No JDK available (set JAVA_HOME)")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    conf = pyspark.SparkConf().setMaster("local[2]").setAppName("armxtend-tests").set("spark.ui.enabled", "false")
    context = pyspark.SparkContext(conf=conf)
    context.setLogLevel("ERROR")
    yield context
    context.stop()
