# -*- coding: utf-8 -*-
"""
Scalability benchmark of the Big Data algorithms of ARMxtend on a REAL Spark in local mode.

It measures, on synthetic data (items with Zipf-like popularity, baskets of ~10 items) read from
text files with ``sc.textFile`` (as in real use: the RDD is re-read and re-parsed unless
it is cached):

* ``data``  -- time versus the number of transactions N (at fixed cores); the exponent
               alpha of t ~ N^alpha is fitted (alpha ~ 1 = linear scaling) and the peak RSS memory
               of the driver is recorded (relevant for DECLAT, which brings the TID-lists to the driver).
* ``cores`` -- speedup S = T1/Tn and efficiency E = S/n versus the number of cores (at fixed N).
* ``items`` -- time versus the number of distinct items (at fixed N).
* ``cache`` -- effect of caching the input RDD.
* ``sequences`` -- PrefixSpan over sequences: ``prefixspan`` (single machine, Python) versus
               ``prefixspan_spark`` (MLlib) with the input as a list, as an RDD and as a DataFrame.

Usage (from the repository root, with a JDK 17+ in JAVA_HOME)::

    python benchmarks/spark_scalability.py data cores items cache sequences --output results.json

It is a benchmark on ONE machine (``local[n]``): it measures how the work is distributed among cores and how
the cost grows with the data, not the network or the distribution among the nodes of a real cluster.
"""
import argparse
import json
import os
import resource
import sys
import tempfile
import time
import zlib

import numpy as np

from ARMxtend.FIM.apriori import DApriori, DAprioriTID
from ARMxtend.FIM.BD_FARE import FuzzyDAprioriTID
from ARMxtend.FIM.Eclat import DECLAT, FuzzyDECLAT
from ARMxtend.FIM.sequential import prefixspan, prefixspan_spark

CRISP = {"DApriori": DApriori, "DAprioriTID": DAprioriTID, "DECLAT": DECLAT}
FUZZY = {"FuzzyDAprioriTID": FuzzyDAprioriTID, "FuzzyDECLAT": FuzzyDECLAT}


def generate(path, n_transactions, n_items, mean_basket=10, seed=0, shards=8):
    """Writes `shards` text files with transactions 'i3,i17,...' (items with Zipf-like popularity)."""
    rng = np.random.RandomState(seed)
    popularity = 1.0 / np.arange(1, n_items + 1) ** 0.9
    probabilities = np.minimum(0.7, popularity * mean_basket / popularity.sum())
    names = np.array(["i{}".format(i) for i in range(n_items)])
    os.makedirs(path, exist_ok=True)
    per_shard = n_transactions // shards
    for shard in range(shards):
        with open(os.path.join(path, "part-{:03d}.txt".format(shard)), "w") as handle:
            for _ in range(per_shard):
                chosen = names[rng.random_sample(n_items) < probabilities]
                handle.write(",".join(chosen) if len(chosen) else "i0")
                handle.write("\n")
    return per_shard * shards


def parse_fuzzy(line):
    """'i3,i17' -> [('i3', degree), ('i17', degree)] con grados deterministas en {0.3, ..., 1.0}."""
    return [(item, 0.3 + 0.7 * ((zlib.crc32(item.encode()) + len(line)) % 8) / 7.0) for item in line.split(",")]


def generate_sequences(path, n_sequences, n_items, mean_elements=6, seed=0, shards=8):
    """
    Writes `shards` files with one sequence per line: elements separated by ';' and items of an element by ','.
    Items with Zipf-like popularity; each sequence has about `mean_elements` elements of 1 or 2 items.
    """
    rng = np.random.RandomState(seed)
    popularity = 1.0 / np.arange(1, n_items + 1) ** 0.9
    probabilities = popularity / popularity.sum()
    names = np.array(["s{}".format(i) for i in range(n_items)])
    os.makedirs(path, exist_ok=True)
    per_shard = n_sequences // shards
    for shard in range(shards):
        with open(os.path.join(path, "part-{:03d}.txt".format(shard)), "w") as handle:
            for _ in range(per_shard):
                elements = []
                for _ in range(max(1, rng.poisson(mean_elements))):
                    size = 1 + int(rng.rand() < 0.3)
                    elements.append(",".join(sorted(set(rng.choice(names, size=size, p=probabilities)))))
                handle.write(";".join(elements) + "\n")
    return per_shard * shards


def parse_sequence(line):
    """'s1,s3;s2' -> [['s1', 's3'], ['s2']]"""
    return [element.split(",") for element in line.split(";")]


def timed(function):
    started = time.perf_counter()
    result = function()
    return time.perf_counter() - started, result


def driver_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def new_context(cores, driver_memory="6g"):
    from pyspark import SparkConf, SparkContext

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    conf = (SparkConf().setMaster("local[{}]".format(cores)).setAppName("armxtend-bench")
            .set("spark.ui.enabled", "false").set("spark.driver.memory", driver_memory)
            .set("spark.sql.shuffle.partitions", str(cores * 2)))
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    return sc


def run_algorithm(sc, name, rdd, min_supp, num_alpha=5):
    if name in CRISP:
        result = CRISP[name].run(sc, rdd, min_supp)
    else:
        result = FUZZY[name].run(sc, rdd, min_supp, num_alpha)
    return len(result)


def input_rdd(sc, data_dir, fuzzy, cache, partitions):
    rdd = sc.textFile(data_dir, minPartitions=partitions)
    if fuzzy:
        rdd = rdd.map(parse_fuzzy)
    if cache:
        rdd.cache()
        rdd.count()                 # materializes the cache outside the measured time
    return rdd


def measure(sc, name, data_dir, min_supp, cache=False, repeats=2, partitions=8):
    fuzzy = name in FUZZY
    best, count = None, None
    for _ in range(repeats):
        rdd = input_rdd(sc, data_dir, fuzzy, cache, partitions)
        seconds, count = timed(lambda: run_algorithm(sc, name, rdd, min_supp))
        rdd.unpersist()
        best = seconds if best is None else min(best, seconds)
    return best, count


def experiment_data(args, tmp):
    sizes = [int(x) for x in args.sizes.split(",")]
    sc = new_context(args.cores)
    rows = []
    try:
        warm = os.path.join(tmp, "warmup")
        generate(warm, 8000, args.items)
        for name in list(CRISP) + list(FUZZY):                 # calentamiento (JIT, workers de Python)
            measure(sc, name, warm, args.min_supp, repeats=1)
        for n in sizes:
            data_dir = os.path.join(tmp, "data_{}".format(n))
            real_n = generate(data_dir, n, args.items)
            for name in list(CRISP) + list(FUZZY):
                if name in FUZZY and n > args.fuzzy_max:
                    continue
                before = driver_rss_mb()
                seconds, count = measure(sc, name, data_dir, args.min_supp)
                rows.append({"experiment": "data", "algorithm": name, "transactions": real_n, "seconds": round(seconds, 3),
                             "itemsets": count, "driver_rss_mb": round(driver_rss_mb(), 1),
                             "driver_rss_growth_mb": round(driver_rss_mb() - before, 1)})
                print(rows[-1], flush=True)
    finally:
        sc.stop()
    return rows


def experiment_cores(args, tmp):
    data_dir = os.path.join(tmp, "cores_data")
    n = generate(data_dir, args.cores_n, args.items)
    rows = []
    for cores in [int(x) for x in args.core_list.split(",")]:
        sc = new_context(cores)
        try:
            measure(sc, "DApriori", data_dir, args.min_supp, repeats=1)                     # calentamiento
            for name in list(CRISP) + ["FuzzyDECLAT"]:
                seconds, count = measure(sc, name, data_dir, args.min_supp, partitions=max(8, cores * 2))
                rows.append({"experiment": "cores", "algorithm": name, "cores": cores, "transactions": n,
                             "seconds": round(seconds, 3), "itemsets": count})
                print(rows[-1], flush=True)
        finally:
            sc.stop()
    return rows


def experiment_items(args, tmp):
    sc = new_context(args.cores)
    rows = []
    try:
        for items in [int(x) for x in args.item_list.split(",")]:
            data_dir = os.path.join(tmp, "items_{}".format(items))
            n = generate(data_dir, args.items_n, items)
            for name in CRISP:
                seconds, count = measure(sc, name, data_dir, args.min_supp)
                rows.append({"experiment": "items", "algorithm": name, "items": items, "transactions": n,
                             "seconds": round(seconds, 3), "itemsets": count})
                print(rows[-1], flush=True)
    finally:
        sc.stop()
    return rows


def experiment_cache(args, tmp):
    sc = new_context(args.cores)
    rows = []
    try:
        data_dir = os.path.join(tmp, "cache_data")
        n = generate(data_dir, args.items_n, args.items)
        measure(sc, "DApriori", data_dir, args.min_supp, repeats=1)
        for name in list(CRISP):
            for cache in (False, True):
                seconds, count = measure(sc, name, data_dir, args.min_supp, cache=cache)
                rows.append({"experiment": "cache", "algorithm": name, "cache_input": cache, "transactions": n,
                             "seconds": round(seconds, 3), "itemsets": count})
                print(rows[-1], flush=True)
    finally:
        sc.stop()
    return rows


def experiment_sequences(args, tmp):
    from pyspark.sql import SparkSession

    sc = new_context(args.cores)
    session = SparkSession.builder.getOrCreate()
    rows = []
    try:
        warm = os.path.join(tmp, "seq_warmup")
        generate_sequences(warm, 800, args.seq_items)
        prefixspan_spark(session, sc.textFile(warm).map(parse_sequence), 0.1)
        for n in [int(x) for x in args.seq_sizes.split(",")]:
            data_dir = os.path.join(tmp, "seq_{}".format(n))
            real_n = generate_sequences(data_dir, n, args.seq_items)
            local = [parse_sequence(line) for part in sorted(os.listdir(data_dir))
                     for line in open(os.path.join(data_dir, part)).read().splitlines()]
            runs = {
                "prefixspan (local)": lambda: prefixspan(local, args.seq_min_supp),
                "prefixspan_spark (list)": lambda: prefixspan_spark(session, local, args.seq_min_supp),
                "prefixspan_spark (RDD)": lambda: prefixspan_spark(
                    session, sc.textFile(data_dir, minPartitions=8).map(parse_sequence), args.seq_min_supp),
            }
            if args.seq_dataframe:                                              # only if the library already accepts DataFrames
                def from_dataframe():
                    frame = session.createDataFrame(sc.parallelize(local, 8).map(lambda s: (s,)), ["sequence"])
                    return prefixspan_spark(session, frame, args.seq_min_supp)
                runs["prefixspan_spark (DataFrame)"] = from_dataframe
            for name, run in runs.items():
                before = driver_rss_mb()
                best, patterns = None, None
                for _ in range(2):
                    seconds, result = timed(run)
                    best = seconds if best is None else min(best, seconds)
                    patterns = len(result)
                rows.append({"experiment": "sequences", "algorithm": name, "transactions": real_n, "seconds": round(best, 3),
                             "itemsets": patterns, "driver_rss_mb": round(driver_rss_mb(), 1),
                             "driver_rss_growth_mb": round(driver_rss_mb() - before, 1)})
                print(rows[-1], flush=True)
    finally:
        sc.stop()
    return rows


def summarize(rows):
    """Scaling exponent t ~ N^alpha, and speedup/efficiency per core."""
    lines = []
    data = [r for r in rows if r["experiment"] == "data"]
    for name in sorted({r["algorithm"] for r in data}):
        points = sorted((r["transactions"], r["seconds"]) for r in data if r["algorithm"] == name)
        if len(points) >= 3:
            alpha = np.polyfit(np.log([p[0] for p in points]), np.log([p[1] for p in points]), 1)[0]
            lines.append("{:<18} t ~ N^{:.2f}  ({} puntos, de {:,} a {:,} transacciones)".format(
                name, alpha, len(points), points[0][0], points[-1][0]))
    sequences = [r for r in rows if r["experiment"] == "sequences"]
    for name in sorted({r["algorithm"] for r in sequences}):
        points = sorted((r["transactions"], r["seconds"]) for r in sequences if r["algorithm"] == name)
        if len(points) >= 3:
            alpha = np.polyfit(np.log([p[0] for p in points]), np.log([p[1] for p in points]), 1)[0]
            lines.append("{:<30} t ~ N^{:.2f}  ({} puntos, de {:,} a {:,} secuencias)".format(
                name, alpha, len(points), points[0][0], points[-1][0]))
    cores = [r for r in rows if r["experiment"] == "cores"]
    for name in sorted({r["algorithm"] for r in cores}):
        series = sorted((r["cores"], r["seconds"]) for r in cores if r["algorithm"] == name)
        lines.append("{:<18} ".format(name) + "  ".join(
            "{}c: {:.1f}s (x{:.2f}, E={:.0%})".format(c, s, series[0][1] / s, series[0][1] / s / (c / series[0][0]))
            for c, s in series))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", choices=["data", "cores", "items", "cache", "sequences", "all"])
    parser.add_argument("--output", help="JSON file where to save the measurements")
    parser.add_argument("--min-supp", type=float, default=0.05)
    parser.add_argument("--cores", type=int, default=8, help="cores of the data/items/cache experiments")
    parser.add_argument("--items", type=int, default=100, help="items distintos (experimentos data/cache/cores)")
    parser.add_argument("--sizes", default="25000,50000,100000,200000,400000", help="N del experimento data")
    parser.add_argument("--fuzzy-max", type=int, default=100000, help="maximum N for the fuzzy algorithms")
    parser.add_argument("--core-list", default="1,2,4,8", help="nucleos del experimento cores")
    parser.add_argument("--cores-n", type=int, default=200000, help="N del experimento cores")
    parser.add_argument("--item-list", default="50,100,200,400", help="items del experimento items")
    parser.add_argument("--items-n", type=int, default=100000, help="N of the items/cache experiments")
    parser.add_argument("--seq-sizes", default="5000,10000,20000,40000", help="number of sequences of the sequences experiment")
    parser.add_argument("--seq-items", type=int, default=60, help="distinct items of the sequences")
    parser.add_argument("--seq-min-supp", type=float, default=0.05, help="minimum support of the sequences experiment")
    parser.add_argument("--seq-dataframe", action="store_true",
                        help="include the input as a Spark DataFrame (if prefixspan_spark already accepts it)")
    args = parser.parse_args(argv)

    experiments = ["data", "cores", "items", "cache", "sequences"] if "all" in args.experiments else args.experiments
    rows = []
    with tempfile.TemporaryDirectory(prefix="armxtend-bench-") as tmp:
        for experiment in experiments:
            print("== experimento: {} ==".format(experiment), flush=True)
            rows += {"data": experiment_data, "cores": experiment_cores,
                     "items": experiment_items, "cache": experiment_cache,
                                  "sequences": experiment_sequences}[experiment](args, tmp)

    print("\nResumen\n-------\n" + summarize(rows))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(rows, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
