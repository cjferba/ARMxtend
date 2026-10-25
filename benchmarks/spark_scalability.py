# -*- coding: utf-8 -*-
"""
Benchmark de escalabilidad de los algoritmos Big Data de ARMxtend sobre un Spark REAL en modo local.

Mide, sobre datos sinteticos (items con popularidad tipo Zipf, cestas de ~10 items) leidos desde
ficheros de texto con ``sc.textFile`` (como en un uso real: el RDD se re-lee y re-parsea salvo que
se cachee):

* ``data``  -- tiempo frente al numero de transacciones N (a nucleos fijos); se ajusta el exponente
               alfa de t ~ N^alfa (alfa ~ 1 = escalado lineal) y se registra el pico de memoria RSS
               del driver (relevante para DECLAT, que lleva las TID-lists al driver).
* ``cores`` -- aceleracion S = T1/Tn y eficiencia E = S/n frente al numero de nucleos (a N fijo).
* ``items`` -- tiempo frente al numero de items distintos (a N fijo).
* ``cache`` -- efecto de cachear el RDD de entrada.

Uso (desde la raiz del repositorio, con un JDK 17+ en JAVA_HOME)::

    python benchmarks/spark_scalability.py data cores items cache --output results.json

Es un benchmark en UNA maquina (``local[n]``): mide como se reparte el trabajo entre nucleos y como
crece el coste con los datos, no la red ni el reparto entre nodos de un cluster real.
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

CRISP = {"DApriori": DApriori, "DAprioriTID": DAprioriTID, "DECLAT": DECLAT}
FUZZY = {"FuzzyDAprioriTID": FuzzyDAprioriTID, "FuzzyDECLAT": FuzzyDECLAT}


def generate(path, n_transactions, n_items, mean_basket=10, seed=0, shards=8):
    """Escribe `shards` ficheros de texto con transacciones 'i3,i17,...' (items con popularidad tipo Zipf)."""
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
        rdd.count()                 # materializa la cache fuera del tiempo medido
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


def summarize(rows):
    """Exponente de escalado t ~ N^alfa, y aceleracion/eficiencia por nucleos."""
    lines = []
    data = [r for r in rows if r["experiment"] == "data"]
    for name in sorted({r["algorithm"] for r in data}):
        points = sorted((r["transactions"], r["seconds"]) for r in data if r["algorithm"] == name)
        if len(points) >= 3:
            alpha = np.polyfit(np.log([p[0] for p in points]), np.log([p[1] for p in points]), 1)[0]
            lines.append("{:<18} t ~ N^{:.2f}  ({} puntos, de {:,} a {:,} transacciones)".format(
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
    parser.add_argument("experiments", nargs="+", choices=["data", "cores", "items", "cache", "all"])
    parser.add_argument("--output", help="fichero JSON donde guardar las mediciones")
    parser.add_argument("--min-supp", type=float, default=0.05)
    parser.add_argument("--cores", type=int, default=8, help="nucleos de los experimentos data/items/cache")
    parser.add_argument("--items", type=int, default=100, help="items distintos (experimentos data/cache/cores)")
    parser.add_argument("--sizes", default="25000,50000,100000,200000,400000", help="N del experimento data")
    parser.add_argument("--fuzzy-max", type=int, default=100000, help="N maximo para los algoritmos difusos")
    parser.add_argument("--core-list", default="1,2,4,8", help="nucleos del experimento cores")
    parser.add_argument("--cores-n", type=int, default=200000, help="N del experimento cores")
    parser.add_argument("--item-list", default="50,100,200,400", help="items del experimento items")
    parser.add_argument("--items-n", type=int, default=100000, help="N de los experimentos items/cache")
    args = parser.parse_args(argv)

    experiments = ["data", "cores", "items", "cache"] if "all" in args.experiments else args.experiments
    rows = []
    with tempfile.TemporaryDirectory(prefix="armxtend-bench-") as tmp:
        for experiment in experiments:
            print("== experimento: {} ==".format(experiment), flush=True)
            rows += {"data": experiment_data, "cores": experiment_cores,
                     "items": experiment_items, "cache": experiment_cache}[experiment](args, tmp)

    print("\nResumen\n-------\n" + summarize(rows))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(rows, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
