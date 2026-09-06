# -*- coding: utf-8 -*-
"""
Doble local (sin dependencia de una JVM/cluster real) de la minima API de
Spark (RDD + SparkContext.broadcast) usada por los algoritmos de
ARMxtend.FIM en Big Data. Permite testear la logica de esos algoritmos en
CI/entornos sin Java, ejecutando exactamente el mismo codigo de produccion
que se ejecutaria sobre un SparkContext real.

Uso exclusivo en tests -- no forma parte de la libreria.
"""
from collections import defaultdict


class FakeBroadcast(object):
    def __init__(self, value):
        self.value = value

    def unpersist(self):
        pass


class FakeSparkContext(object):
    def parallelize(self, data):
        return FakeRDD(list(data))

    def broadcast(self, value):
        return FakeBroadcast(value)

    def union(self, rdds):
        data = []
        for rdd in rdds:
            data.extend(rdd.collect())
        return FakeRDD(data)


class FakeRDD(object):
    def __init__(self, data):
        self._data = list(data)

    def map(self, f):
        return FakeRDD([f(x) for x in self._data])

    def flatMap(self, f):
        result = []
        for x in self._data:
            result.extend(f(x))
        return FakeRDD(result)

    def filter(self, f):
        return FakeRDD([x for x in self._data if f(x)])

    def reduceByKey(self, f):
        acc = {}
        order = []
        for key, value in self._data:
            if key in acc:
                acc[key] = f(acc[key], value)
            else:
                acc[key] = value
                order.append(key)
        return FakeRDD([(key, acc[key]) for key in order])

    def groupByKey(self):
        groups = defaultdict(list)
        for key, value in self._data:
            groups[key].append(value)
        return FakeRDD(list(groups.items()))

    def mapValues(self, f):
        return FakeRDD([(key, f(value)) for key, value in self._data])

    def zipWithIndex(self):
        return FakeRDD(list(zip(self._data, range(len(self._data)))))

    def collect(self):
        return list(self._data)

    def collectAsMap(self):
        return dict(self._data)

    def count(self):
        return len(self._data)

    def cache(self):
        return self

    def unpersist(self):
        return self
