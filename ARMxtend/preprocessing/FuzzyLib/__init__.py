# -*- coding: utf-8 -*-
"""
Fuzzification utilities for tabular data, as a preprocessing step before the fuzzy itemset / association rule mining
algorithms of ARMxtend (FIM.BD_FARE, FIM.FARE, FFIM).

It supports triangular (Ruspini), trapezoidal and Gaussian partitions, defined by hand (`FuzzyLib.Fuzzification`) or
computed automatically from the data by uniform spacing, quantiles or k-means (`FuzzyLib.AutoFuzzification`); see
``preprocessing.membership``.

@author Carlos Fernandez-Basso
"""
from typing import Any, Callable, Dict

import pandas as pd

from ..membership import (
    MEMBERSHIPS,
    automatic_partition,
    default_labels,
    gaussian_partition,
    trapezoidal_partition,
    triangular_partition,
)

# Historical (private) name of the triangular partition
_triangular_partition = triangular_partition


class FuzzyLib(object):
    """
    Load a tabular dataset and allow fuzzifying its numeric attributes into named fuzzy sets (e.g. 'Temperature' ->
    'cold', 'comfort', 'warm'), adding one membership-degree column per fuzzy label.
    """

    def __init__(self):
        self.data = pd.DataFrame()
        self.atributes: list = []
        self.types: Any = []
        # {attribute: {'membership', 'labels', 'params', 'columns'}} of each fuzzified attribute
        self.partitions = {}

    def LoadData(self, File=""):
        if File == "":
            raise ValueError("The path of the data file to load must be specified")
        self.data = pd.read_csv(File)
        self.atributes = list(self.data.columns)
        self.types = self.data.dtypes

    def GetData(self):
        return self.data

    def GetAtributes(self):
        return self.atributes

    def GetTypes(self):
        return self.types

    def _add_memberships(self, attribute, labels, memberships, membership, params):
        columns = []
        for labelIdx, label in enumerate(labels):
            columnName = "{}_{}".format(attribute, label)
            self.data[columnName] = memberships[:, labelIdx]
            columns.append(columnName)
        self.partitions[attribute] = {
            "membership": membership, "labels": list(labels), "params": list(params), "columns": columns,
        }
        return columns

    def Fuzzification(self, Atri=None, thresholds=None, FuzzyLabel=None, membership="triangular"):
        """
        Fuzzify a set of numeric attributes of the loaded dataset, adding a new column '<attribute>_<label>' for each
        fuzzy label, with the membership degree of each row in that label.

        Args:
            Atri (list[str]): names of the numeric attributes to fuzzify
            thresholds (list[list]): for each attribute of `Atri`, the parameters of its fuzzy sets (one per label),
                according to `membership`:
                'triangular'  -> the peaks (float) of a Ruspini partition (sum of degrees = 1)
                'trapezoidal' -> tuples (a, b, c, d) (see `preprocessing.membership.trapezoidal_membership`)
                'gaussian'    -> tuples (center, sigma)
            FuzzyLabel (list[list[str]]): for each attribute of `Atri`, the names of the fuzzy labels
            membership (str): 'triangular' (default), 'trapezoidal' or 'gaussian'

        Returns:
            The list of names of the fuzzy columns added to the dataset
        """
        if membership not in MEMBERSHIPS:
            raise ValueError("membership must be one of {}, got: '{}'".format(MEMBERSHIPS, membership))
        Atri = Atri or []
        thresholds = thresholds or []
        FuzzyLabel = FuzzyLabel or []

        if not (len(Atri) == len(thresholds) == len(FuzzyLabel)):
            raise ValueError("Atri, thresholds and FuzzyLabel must have the same length")

        partitions: Dict[str, Callable[[Any, Any], Any]] = {
            "triangular": triangular_partition,
            "trapezoidal": trapezoidal_partition,
            "gaussian": gaussian_partition,
        }
        partition = partitions[membership]

        addedColumns = []
        for attr, params, labels in zip(Atri, thresholds, FuzzyLabel):
            if len(params) != len(labels):
                raise ValueError(
                    "Attribute '{}' has {} peaks but {} labels".format(attr, len(params), len(labels)))

            memberships = partition(self.data[attr].values, params)
            addedColumns.extend(self._add_memberships(attr, labels, memberships, membership, params))

        self.atributes = list(self.data.columns)
        return addedColumns

    def AutoFuzzification(self, Atri=None, n_labels=3, method="quantile", FuzzyLabel=None,
                          membership="triangular"):
        """
        Like `Fuzzification`, but computing the partition of each attribute automatically from its data (see
        `preprocessing.membership.automatic_partition`). The computed parameters are left in
        `self.partitions[attribute]['params']`, in case you want to reuse them with `Fuzzification` on other data.

        Args:
            Atri (list[str], optional): attributes to fuzzify (all the numeric columns by default)
            n_labels (int): number of fuzzy sets per attribute
            method (str): 'uniform', 'quantile' (default) or 'kmeans'
            FuzzyLabel (list[str], optional): names of the labels, common to all the attributes (by default, 'low',
                'medium', 'high' for 3, etc.)
            membership (str): 'triangular' (default), 'trapezoidal' or 'gaussian'

        Returns:
            The list of names of the fuzzy columns added to the dataset
        """
        if Atri is None:
            Atri = list(self.data.select_dtypes("number").columns)
        labels = list(FuzzyLabel) if FuzzyLabel is not None else default_labels(n_labels)
        if len(labels) != n_labels:
            raise ValueError("{} labels were requested but {} names were given".format(n_labels, len(labels)))

        addedColumns = []
        for attr in Atri:
            try:
                memberships, params = automatic_partition(self.data[attr].values, n_labels, method, membership)
            except ValueError as error:
                raise ValueError("Attribute '{}': {}".format(attr, error))
            addedColumns.extend(self._add_memberships(attr, labels, memberships, membership, params))

        self.atributes = list(self.data.columns)
        return addedColumns

    def GetFuzzyTransactions(self, columns=None, min_degree=0.0):
        """
        Convert membership-degree columns into fuzzy transactions, the input format of `FFIM.fuzzy_fpgrowth` (and of the
        Big Data fuzzy algorithms once parallelized): a list of (item, degree) pairs per row.

        Args:
            columns (list[str], optional): columns to use (by default, all those added by
                `Fuzzification`/`AutoFuzzification`)
            min_degree (float): only the items with degree strictly greater than this value are included (0.0 omits
                the items with null membership)

        Returns:
            list[list[tuple[str, float]]]
        """
        if columns is None:
            columns = [column for partition in self.partitions.values() for column in partition["columns"]]
        degrees = self.data[columns].to_numpy(dtype=float)
        return [
            [(column, float(degree)) for column, degree in zip(columns, row) if degree > min_degree]
            for row in degrees
        ]
