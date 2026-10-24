# -*- coding: utf-8 -*-
"""
Utilidades de fuzzificacion de datos tabulares, como paso de preprocesado
previo a los algoritmos difusos de mineria de itemsets/reglas de asociacion
de ARMxtend (FIM.BD_FARE, FIM.FARE, FFIM).

Admite particiones triangulares (de Ruspini), trapezoidales y gaussianas,
definidas a mano (`FuzzyLib.Fuzzification`) o calculadas automaticamente a
partir de los datos por reparto uniforme, cuantiles o k-means
(`FuzzyLib.AutoFuzzification`); ver ``preprocessing.membership``.

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

# Nombre historico (privado) de la particion triangular
_triangular_partition = triangular_partition


class FuzzyLib(object):
    """
    Carga un conjunto de datos tabular y permite fuzzificar sus atributos
    numericos en conjuntos difusos con nombre (p.ej. 'Temperature' -> 'cold',
    'comfort', 'warm'), anadiendo una columna de grado de pertenencia por
    cada etiqueta difusa.
    """

    def __init__(self):
        self.data = pd.DataFrame()
        self.atributes: list = []
        self.types: Any = []
        # {atributo: {'membership', 'labels', 'params', 'columns'}} de cada atributo fuzzificado
        self.partitions = {}

    def LoadData(self, File=""):
        if File == "":
            raise ValueError("Debe especificarse la ruta del fichero de datos a cargar")
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
        Fuzzifica un conjunto de atributos numericos del dataset cargado,
        anadiendo una columna nueva '<atributo>_<etiqueta>' por cada
        etiqueta difusa, con el grado de pertenencia de cada fila a dicha
        etiqueta.

        Argumentos:
            Atri (list[str]): nombres de los atributos numericos a fuzzificar
            thresholds (list[list]): para cada atributo de `Atri`, los
                parametros de sus conjuntos difusos (uno por etiqueta),
                segun `membership`:
                'triangular'  -> los picos (float) de una particion de
                                 Ruspini (suma de grados = 1)
                'trapezoidal' -> tuplas (a, b, c, d) (ver
                                 `preprocessing.membership.trapezoidal_membership`)
                'gaussian'    -> tuplas (centro, sigma)
            FuzzyLabel (list[list[str]]): para cada atributo de `Atri`, los
                nombres de las etiquetas difusas
            membership (str): 'triangular' (por defecto), 'trapezoidal' o
                'gaussian'

        Retorna:
            La lista de nombres de las columnas difusas anadidas al dataset
        """
        if membership not in MEMBERSHIPS:
            raise ValueError("membership debe ser uno de {}, recibido: '{}'".format(MEMBERSHIPS, membership))
        Atri = Atri or []
        thresholds = thresholds or []
        FuzzyLabel = FuzzyLabel or []

        if not (len(Atri) == len(thresholds) == len(FuzzyLabel)):
            raise ValueError("Atri, thresholds y FuzzyLabel deben tener la misma longitud")

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
                    "El atributo '{}' tiene {} picos pero {} etiquetas".format(attr, len(params), len(labels)))

            memberships = partition(self.data[attr].values, params)
            addedColumns.extend(self._add_memberships(attr, labels, memberships, membership, params))

        self.atributes = list(self.data.columns)
        return addedColumns

    def AutoFuzzification(self, Atri=None, n_labels=3, method="quantile", FuzzyLabel=None,
                          membership="triangular"):
        """
        Como `Fuzzification`, pero calculando automaticamente la particion de
        cada atributo a partir de sus datos (ver
        `preprocessing.membership.automatic_partition`). Los parametros
        calculados quedan en `self.partitions[atributo]['params']`, por si
        se quieren reutilizar con `Fuzzification` sobre otros datos.

        Argumentos:
            Atri (list[str], opcional): atributos a fuzzificar (por
                defecto, todas las columnas numericas)
            n_labels (int): numero de conjuntos difusos por atributo
            method (str): 'uniform', 'quantile' (por defecto) o 'kmeans'
            FuzzyLabel (list[str], opcional): nombres de las etiquetas,
                comunes a todos los atributos (por defecto, 'low',
                'medium', 'high' para 3, etc.)
            membership (str): 'triangular' (por defecto), 'trapezoidal'
                o 'gaussian'

        Retorna:
            La lista de nombres de las columnas difusas anadidas al dataset
        """
        if Atri is None:
            Atri = list(self.data.select_dtypes("number").columns)
        labels = list(FuzzyLabel) if FuzzyLabel is not None else default_labels(n_labels)
        if len(labels) != n_labels:
            raise ValueError("Se piden {} etiquetas pero se dan {} nombres".format(n_labels, len(labels)))

        addedColumns = []
        for attr in Atri:
            try:
                memberships, params = automatic_partition(self.data[attr].values, n_labels, method, membership)
            except ValueError as error:
                raise ValueError("Atributo '{}': {}".format(attr, error))
            addedColumns.extend(self._add_memberships(attr, labels, memberships, membership, params))

        self.atributes = list(self.data.columns)
        return addedColumns

    def GetFuzzyTransactions(self, columns=None, min_degree=0.0):
        """
        Convierte columnas de grados de pertenencia en transacciones difusas,
        el formato de entrada de `FFIM.fuzzy_fpgrowth` (y de los algoritmos
        difusos de Big Data tras paralelizarlas): una lista de pares
        (item, grado) por fila.

        Argumentos:
            columns (list[str], opcional): columnas a usar (por defecto,
                todas las anadidas por `Fuzzification`/`AutoFuzzification`)
            min_degree (float): solo se incluyen los items con grado
                estrictamente mayor que este valor (0.0 omite los
                items con pertenencia nula)

        Retorna:
            list[list[tuple[str, float]]]
        """
        if columns is None:
            columns = [column for partition in self.partitions.values() for column in partition["columns"]]
        degrees = self.data[columns].to_numpy(dtype=float)
        return [
            [(column, float(degree)) for column, degree in zip(columns, row) if degree > min_degree]
            for row in degrees
        ]
