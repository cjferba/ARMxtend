# -*- coding: utf-8 -*-
"""
Utilidades de fuzzificacion de datos tabulares, como paso de preprocesado
previo a los algoritmos difusos de mineria de itemsets/reglas de asociacion
de ARMxtend (FIM.BD_FARE, FIM.FARE, FFIM).

@author Carlos Fernandez-Basso
"""
import pandas as pd
import numpy as np


def _triangular_partition(values, peaks):
    """
    Particion difusa triangular de Ruspini: dado un conjunto de picos
    ordenados ascendentemente, devuelve el grado de pertenencia de cada
    valor a cada uno de los `len(peaks)` conjuntos difusos definidos, de
    forma que los grados de todos los conjuntos suman siempre 1 para
    cualquier valor.

    Los dos conjuntos extremos son "hombros" (mantienen pertenencia total
    fuera de su ultimo tramo de pendiente); los conjuntos intermedios son
    triangulos con vertice en su pico correspondiente.

    Argumentos:
        values (array-like): valores numericos a fuzzificar
        peaks (Sequence[float]): picos, ordenados ascendentemente, de cada
            conjunto difuso (uno por etiqueta)

    Retorna:
        numpy.ndarray de forma (len(values), len(peaks)) con los grados de
        pertenencia a cada conjunto difuso
    """
    values = np.asarray(values, dtype=float)
    peaks = np.asarray(sorted(peaks), dtype=float)
    n = len(peaks)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 picos/etiquetas para fuzzificar un atributo")

    memberships = np.zeros((len(values), n))

    # Hombro izquierdo (primera etiqueta)
    memberships[:, 0] = np.clip((peaks[1] - values) / (peaks[1] - peaks[0]), 0.0, 1.0)
    memberships[values <= peaks[0], 0] = 1.0

    # Hombro derecho (ultima etiqueta)
    memberships[:, -1] = np.clip((values - peaks[-2]) / (peaks[-1] - peaks[-2]), 0.0, 1.0)
    memberships[values >= peaks[-1], -1] = 1.0

    # Etiquetas intermedias: triangulo con vertice en peaks[i]
    for i in range(1, n - 1):
        left = np.clip((values - peaks[i - 1]) / (peaks[i] - peaks[i - 1]), 0.0, 1.0)
        right = np.clip((peaks[i + 1] - values) / (peaks[i + 1] - peaks[i]), 0.0, 1.0)
        memberships[:, i] = np.minimum(left, right)

    return memberships


class FuzzyLib(object):
    """
    Carga un conjunto de datos tabular y permite fuzzificar sus atributos
    numericos en conjuntos difusos con nombre (p.ej. 'Temperature' -> 'cold',
    'comfort', 'warm'), anadiendo una columna de grado de pertenencia por
    cada etiqueta difusa.
    """

    def __init__(self):
        self.data = pd.DataFrame()
        self.atributes = []
        self.types = []

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

    def Fuzzification(self, Atri=None, thresholds=None, FuzzyLabel=None):
        """
        Fuzzifica un conjunto de atributos numericos del dataset cargado,
        anadiendo una columna nueva '<atributo>_<etiqueta>' por cada
        etiqueta difusa, con el grado de pertenencia (particion triangular
        de Ruspini, ver `_triangular_partition`) de cada fila a dicha
        etiqueta.

        Argumentos:
            Atri (list[str]): nombres de los atributos numericos a fuzzificar
            thresholds (list[list[float]]): para cada atributo de `Atri`, la
                lista de picos (uno por etiqueta difusa) que definen la
                particion triangular
            FuzzyLabel (list[list[str]]): para cada atributo de `Atri`, los
                nombres de las etiquetas difusas asociadas a cada pico

        Retorna:
            La lista de nombres de las columnas difusas anadidas al dataset
        """
        Atri = Atri or []
        thresholds = thresholds or []
        FuzzyLabel = FuzzyLabel or []

        if not (len(Atri) == len(thresholds) == len(FuzzyLabel)):
            raise ValueError("Atri, thresholds y FuzzyLabel deben tener la misma longitud")

        addedColumns = []
        for attr, peaks, labels in zip(Atri, thresholds, FuzzyLabel):
            if len(peaks) != len(labels):
                raise ValueError(
                    "El atributo '{}' tiene {} picos pero {} etiquetas".format(attr, len(peaks), len(labels)))

            memberships = _triangular_partition(self.data[attr].values, peaks)
            for labelIdx, label in enumerate(labels):
                columnName = "{}_{}".format(attr, label)
                self.data[columnName] = memberships[:, labelIdx]
                addedColumns.append(columnName)

        self.atributes = list(self.data.columns)
        return addedColumns
