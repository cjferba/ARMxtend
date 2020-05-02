# -*- coding: utf-8 -*-


"""
Algoritmo SeqApriori usando Spark
@author Carlos Fernandez-Basso (2016)

Algoritmo Apriori para la obtención de itemsets frecuentes en conjuntos difusos
"""

import numpy as np
import pandas as pd


class AREtoGraph():
    __Rules=""
    __FreItem=""
    __MeasuresRules=[]
    __MeasuresItems=[]
    __SepRule=""
    __SepFI=""
    __SepItems=""

    def __init__(self, path, MeasuresRules, MeasuresItems, SepRule=";", SepFI=";", SepItems=";"):
        """
        Funcion contructor que crea el objeto de la clase para relizar el procesamiento
        :param path: Path del archivo de resultados que va  cargar y procesar
        :param MeasuresRules: Tipos de medidas que tiene las reglas
        :param MeasuresItems: Tipos de medidas que tiene los itemsets
        :param SepRule:
        :param SepFI:
        :param SepItems:
        :return: Objeto de la clase AREtoGraph
        """

    def exportGraph(self,type=0):
        gp=""
        return gp