# -*- coding: utf-8 -*-


"""
Algoritmo SeqApriori usando Spark
@author Carlos Fernandez-Basso (2016)

Algoritmo Apriori para la obtención de itemsets frecuentes en conjuntos difusos
"""
from _decimal import Decimal
import numpy as np
from decimal import Decimal
from decimal import Decimal

class AprioriTID(object):
    @staticmethod
    def transformation_function(a_model):
        delim = a_model.delim

        def _transformation_function(row):
            return row.split(delim)

        return _transformation_function

    def __init__(self):
        self.delim = ','
        self.data = sc.textFile('some.csv')

    def run_model(self):
        self.data = self.data.map(model.transformation_function(self))

    def GenItems(Long, Items):
        """
        Funcion que nos permite generar los itemset para las siguientes fases del algoritmo SeqApriori.
        :param Long: Longitud de los item que se quieren obtener
        :param Items: Lista de (i-1)-items que se quieren generar
        :return: Lista con los itemsets generados
        """
        listaOut = []
        ListaRepes = []
        for i in range(0, len(Items)):
            if Long == 1:
                A = Items[i]
                Used = Items[i].split("-")
            else:
                A = Items[i].split("-")
                Used = Items[i].split("-")
            for j in range(i + 1, len(Items)):
                if Long == 1:
                    B = Items[j]
                else:
                    B = Items[j].split("-")

                common = list(set(A) - (set(A) - set(B)))
                NoRepeat = [x for x in B if x not in set(common + Used)]

                for item in NoRepeat:
                    Used.append(item)
                    A1 = '-'.join(map(str, A + [item]))
                    if not (sorted(Used) in ListaRepes):
                        ListaRepes.append(sorted(Used))
                        listaOut.append(A1)
        return listaOut

    def GenFuzzItems(Long, Items):
        """
        Funcion que nos permite generar los itemset para las siguientes fases del algoritmo SeqApriori.
        :param Long: Longitud de los item que se quieren obtener
        :param Items: Lista de (i-1)-items que se quieren generar
        :return: Lista con los itemsets generados
        """
        listaOut = []
        for i in range(0, len(Items)):
            if Long == 1:
                A = Items[i]
                Used = Items[i].split("-")
            else:
                A = Items[i].split("-")
                Used = Items[i].split("-")
            for j in range(i + 1, len(Items)):
                if Long == 1:
                    B = Items[j]
                else:
                    B = Items[j].split("-")
                common = list(set(A) - (set(A) - set(B)))
                NoRepeat = [x for x in B if x not in set(common + Used)]

                for item in NoRepeat:
                    Used.append(item)
                    A1 = '-'.join(map(str, A + [item]))
                    listaOut.append(A1)
        return listaOut

    def FiltradoFrecuentes(broadcastFrequ, transaction):
        lista = []
        for i in transaction:
            if i[0] in broadcastFrequ.value:
                lista.append(i)
        if len(lista) != 0:
            return lista
        else:
            return []

    def FiltradoFrecuentesVset(broadcastFrequ, transaction):
        s = set(broadcastFrequ.value) & set(transaction)
        lista = dict((k, transaction[k]) for k in s)
        return lista

    def FilterEmpty(transaction):
        lista = []
        for i in transaction:
            if sum(i[1]) != 0:
                lista.append(i)
        if len(lista) != 0:
            return lista
        else:
            return []

    #######################################################################################################################
    ########################                              Phase1                                    ########################
    ########################################################################################################################
    def Ordenar(transaction):
        # split the input line in word and count on the comma
        items = transaction.split(",")
        lista = []
        for a in broadcastOrden.value:
            if a in items:
                lista.append(a)
        # turn the count to an integer
        if lista == []:
            lista = 0
        return (lista)

    def ContarPhase1(transaction):
        return transaction

    def ReducePhase1(x, y):
        return x + y

    ########################################################################################################################
    ########################                              Phase 2                                   ########################
    ########################################################################################################################
    def ContarPhase2(Items, AphaCuts, transaction):
        x = Items.value
        if type(transaction) == type((1, 2)):
            transaction = [transaction]
        tra = dict(transaction)
        lista = []
        #  print("\nTransacction " + str(tra))

        for i in x:
            s = i.split("-")
            s = set(s)
            keys_tra = set(tra.keys())
            # print("\nKeys: " + str(tra))
            #   print("\nitems: " + str(i))
            intersection = s & keys_tra
            auxFuz = np.full(AphaCuts.value[0], 1)
            # for j in intersection:
            for j in s:
                if j in tra:
                    auxFuz = tra[j] * auxFuz
            # print("\nFuzz: " + str(auxFuz))
            lista.append((i, auxFuz))
        #  print("\nlista: " + str(lista))
        return lista

    def ReducePhase2(x, y):
        return x + y

    ########################################################################################################################
    ########################                          FUZZY FNTIONS                                 ########################
    ########################################################################################################################
    # def Ordenar(transaction):
    #     # split the input line in word and count on the comma
    #     items = transaction.split(",")
    #     lista = []
    #     for a in broadcastOrden.value:
    #         if a in items:
    #             lista.append(a)
    #     # turn the count to an integer
    #     if lista == []:
    #         lista = 0
    #     return (lista)

    def AlphaCortes(item, alpha):
        # Alphacutincre=Decimal(1.0 / alpha)
        Alfacut = np.linspace(0, 1, num=alpha, endpoint=False)[::-1]
        Alfacut = np.append(1, Alfacut)
        med = Decimal(1.0)
        list = []
        for i in range(0, alpha):
            med = Alfacut[i]
            if round(med, 10) <= item > 0:
                list.append(1)
            else:
                list.append(0)
            # max = med
            # med = Decimal(1) - Decimal(Alphacutincre * Decimal(i+1))
        return np.array(list)

    def CreateTList(broadcastParam, transaction):
        items = transaction.split(",")
        lista = []
        for i in range(0, len(items)):
            if items[i] == "":
                items[i] = 0
            x = AlphaCortes(float(items[i]), broadcastParam.value[0])
            lista.append((str(broadcastParam.value[4][i]), x))  # .copy()))
        return lista