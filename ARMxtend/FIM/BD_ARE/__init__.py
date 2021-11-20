# -*- coding: utf-8 -*-


"""
Algoritmo SeqApriori usando Spark
@author Carlos Fernandez-Basso (2016)

Algoritmo Apriori para la obtención de itemsets frecuentes en conjuntos difusos
"""
from _decimal import Decimal
import pickle
import itertools
import numpy as np

from decimal import Decimal


class Apriori(object):
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


    def CreateTList(broadcastParam, transaction):
        items = transaction.split(",")
        lista = []
        for i in range(0, len(items)):
            if items[i] == "":
                items[i] = 0
            x = AlphaCortes(float(items[i]), broadcastParam.value[0])
            lista.append((str(broadcastParam.value[4][i]), x))  # .copy()))
        return lista


class Eclat(object):
    def GenItems(Long, Items):
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

    def FiltradoFrecuentes(broadcastFrequ, transaction):
        lista = []
        for i in transaction:
            if i[0] in broadcastFrequ.value:
                lista.append(i)
        return lista
        return 0

    def FilterEmpty(transaction):
        lista = []
        for i in transaction:
            if sum(i[1]) != 0:
                lista.append(i)
        if len(lista) != 0:
            return lista
        else:
            return []

    def TdiList(transaction):
        lista = list()
        for i in transaction[0]:
            if sum(i[1]) != 0:
                lista.append((i[0], (transaction[1], i[1])))
        return lista

    def ReduceTDiList(x, y):
        if type(x) != type(list()):
            x = [x]
        if type(y) != type(list()):
            y = [y]
        return x + y

    ######################################################################################################################
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
        for i in x:
            s = i.split("-")
            # s = set(s)
            # keys_tra = set(tra.keys())
            # intersection = s & keys_tra
            auxFuz = np.full(AphaCuts.value[0], 1)
            # for j in intersection:
            for j in s:
                if j in tra:
                    auxFuz = tra[j] * auxFuz
            lista.append((i, auxFuz))
        return lista

    def ReducePhase2(x, y):
        return x + y

    ########################################################################################################################
    ########################                          FUZZY FNTIONS                                 ########################
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

    def AlphaCortes(item, alpha):
        max = 1.0
        med = 1.0 - (1.0 / alpha)
        list = []
        for i in range(0, alpha):
            if med <= item > 0:
                list.append(1)
            else:
                list.append(0)
            # max = med
            med = 1.0 - (1.0 / float(alpha)) * (i + 2)
        return np.array(list)

    def CreateTList(broadcastParam, transaction):
        items = transaction.split(",")
        lista = []
        for i in range(0, len(items)):
            x = AlphaCortes(float(items[i]), broadcastParam.value[0])
            lista.append((str(broadcastParam.value[4][i]), x))  # .copy()))
        return lista


class ARE(object):

    def cargar_datos(path="/home/carlos/Gits/Gitlab/AR_SPARK/Result/Angel Rules/0.003Items.dat"):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, IOError) as e:
            return dict()

    def guardar_datos(dic):
        with open("Items.dat", "wb") as f:
            pickle.dump(dic, f)

    minConf = 0.70
    setOfRules = []
    LisOfDict = cargar_datos("/home/carlos/Gits/Gitlab/AR_SPARK/0.001Items.dat")
    # /home/carlos/Gits/Gitlab/AR_SPARK/0.003Items.dat

    ListFrozensets = []
    for i in (range(0, len(LisOfDict))):
        d = LisOfDict[i]
        Dicaux = {}
        for key, value in d.items():
            Dicaux[frozenset(key.split("-"))] = value
        ListFrozensets.append(Dicaux.copy())
    # print(ListFrozensets[8])
    for i in numpy.arange(1, len(ListFrozensets), 1)[::-1]:
        # print("I:" + str(i))
        D = ListFrozensets[i]
        for j in D.keys():
            # print("\n\n\nJ " + str(j))
            # print("\n\n\nD " + str(D))
            # j=frozenset(j.split("-"))
            value = D[j]
            for s in numpy.arange(1, len(ListFrozensets), 1):
                for d in (itertools.combinations(j, s)):
                    # print("\nJ:"+str(j))
                    # print("S:"+str(s)+" D: "+str(d))
                    x = j.difference(frozenset(list(d)))
                    if x != frozenset():
                        # if s==1:
                        #     key=str(list(d)[0])
                        # else:
                        key = frozenset(list(d))
                        key2 = frozenset(list(x))
                        Confidence = value / ListFrozensets[s - 1][key]
                        lift = value / (ListFrozensets[s - 1][key] * ListFrozensets[len(list(x)) - 1][key2])
                        Rule1 = [(','.join(list(x))) + "-->" + (','.join(list(d))), [Confidence]]  # , lift]
                        if Confidence > minConf:
                            print(Rule1)

    ###########################################################################
    ###########################################################################
    ###########################################################################
    exit()
    if 8 == 8:
        for key, value in d.items():
            keys = key.split("-")
            if (i != 1):
                Ban = False
                listapermu = []
                for numAux in range(i, len(keys)):
                    listapermu = listapermu + (list(itertools.combinations(keys, numAux)))
                listapermu = list(map('-'.join, listapermu))
            else:
                Ban = True
                listapermu = keys
            # print("\n\n Value" + str(value) + "\n\n")
            for p in range(0, len(listapermu)):
                if Ban != True:
                    pcom = listapermu[p].split("-")
                    for k in keys:
                        if listapermu[p] != k and not (k in pcom):
                            id = len(listapermu[p].split("-")) - 1
                            keysearch = listapermu[p]
                            if keysearch in LisOfDict[id].keys():
                                Confidence = value / LisOfDict[id][keysearch]
                            else:
                                # print("\n\nERROR"+str(keysearch))
                                Confidence = 0
                            Rule1 = [(listapermu[p]) + "-->" + (k), Confidence]
                            id = len(k.split("-")) - 1
                            keysearch = k
                            Confidence = value / LisOfDict[id][keysearch]
                            Rule2 = [(k) + "-->" + (listapermu[p]), Confidence]
                            if (Rule1[1] > MinSupport):
                                setOfRules.append(Rule1)
                                # print(Rule1)
                                # print(value)
                                # print(LisOfDict[id][keysearch])
                                # print(keysearch)
                                # print(Confidence)
                            if (Rule2[1] > MinSupport):
                                setOfRules.append(Rule2)
                                # print(Rule2)
                                # print(value)
                                # print(LisOfDict[id][keysearch])
                                # print(Confidence)
                else:
                    for k in range(1, len(keys)):
                        if listapermu[p] != keys[k]:
                            id = len(listapermu[p].split("-")) - 1
                            keysearch = listapermu[p]
                            Confidence = value / LisOfDict[id][keysearch]
                            # print(LisOfDict[id][keysearch])
                            # print(Confidence)
                            Rule1 = [(listapermu[p]) + "-->" + (keys[k]), Confidence]
                            id = len(keys[k].split("-")) - 1
                            keysearch = keys[k]
                            Confidence = value / LisOfDict[id][keysearch]
                            Rule2 = [(keys[k]) + "-->" + (listapermu[p]), Confidence]
                            # print(Rule1)
                            if (Rule1[1] > MinSupport):
                                setOfRules.append(Rule1)
                                # print(Rule1)
                                # print(value)
                                # print(LisOfDict[id][keysearch])
                                # print(Confidence)
                            if (Rule2[1] > MinSupport):
                                setOfRules.append(Rule2)
                                # print(Rule2)
                                # print(value)
                                # print(LisOfDict[id][keysearch])
                                # print(Confidence)
    print(setOfRules)
