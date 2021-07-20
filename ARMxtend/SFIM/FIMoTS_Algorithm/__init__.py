#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Algoritmo para la obtención e identificación eficiente de itemsets frecuentes en ventanas de tiempo
sensibles al tiempo (deslizantes), y en ambientes de flujo constante de datos (streams).

La concepción inicial del algoritmo se encuentra en el artículo "Efficient frequent itemset mining
methods over time-sensitive streams", publicado en la revista Knowledge-Based Systems, en 2014.
 
@author Haifeng Li, Ning Zhang, Jianming Zhu, Huaihu Cao, Yue Wang (Especificación - 2014)
		Abel Francisco (Implementación - 2016)  
"""

import sys
import os
import time
import datetime

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

from math import floor

from ..DecimalFraction import DecimalFraction
from ..FIMoTS_Node import FIMoTS_Node
from ..FIMoTS_Bounds import FIMoTS_Bounds
from ..FIMoTS_List import FIMoTS_List
from ..FIMoTS_Tree import FIMoTS_Tree
from ..FIMoTS_Structure import FIMoTS_Structure


def FIMoTS_TreeInitialization(initialStructure, slidingWindow, minRelSupportElems, sparkContext):
    """
	Construcción del árbol FP-Tree asociado a la ventana de tiempo inicial.
	
	Argumentos:
		initialStructure (FIMoTS_Structure): una estructura vacía para ser inicializada y 
			utilizada durante la aplicación y resolución del algoritmo FIMoTS
		slidingWindow (WindowData, obligatorio): el conjunto de transacciones que integran a la
			ventana de tiempo
		minRelSupportElems (DecimalFraction, obligatorio): el soporte mínimo con el que
			un itemset puede ser considerado frecuente, expresado como una fracción de 
			números enteros
		sparkContext(SparkContext): contexto distribuido para aplicar técnicas de BigData
			en un RDD con Spark
	
	Retorna:
		Una estructura para la resolución y aplicación del algoritmo FIMoTS, conteniendo
		un árbol FP-Tree con el conteo inicial de itemsets en la ventana de tiempo
		original.
    """
    minSupp = minRelSupportElems.numeratorFraction / float(minRelSupportElems.denominatorFraction)
    currentTree = FIMoTS_GetFirstLevelTree(slidingWindow, sparkContext)
    initialStructure.itemsetsTree = currentTree
    initialStructure.frequentItemsetsBounds = FIMoTS_List([])
    initialStructure.infrequentItemsetsBounds = FIMoTS_List([])
    currentLevel = 1

    # Construir de forma iterativa los nuevos niveles del árbol, siempre y 
    # cuando existan nodos frecuentes candidatos
    while initialStructure.itemsetsTree.getLevelTree(currentLevel):
        FIMoTS_BuildLevelTree(initialStructure, slidingWindow, minRelSupportElems, currentLevel, sparkContext)
        currentLevel += 1


def FIMoTS_GetFirstLevelTree(slidingWindow, sparkContext):
    """
    Armar el primer nivel del árbol lexicográfico FP, realizando un conteo inicial de 
    items (soportes relativos) sobre la primera ventana de tiempo.
	
    Argumentos:
        slidingWindow (WindowData, obligatorio): el conjunto de transacciones que integran a la
        ventana de tiempo
			
    Retorna:
        Un árbol lexicográfico, representado como un Diccionario o HashMap, con los
        items (vocabulario) frecuentes identificados en la ventana de tiempo (nivel 1 del
        árbol).
    """
    transactions = slidingWindow.getUnionTransactions(sparkContext)
    transacsTotal = transactions.count()

    # Dividir cada línea del RDD en palabras
    items = transactions.flatMap(lambda transaction: transaction.split(" "))

    # Contar cada palabra en cada subconjunto
    itemPairs = items.map(lambda item: (item, 1))
    itemCounts = itemPairs.reduceByKey(lambda x, y: x + y).collect()

    # Crear el Diccionario FP-Tree
    fimotsTree = FIMoTS_Tree({})

    # Crear el Nivel 0 del árbol (Raíz)
    fractionElems = DecimalFraction(-1, transacsTotal)
    rootNode = FIMoTS_Node([], -1.0, fractionElems, None, len(itemCounts), slidingWindow.timeWindow)
    fimotsTree.nodeMap['-1'] = rootNode

    # Crear Nivel 1 del árbol (items - itemsets de tamaño 1)
    for (item, countItem) in itemCounts:
        fractionElems = DecimalFraction(countItem, transacsTotal)
        node = FIMoTS_Node([item], countItem / float(transacsTotal), fractionElems, '-1', 0, slidingWindow.timeWindow)
        fimotsTree.nodeMap[node.getItemsetKey()] = node

    return fimotsTree


def FIMoTS_BuildLevelTree(algStructure, slidingWindow, minRelSupportElems, currentLevel, sparkContext):
    """
	Construir un nuevo nivel de itemsets frecuentes en el árbol FP-Tree inicial.
	
	Argumentos:
		algStructure (FIMoTS_Structure, obligatorio): estructura actual para la ejecución del
			algoritmo FIMoTS
		slidingWindow (WindowData, obligatorio): el conjunto de transacciones que integran a la
			ventana de tiempo
		minRelSupportElems (DecimalFraction, obligatorio): el soporte mínimo con el que
			un itemset puede ser considerado frecuente, expresado como una fracción de 
			números enteros
		currentLevel (int, obligatorio): nivel actual máximo del árbol FP
    """
    transactions = slidingWindow.getUnionTransactions(sparkContext)
    transacsTotal = slidingWindow.totalWindowTransacs

    itemsetsCandidates = []
    itemsetsCounts = []

    # Ordenar lexicográficamente nodos del nivel máximo actual del árbol
    mapCurrentLevel = sorted(algStructure.itemsetsTree.getLevelTree(currentLevel),
                             key=algStructure.itemsetsTree.getLevelTree(currentLevel).get)

    # Generar descendencia de cada nodo hoja actual (nodos en el máximo nivel)
    for keyItemsetNode in mapCurrentLevel:
        itemsetsCandidates += FIMoTS_Initial(keyItemsetNode, algStructure.itemsetsTree.nodeMap[keyItemsetNode],
                                             algStructure, slidingWindow, minRelSupportElems, True, True, sparkContext)

    # Si se ha generado un nuevo nivel del árbol
    if algStructure.itemsetsTree.getLevelTree(currentLevel + 1):

        # Calcular los soportes de los nuevos itemsets hojas
        itemsetsCounts = FIMoTS_CalculateSupportLevelsTree(itemsetsCandidates, transactions)

        # Armar y agregar nodos en el Diccionario para cada itemset nuevo
        for (itemsetKey, countItemset) in itemsetsCounts:
            fractionElems = DecimalFraction(countItemset, transacsTotal)
            currentNode = algStructure.itemsetsTree.nodeMap[itemsetKey]
            currentNode.relativeSupport = countItemset / float(transacsTotal)
            currentNode.fractionElems = fractionElems


def FIMoTS_Initial(keyCurrentItemNode, currentItemNode, algStructure, slidingWindow, minRelSupportElems,
                   recalculateList, isTreeInitialization, sparkContext):
    """
	Método para desplegar un nodo que represente un itemset frecuente en el árbol lexicográfico,
	generando su descendencia, es decir, los nodos hijos que representan superconjuntos de
	dicho itemset.
	
	Argumentos:
	
		keyCurrentItemNode (String, obligatorio): clave que identifica al nodo en el Diccionario/Árbol
		currentItemNode (FIMoTS_Node, obligatorio): contenido del nodo al que se debe desplegar 
			su descendencia
		algStructure (FIMoTS_Structure, obligatorio): estructura actual con la ejecución del 
			algoritmo FIMoTS
		slidingWindow (WindowData, obligatorio): el conjunto de transacciones que integran a la
			ventana de tiempo
		minRelSupportElems (DecimalFraction, obligatorio): mínimo Soporte relativo, 
			especificado como una fracción de enteros
		recalculateList (Boolean, obligatorio): indica si se deben recalcular las cotas del 
			nodo actual, así como su ubicación en la lista de claves de cotas transformadoras
		isTreeInitialization (Boolean, obligatorio): indica si se está realizando el proceso inicial
			de construcción del árbol lexicográfico, o se está actualizando el mismo en tiempo
			real. Si es la inicialización del árbol, solo se debe generar el primer nivel de
			la descendencia del nodo y no se debe calcular su soporte, ya que dicho despliegue
			y cálculos se realizan en el proceso externo de construcción de niveles del árbol.
		sparkContext(SparkContext): contexto distribuido para aplicar técnicas de BigData
			en un RDD con Spark
			
	Retorna:
		Lista de nodos hijos generados a partir del nodo padre recibido como argumento
    """
    minRelSupport = minRelSupportElems.numeratorFraction / float(minRelSupportElems.denominatorFraction)
    transactions = slidingWindow.getUnionTransactions(sparkContext)
    transacTotal = slidingWindow.totalWindowTransacs
    currentLevel = len(currentItemNode.itemPrefix)

    # Si el itemset es infrecuente
    if currentItemNode.relativeSupport < minRelSupport:

        # Actualizar cotas transformadoras del nodo en listas
        FIMoTS_UpdateBoundsList(currentItemNode, algStructure, minRelSupportElems, transacTotal)

        return []

    # Si el itemset actual es frecuente
    else:
        rightSiblingsCurrentNode = algStructure.itemsetsTree.getRightSiblings(keyCurrentItemNode, currentItemNode)
        newLevelCandidates = []

        # Para cada nodo hermano derecho
        for keyRightSibling, rightSibling in rightSiblingsCurrentNode.iteritems():

            # Si el itemset asociado al nodo hermano es frecuente
            if rightSibling.relativeSupport >= minRelSupport:
                itemPrefixNewNode = sorted(list(set(currentItemNode.itemPrefix + rightSibling.itemPrefix)))
                newLevelCandidates.append(itemPrefixNewNode)

        newNodes = []

        for itemsetCandidate in newLevelCandidates:
            newNode = FIMoTS_Node(itemsetCandidate, 0.0, None, currentItemNode.getItemsetKey(), 0, 0)
            newNodes.append(newNode)

        itemsetsCounts = []

        # Calcular soporte de nuevos nodos hijos si se está
        # actualizando el árbol (no inicialización)
        # Si es inicialización, el cálculo de los soportes se
        # realiza en el método externo FIMoTS_BuildLevelTree
        if not isTreeInitialization:
            itemsetsCounts = FIMoTS_CalculateSupportLevelsTree(newNodes, transactions)

        for nodeChild in newNodes:
            if not isTreeInitialization:
                currentItemTransac = next(countItemset for (itemsetKey, countItemset) in itemsetsCounts if
                                          itemsetKey == nodeChild.getItemsetKey())
                nodeChild.relativeSupport = (float(currentItemTransac) / float(transacTotal))
                nodeChild.fractionElems = DecimalFraction(currentItemTransac, transacTotal)

            nodeChild.lastUpdate = slidingWindow.timeWindow
            nodeChild.orderItemPrefix()
            algStructure.itemsetsTree.nodeMap[nodeChild.getItemsetKey()] = nodeChild

        # Actualizar valores de identificación de hijos en el nodo padre
        currentItemNode.totalChildNodes = len(newNodes)

        if recalculateList:
            # Actualizar cotas transformadoras del nodo en listas
            FIMoTS_UpdateBoundsList(currentItemNode, algStructure, minRelSupportElems, transacTotal)

        if not isTreeInitialization:
            for newChildNode in newNodes:
                FIMoTS_Initial(newChildNode.getItemsetKey(), newChildNode, algStructure, slidingWindow,
                               minRelSupportElems, True, False, sparkContext)

        return newNodes


def FIMoTS_UpdateBoundsList(currentItemNode, algStructure, minSupportElems, windowSize):
    """
	Actualizar las listas con las cotas de transformación de un nodo/itemset en la 
	estructura de solución del algoritmo.
	
	Argumentos:
		currentItemNode (FIMoTS_Node, obligatorio): nodo al cual se le deben recalcular sus
			cotas, así como su ubicación en las listas enlazadas de cotas de itemsets
		algStructure (FIMoTS_Structure, obligatorio): estructura actual para la ejecución del
			algoritmo FIMoTS
		minSupportElems (DecimalFraction, obligatorio): el soporte mínimo con el que
			un itemset puede ser considerado frecuente, expresado como una fracción de 
			números enteros
		windowSize (int, obligatorio): total de transacciones que integran la ventana de
			tiempo actual
	"""
    # Calcular nuevos límites transformadores del nodo
    nodeSupportElems = currentItemNode.fractionElems
    minSupport = float(minSupportElems.numeratorFraction) / float(minSupportElems.denominatorFraction)
    nodeItemsetKey = currentItemNode.getItemsetKey()
    nodeNewBounds = FIMoTS_Bounds()

    # Si el itemset es itemset infrecuente
    if currentItemNode.relativeSupport < minSupport:
        nodeUpperBound = FIMoTS_CalculateUpperBound(minSupportElems, nodeSupportElems, windowSize, False)
        nodeLowerBound = FIMoTS_CalculateLowerBound(minSupportElems, nodeSupportElems, windowSize, False)
        nodeNewBounds = FIMoTS_Bounds(nodeUpperBound, nodeLowerBound, [nodeItemsetKey])

    # Si el itemset es frecuente
    else:
        nodeUpperBound = FIMoTS_CalculateUpperBound(minSupportElems, nodeSupportElems, windowSize, True)
        nodeLowerBound = FIMoTS_CalculateLowerBound(minSupportElems, nodeSupportElems, windowSize, True)
        nodeNewBounds = FIMoTS_Bounds(nodeUpperBound, nodeLowerBound, [nodeItemsetKey])

    currentItemList = FIMoTS_List()

    # Generar elemento correspondiente en lista de cotas (ajustar apuntadores)
    # Si el itemset se mantiene como itemset infrecuente
    if currentItemNode.relativeSupport < minSupport:
        currentItemList = algStructure.infrequentItemsetsBounds

    # Si el itemset infrecuente ha cambiado a frecuente
    else:
        currentItemList = algStructure.frequentItemsetsBounds

    equalBounds = None
    findEqualBounds = False

    # Buscar otro elemento en la lista de cotas transformadoras con la misma combinación
    # de límite superior - inferior
    for bounds in currentItemList.itemsBounds:

        if nodeNewBounds.upperBound == bounds.upperBound and nodeNewBounds.lowerBound == bounds.lowerBound:
            # Si se encuentra un elemento equivalente
            equalBounds = bounds
            findEqualBounds = True
            break

    # Si ya existe una combinación de cotas equivalente en la lista
    if findEqualBounds:

        # Agregar nodo actual al final de la lista de
        # apuntadores de la cota equivalente
        equalBounds.itemNodes.append(nodeItemsetKey)

    # Si es un nuevo par de cotas superior/inferior
    else:

        # Agregar nuevo elemento a lista de cotas distintas de
        # itemsets frecuentes
        currentItemList.itemsBounds.append(nodeNewBounds)


def FIMoTS_CalculateUpperBound(relSupportElems, currentSupportElems, windowSize, isFrequent):
    """
	Calcular nuevo valor de la Cota de Transformación Superior de un itemset
	 
	relSupportElems (DecimalFraction, obligatorio): mínimo Soporte relativo, especificado 
		como una fracción de enteros
	currentSupportElems (DecimalFraction, obligatorio): soporte actual del itemset, 
		especificado como una fracción de enteros
	windowSize (int, obligatorio): cantidad de Transacciones en la ventana deslizante actual
	isFrequent (Boolean, obligatorio): indica si el itemset sobre el que se aplica el 
		cálculo es frecuente, o no
	
	Retorna: 
		El nuevo valor de la Cota Transformadora Superior del itemset
	"""
    computeUpperBound = 0.0

    if isFrequent:
        computeUpperBound1 = float(currentSupportElems.numeratorFraction * relSupportElems.denominatorFraction) / float(
            currentSupportElems.denominatorFraction * relSupportElems.numeratorFraction)
        computeUpperBound = computeUpperBound1 * windowSize
    else:
        computeUpperBound1 = float(relSupportElems.denominatorFraction) / float(currentSupportElems.denominatorFraction)
        computeUpperBound2 = float(
            currentSupportElems.denominatorFraction - currentSupportElems.numeratorFraction) / float(
            relSupportElems.denominatorFraction - relSupportElems.numeratorFraction)
        computeUpperBound = computeUpperBound1 * computeUpperBound2 * windowSize

    newUpperBound = computeUpperBound - windowSize + 1

    if isFrequent:
        return int(floor(newUpperBound));
    else:
        if newUpperBound == rint(newUpperBound):
            return int(newUpperBound - 1)
        else:
            return int(floor(newUpperBound))


def FIMoTS_CalculateLowerBound(relSupportElems, currentSupportElems, windowSize, isFrequent):
    """
	Calcular nuevo valor de la Cota de Transformación Inferior de un itemset
	 
	relSupportElems (DecimalFraction, obligatorio): mínimo Soporte relativo, especificado 
		como una fracción de enteros
	currentSupportElems (DecimalFraction, obligatorio): Soporte actual del itemset, 
		especificado como una fracción de enteros
	windowSize (int, obligatorio): cantidad de Transacciones en la ventana deslizante actual
	isFrequent (Boolean, obligatorio): indica si el itemset sobre el que se aplica el cálculo es frecuente, o no
	
	Retorna:
		El nuevo valor de la Cota Transformadora Inferior del itemset
	"""
    computeLowerBound = 0.0

    if isFrequent:
        computeLowerBound1 = float(relSupportElems.denominatorFraction) / float(currentSupportElems.denominatorFraction)
        computeLowerBound2 = float(
            currentSupportElems.denominatorFraction - currentSupportElems.numeratorFraction) / float(
            relSupportElems.denominatorFraction - relSupportElems.numeratorFraction)
        computeLowerBound = computeLowerBound1 * computeLowerBound2 * windowSize
    else:
        computeLowerBound1 = float(currentSupportElems.numeratorFraction * relSupportElems.denominatorFraction) / float(
            currentSupportElems.denominatorFraction * relSupportElems.numeratorFraction)
        computeLowerBound = computeLowerBound1 * windowSize

    newLowerBound = windowSize - computeLowerBound + 1

    if isFrequent:
        return int(floor(newLowerBound))
    else:
        if newLowerBound == rint(newLowerBound):
            return int(newLowerBound - 1)
        else:
            return int(floor(newLowerBound))


def rint(num):
    """
    Retorna el valor decimal que es el más cercano al argumento recibido y es igual a un
    número entero.
	
    Argumentos:
        num (float, obligatorio): El valor numérico a redondear a un entero
		
    Retorna:
        El valor decimal que es el más cercano al argumento recibido y es igual a un
        número entero.
    """
    return round(num + (num % 2 - 1 if (num % 1 == 0.5) else 0))


def FIMoTS_CalculateSupportLevelsTree(itemsetsCandidates, transactions):
    """
    Calcular los soportes relativos de un conjunto de itemsets candidatos, sobre un
    grupo de transacciones que integran una ventana de tiempo.
	
    Argumentos:
        itemsetsCandidates(Array(FIMoTS_Node), obligatorio): lista de nodos del árbol
            en los cuales se busca actualizar o calcular su soporte en un conjunto de
            transacciones
        transactions (RDD, obligatorio): el conjunto de transacciones que integran a la
            ventana de tiempo
			
    Retorna:
        Una lista con los pares de (claves,conteos) asociados con los itemsets candidatos
        recibidos como argumentos.
    """
    # Dividir cada línea del RDD en palabras
    transactionsAsList = transactions.map(lambda transaction: transaction.split(" "))

    # Contar cada palabra en cada subconjunto
    itemsetPairs = transactionsAsList.flatMap(
        lambda transaction: getItemsetsInTransaction(transaction, itemsetsCandidates))
    itemsetCounts = itemsetPairs.reduceByKey(lambda x, y: x + y).collect()

    return itemsetCounts


def getItemsetsInTransaction(transaction, itemsetsCandidates):
    """
	Función auxiliar para determinar si un itemset se encuentra contenido en una transacción
	de items determinada.
	
	Argumentos:
		itemsetsCandidates(Array(FIMoTS_Node), obligatorio): lista de nodos del árbol
			en los cuales se busca determinar si está presente en la transacción
		transaction (Array(String), obligatorio): una transacción de items
			
	Retorna:
		Una lista con los pares de (clave,1) para cada itemset que se encuentre contenido
		en la transacción, o (clave,0) si el itemset no se encuentra contenido en la
		transacción
    """
    itemsetsInTransaction = []
    for itemset in itemsetsCandidates:
        if set(transaction).issuperset(set(itemset.itemPrefix)):
            itemsetsInTransaction.append((itemset.getItemsetKey(), 1))
        else:
            itemsetsInTransaction.append((itemset.getItemsetKey(), 0))

    return itemsetsInTransaction


def FIMoTS_Main(deletedTransacSize, addedTransacSize, algStructure, slidingWindow, minRelSupportElems, sparkContext):
    """
	Programa principal del algoritmo FIMoTS, en el que se actualiza el estado de los itemsets frecuentes
	e infrecuentes, en base a la actualización de una ventana de datos deslizante en el tiempo. La
	actualización depende de los cambios ocurridos en las cotas de transformación superiores e inferiores
	de cada itemset identificado.
	
	deletedTransacSize (int, obligatorio): total de transacciones eliminadas de la ventana 
	    de tiempo
	addedTransacSize (int, obligatorio): total de transacciones agregadas a la ventana de 
	    tiempo
	algStructure (FIMoTS_Structure, obligatorio): estructura actual con la ejecución del 
		algoritmo FIMoTS
	slidingWindow (WindowData, obligatorio): Ventana deslizante de tiempo actualizada
	minRelSupportElems (DecimalFraction, obligatorio): Mínimo Soporte relativo, especificado 
		como una fracción de enteros
	sparkContext(SparkContext): contexto distribuido para aplicar técnicas de BigData
		en un RDD con Spark
    """
    relSupportElems = minRelSupportElems
    minRelSupport = float(minRelSupportElems.numeratorFraction) / float(minRelSupportElems.denominatorFraction)

    transactions = slidingWindow.getUnionTransactions(sparkContext)
    windowSize = slidingWindow.totalWindowTransacs

    # Estudiar lista de itemset frecuentes
    for currentBounds in algStructure.frequentItemsetsBounds.itemsBounds:
        # Actualizar Cota Inferior de Transformación (Lower Transformation Bound)
        currentLowerBound = currentBounds.lowerBound
        currentBounds.lowerBound = FIMoTS_UpdateLowerBound(currentLowerBound, relSupportElems, deletedTransacSize,
                                                           addedTransacSize, True)

        # Actualizar Cota Superior de Transformación (Upper Transformation Bound)
        currentUpperBound = currentBounds.upperBound
        currentBounds.upperBound = FIMoTS_UpdateUpperBound(currentUpperBound, relSupportElems, deletedTransacSize,
                                                           addedTransacSize, True)

    # Estudiar lista de itemset infrecuentes
    for currentBounds in algStructure.infrequentItemsetsBounds.itemsBounds:
        # Actualizar Cota Inferior de Transformación (Lower Transformation Bound)
        currentLowerBound = currentBounds.lowerBound
        currentBounds.lowerBound = FIMoTS_UpdateLowerBound(currentLowerBound, relSupportElems, deletedTransacSize,
                                                           addedTransacSize, False)

        # Actualizar Cota Superior de Transformación (Upper Transformation Bound)
        currentUpperBound = currentBounds.upperBound
        currentBounds.upperBound = FIMoTS_UpdateUpperBound(currentUpperBound, relSupportElems, deletedTransacSize,
                                                           addedTransacSize, False)

    nodesUpdated = []
    itemsetsCounts = []

    # Actualizar itemsets frecuentes con cotas de transformación igual o menores que cero
    for currentBounds in algStructure.frequentItemsetsBounds.itemsBounds:

        if (currentBounds.lowerBound <= 0) or (currentBounds.upperBound <= 0):

            # Mientras exista otro nodo con la misma combinación de cota superior - inferior
            for currentFreqItemKey in currentBounds.itemNodes:
                freqNodeUpdated = algStructure.itemsetsTree.nodeMap.get(currentFreqItemKey)
                nodesUpdated.append(freqNodeUpdated)

            # Eliminar todos los itemsets ya estudiados de la lista de frecuentes (con sus cotas originales)
            currentBounds.itemNodes = []

    itemsetsCounts = FIMoTS_CalculateSupportLevelsTree(nodesUpdated, transactions)

    for (currentFreqItemKey, currentFreqItemCount) in itemsetsCounts:
        currentFreqItemNode = algStructure.itemsetsTree.nodeMap.get(currentFreqItemKey)

        if currentFreqItemNode is not None:

            # Actualizar soporte del itemset actual
            currentFreqItemNode.relativeSupport = (float(currentFreqItemCount) / float(windowSize))
            currentFreqItemNode.fractionElems = DecimalFraction(currentFreqItemCount, windowSize)
            currentFreqItemNode.lastUpdate = slidingWindow.timeWindow

            # Actualizar cotas transformadoras del nodo en listas
            FIMoTS_UpdateBoundsList(currentFreqItemNode, algStructure, relSupportElems, windowSize)

            # Si el itemset frecuente ha cambiado a infrecuente
            if currentFreqItemNode.relativeSupport < minRelSupport:
                # Podar los sub-árboles hijos del nodo
                algStructure.deleteChildNodes(currentFreqItemKey)

        nodesUpdated = []

    # Actualizar itemsets infrecuentes con cotas de transformación igual o menores que cero
    for currentBounds in algStructure.infrequentItemsetsBounds.itemsBounds:

        if (currentBounds.lowerBound <= 0) or (currentBounds.upperBound <= 0):

            # Mientras exista otro nodo con la misma combinación de cota superior - inferior
            for currentInfreqItemKey in currentBounds.itemNodes:
                infreqNodeUpdated = algStructure.itemsetsTree.nodeMap.get(currentInfreqItemKey)
                nodesUpdated.append(infreqNodeUpdated)

            # Eliminar todos los itemsets ya estudiados de la lista de infrecuentes (con sus cotas originales)
            currentBounds.itemNodes = []

    itemsetsCounts = FIMoTS_CalculateSupportLevelsTree(nodesUpdated, transactions)

    newFrequentItemsKeys = []

    for (currentInfreqItemKey, currentInfreqItemCount) in itemsetsCounts:
        currentInfreqItemNode = algStructure.itemsetsTree.nodeMap.get(currentInfreqItemKey)

        if currentInfreqItemNode is not None:

            # Actualizar soporte del itemset actual
            currentInfreqItemNode.relativeSupport = float(currentInfreqItemCount) / float(windowSize)
            currentInfreqItemNode.fractionElems = DecimalFraction(currentInfreqItemCount, windowSize)
            currentInfreqItemNode.lastUpdate = slidingWindow.timeWindow

            # Actualizar cotas transformadoras del nodo en listas
            FIMoTS_UpdateBoundsList(currentInfreqItemNode, algStructure, relSupportElems, windowSize)

            # Si el itemset infrecuente ha cambiado a frecuente
            if currentInfreqItemNode.relativeSupport >= minRelSupport:
                newFrequentItemsKeys.append(currentInfreqItemNode.getItemsetKey())

    # Generar la descendencia de todos los nuevos nodos frecuentes
    for newFreqItemKey in newFrequentItemsKeys:
        newFreqItemNode = algStructure.itemsetsTree.nodeMap.get(newFreqItemKey)
        FIMoTS_Initial(newFreqItemKey, newFreqItemNode, algStructure, slidingWindow, minRelSupportElems, False, False,
                       sparkContext)

    algStructure.frequentItemsetsBounds.clearEmptyBounds()
    algStructure.infrequentItemsetsBounds.clearEmptyBounds()


def FIMoTS_UpdateLowerBound(currentLowerBound, relSupportElems, deletedTransacsSize, addedTransacsSize, isFrequent):
    """
	Actualizar el valor de la Cota Transformadora Inferior de un itemset.
	 
	currentLowerBound (int, obligatorio): el valor de la cota transformadora inferior actual
	relSupportElems (DecimalFraction, obligatorio): mínimo Soporte relativo, especificado 
		como una fracción de enteros
	deletedTransacsSize (int, obligatorio): Cantidad de transacciones eliminadas de la 
		ventana de tiempo
	addedTransacsSize (int, obligatorio): total de transacciones agregadas a la ventana 
		deslizante
	isFrequent (Boolean, obligatorio): indica si el itemset sobre el que se aplica la 
		actualización es frecuente, o no
	
	Retorna:
		El valor actualizado de la Cota Transformadora Inferior del itemset
	"""
    lowerBoundUpdate = 0.0

    if isFrequent:
        lowerBoundUpdate = (float(relSupportElems.numeratorFraction) / float(
            relSupportElems.denominatorFraction - relSupportElems.numeratorFraction)) * addedTransacsSize
    else:
        lowerBoundUpdate = (float(relSupportElems.denominatorFraction - relSupportElems.numeratorFraction) / float(
            relSupportElems.numeratorFraction)) * addedTransacsSize

    return int(floor(currentLowerBound - deletedTransacsSize - lowerBoundUpdate))


def FIMoTS_UpdateUpperBound(currentUpperBound, relSupportElems, deletedTransacsSize, addedTransacsSize, isFrequent):
    """
	Actualizar el valor de la Cota Transformadora Superior de un itemset.
	 
	currentUpperBound (int, obligatorio): el valor de la cota transformadora superior actual
	relSupportElems (DecimalFraction, obligatorio): mínimo Soporte relativo, especificado 
		como una fracción de enteros
	deletedTransacsSize (int, obligatorio): cantidad de transacciones eliminadas de la 
		ventana de tiempo
	addedTransacsSize (int, obligatorio): total de transacciones agregadas a la ventana 
		deslizante
	isFrequent (Boolean, obligatorio): indica si el itemset sobre el que se aplica la 
		actualización es frecuente, o no
	
	Retorna:
		El valor actualizado de la Cota Transformadora Superior del itemset
	"""
    upperBoundUpdate = 0.0

    if isFrequent:
        upperBoundUpdate = (float(relSupportElems.denominatorFraction - relSupportElems.numeratorFraction) / float(
            relSupportElems.numeratorFraction)) * deletedTransacsSize
    else:
        upperBoundUpdate = (float(relSupportElems.numeratorFraction) / float(
            relSupportElems.denominatorFraction - relSupportElems.numeratorFraction)) * deletedTransacsSize

    return int(floor(currentUpperBound - upperBoundUpdate - addedTransacsSize))
