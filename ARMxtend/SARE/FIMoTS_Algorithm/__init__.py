#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Algorithm for the efficient obtention and identification of frequent itemsets in time-sensitive (sliding) time
windows, and in constant data stream environments (streams).

The initial conception of the algorithm is found in the paper "Efficient frequent itemset mining
methods over time-sensitive streams", published in Knowledge-Based Systems, in 2014.

@author Haifeng Li, Ning Zhang, Jianming Zhu, Huaihu Cao, Yue Wang (Specification - 2014)
        Abel Francisco (Implementation - 2016)
"""

from math import floor

from ..DecimalFraction import DecimalFraction
from ..FIMoTS_Node import FIMoTS_Node
from ..FIMoTS_Bounds import FIMoTS_Bounds
from ..FIMoTS_List import FIMoTS_List
from ..FIMoTS_Tree import FIMoTS_Tree


def FIMoTS_TreeInitialization(initialStructure, slidingWindow, minRelSupportElems, sparkContext):
    """
    Construction of the FP-Tree associated with the initial time window.

    Args:
        initialStructure (FIMoTS_Structure): an empty structure to be initialized and used during the application
            and resolution of the FIMoTS algorithm
        slidingWindow (WindowData, required): the set of transactions that make up the time window
        minRelSupportElems (DecimalFraction, required): the minimum support with which an itemset can be considered
            frequent, expressed as a fraction of integers
        sparkContext(SparkContext): distributed context to apply Big Data techniques on an RDD with Spark

    Returns:
        A structure for the resolution and application of the FIMoTS algorithm, containing an FP-Tree with the
        initial count of itemsets in the original time window.
    """
    currentTree = FIMoTS_GetFirstLevelTree(slidingWindow, sparkContext)
    initialStructure.itemsetsTree = currentTree
    initialStructure.frequentItemsetsBounds = FIMoTS_List([])
    initialStructure.infrequentItemsetsBounds = FIMoTS_List([])
    currentLevel = 1

    # Iteratively build the new levels of the tree, as long as
    # cuando existan nodos frecuentes candidatos
    while initialStructure.itemsetsTree.getLevelTree(currentLevel):
        FIMoTS_BuildLevelTree(initialStructure, slidingWindow, minRelSupportElems, currentLevel, sparkContext)
        currentLevel += 1


def FIMoTS_GetFirstLevelTree(slidingWindow, sparkContext):
    """
    Build the first level of the lexicographic FP tree, performing an initial count of items (relative supports) over
    the first time window.

    Args:
        slidingWindow (WindowData, required): the set of transactions that make up the time window

    Returns:
        A lexicographic tree, represented as a Dictionary or HashMap, with the frequent items (vocabulary) identified
        in the time window (level 1 of the tree).
    """
    transactions = slidingWindow.getUnionTransactions(sparkContext)
    transacsTotal = transactions.count()

    # Split each line of the RDD into words
    items = transactions.flatMap(lambda transaction: transaction.split())

    # Count each word in each subset
    itemPairs = items.map(lambda item: (item, 1))
    itemCounts = itemPairs.reduceByKey(lambda x, y: x + y).collect()

    # Crear el Diccionario FP-Tree
    fimotsTree = FIMoTS_Tree({})

    # Create Level 0 of the tree (Root)
    fractionElems = DecimalFraction(-1, transacsTotal)
    rootNode = FIMoTS_Node([], -1.0, fractionElems, None, len(itemCounts), slidingWindow.timeWindow)
    fimotsTree.nodeMap['-1'] = rootNode

    # Create Level 1 of the tree (items - itemsets of size 1)
    for (item, countItem) in itemCounts:
        fractionElems = DecimalFraction(countItem, transacsTotal)
        node = FIMoTS_Node([item], countItem / float(transacsTotal), fractionElems, '-1', 0, slidingWindow.timeWindow)
        fimotsTree.nodeMap[node.getItemsetKey()] = node

    return fimotsTree


def FIMoTS_BuildLevelTree(algStructure, slidingWindow, minRelSupportElems, currentLevel, sparkContext):
    """
    Build a new level of frequent itemsets in the initial FP-Tree.

    Args:
        algStructure (FIMoTS_Structure, required): current structure for the execution of the FIMoTS algorithm
        slidingWindow (WindowData, required): the set of transactions that make up the time window
        minRelSupportElems (DecimalFraction, required): the minimum support with which an itemset can be considered
            frequent, expressed as a fraction of integers
        currentLevel (int, required): current maximum level of the FP tree
    """
    transactions = slidingWindow.getUnionTransactions(sparkContext)
    transacsTotal = slidingWindow.totalWindowTransacs

    itemsetsCandidates = []
    itemsetsCounts = []

    # Sort lexicographically the nodes of the current maximum level of the tree
    mapCurrentLevel = sorted(algStructure.itemsetsTree.getLevelTree(currentLevel),
                             key=algStructure.itemsetsTree.getLevelTree(currentLevel).get)

    # Generate the descendants of each current leaf node (nodes at the maximum level)
    for keyItemsetNode in mapCurrentLevel:
        itemsetsCandidates += FIMoTS_Initial(keyItemsetNode, algStructure.itemsetsTree.nodeMap[keyItemsetNode],
                                             algStructure, slidingWindow, minRelSupportElems, True, True, sparkContext)

    # If a new level of the tree has been generated
    if algStructure.itemsetsTree.getLevelTree(currentLevel + 1):

        # Compute the supports of the new leaf itemsets
        itemsetsCounts = FIMoTS_CalculateSupportLevelsTree(itemsetsCandidates, transactions)

        # Build and add nodes to the Dictionary for each new itemset
        for (itemsetKey, countItemset) in itemsetsCounts:
            fractionElems = DecimalFraction(countItemset, transacsTotal)
            currentNode = algStructure.itemsetsTree.nodeMap[itemsetKey]
            currentNode.relativeSupport = countItemset / float(transacsTotal)
            currentNode.fractionElems = fractionElems


def FIMoTS_Initial(keyCurrentItemNode, currentItemNode, algStructure, slidingWindow, minRelSupportElems,
                   recalculateList, isTreeInitialization, sparkContext):
    """
    Method to unfold a node that represents a frequent itemset in the lexicographic tree, generating its descendants,
    that is, the child nodes that represent supersets of that itemset.

    Args:

        keyCurrentItemNode (String, required): key that identifies the node in the Dictionary/Tree
        currentItemNode (FIMoTS_Node, required): content of the node whose descendants must be unfolded
        algStructure (FIMoTS_Structure, required): current structure with the execution of the FIMoTS algorithm
        slidingWindow (WindowData, required): the set of transactions that make up the time window
        minRelSupportElems (DecimalFraction, required): minimum relative support, specified as a fraction of integers
        recalculateList (Boolean, required): indicates whether the bounds of the current node must be recomputed, as
            well as its location in the list of transformation-bound keys
        isTreeInitialization (Boolean, required): indicates whether the initial process of building the lexicographic
            tree is being carried out, or the tree is being updated in real time. If it is the initialization of the
            tree, only the first level of the descendants of the node must be generated and its support must not be
            computed, since that unfolding and those computations are carried out in the external process of building
            the levels of the tree.
        sparkContext(SparkContext): distributed context to apply Big Data techniques on an RDD with Spark

    Returns:
        List of child nodes generated from the parent node received as argument
    """
    minRelSupport = minRelSupportElems.numeratorFraction / float(minRelSupportElems.denominatorFraction)
    transactions = slidingWindow.getUnionTransactions(sparkContext)
    transacTotal = slidingWindow.totalWindowTransacs

    # If the itemset is infrequent
    if currentItemNode.relativeSupport < minRelSupport:

        # Actualizar cotas transformadoras del nodo en listas
        FIMoTS_UpdateBoundsList(currentItemNode, algStructure, minRelSupportElems, transacTotal)

        return []

    # If the current itemset is frequent
    else:
        rightSiblingsCurrentNode = algStructure.itemsetsTree.getRightSiblings(keyCurrentItemNode, currentItemNode)
        newLevelCandidates = []

        # Para cada nodo hermano derecho
        for keyRightSibling, rightSibling in rightSiblingsCurrentNode.items():

            # If the itemset associated with the sibling node is frequent
            if rightSibling.relativeSupport >= minRelSupport:
                itemPrefixNewNode = sorted(list(set(currentItemNode.itemPrefix + rightSibling.itemPrefix)))
                newLevelCandidates.append(itemPrefixNewNode)

        newNodes = []

        for itemsetCandidate in newLevelCandidates:
            newNode = FIMoTS_Node(itemsetCandidate, 0.0, None, currentItemNode.getItemsetKey(), 0, 0)
            # A child that already exists (e.g. created by FIMoTS_ExtendLeftSiblings when two sibling itemsets
            # become frequent in the same window) is not created again: it would be
            # registered twice in the bound lists and its support would be counted twice.
            if not isTreeInitialization and newNode.getItemsetKey() in algStructure.itemsetsTree.nodeMap:
                continue
            newNodes.append(newNode)

        itemsetsCounts = []

        # Compute the support of new child nodes if the tree is
        # being updated (not initialization)
        # If it is initialization, the computation of the supports is
        # carried out in the external method FIMoTS_BuildLevelTree
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

        # Update the child identification values in the parent node
        currentItemNode.totalChildNodes = len(newNodes)

        if recalculateList:
            # Actualizar cotas transformadoras del nodo en listas
            FIMoTS_UpdateBoundsList(currentItemNode, algStructure, minRelSupportElems, transacTotal)

        if not isTreeInitialization:
            for newChildNode in newNodes:
                FIMoTS_Initial(newChildNode.getItemsetKey(), newChildNode, algStructure, slidingWindow,
                               minRelSupportElems, True, False, sparkContext)

        return newNodes


def FIMoTS_ExtendLeftSiblings(keyNode, node, algStructure, slidingWindow, minRelSupportElems, sparkContext):
    """
    Incorporate an itemset that has just become frequent as a "right sibling" of its frequent left siblings. In the
    tree, the children of a node L are formed by joining L with its FREQUENT right siblings. If a right sibling N was
    infrequent when the children of L were unfolded, L does not have the child L u N; when N becomes frequent (e.g.
    because of a change in the data of the window) it has to be created now. Without this step, the incremental
    algorithm loses frequent itemsets that a full initialization of the same window does find.

    Each new child L u N is registered in the bound lists (FIMoTS_Initial) and, if it turns out to be frequent, its own
    descendants are unfolded and the process is repeated among its left siblings (the other children of L).

    Args:
        keyNode (String, required): key in the tree of the node that has become frequent
        node (FIMoTS_Node, required): content of that node
        algStructure (FIMoTS_Structure, required): current structure of the FIMoTS algorithm
        slidingWindow (WindowData, required): current time window
        minRelSupportElems (DecimalFraction, required): minimum relative support
        sparkContext(SparkContext): Spark distributed context
    """
    minRelSupport = minRelSupportElems.numeratorFraction / float(minRelSupportElems.denominatorFraction)
    transactions = slidingWindow.getUnionTransactions(sparkContext)
    transacTotal = slidingWindow.totalWindowTransacs
    nodeMap = algStructure.itemsetsTree.nodeMap

    newChildren = []
    for keyLeft, left in sorted(algStructure.itemsetsTree.getLeftSiblings(keyNode, node).items()):
        if left.relativeSupport < minRelSupport:
            continue
        itemPrefix = sorted(set(left.itemPrefix + node.itemPrefix))
        child = FIMoTS_Node(itemPrefix, 0.0, None, left.getItemsetKey(), 0, slidingWindow.timeWindow)
        if child.getItemsetKey() in nodeMap:
            continue
        newChildren.append((left, child))

    if not newChildren:
        return

    counts = dict(FIMoTS_CalculateSupportLevelsTree([child for _, child in newChildren], transactions))
    for left, child in newChildren:
        count = counts.get(child.getItemsetKey(), 0)
        child.relativeSupport = float(count) / float(transacTotal)
        child.fractionElems = DecimalFraction(count, transacTotal)
        nodeMap[child.getItemsetKey()] = child
        left.totalChildNodes += 1

    for left, child in newChildren:
        # Register the bounds of the new node and, if it is frequent, unfold its children
        FIMoTS_Initial(child.getItemsetKey(), child, algStructure, slidingWindow, minRelSupportElems, True, False,
                       sparkContext)
        if child.relativeSupport >= minRelSupport:
            FIMoTS_ExtendLeftSiblings(child.getItemsetKey(), child, algStructure, slidingWindow, minRelSupportElems,
                                      sparkContext)


def FIMoTS_AddNewItems(algStructure, slidingWindow, minRelSupportElems, sparkContext):
    """
    Incorporate into the tree the items that appear for the FIRST TIME in the data stream. The first level of the tree
    is built with the vocabulary of the first window, and the support changes of the existing nodes are detected through
    the transformation bounds; but an item that was not in that first window has no node, and without this step it would
    never be detected (neither it nor the itemsets that contain it), however frequent it became.

    The items of the last received batch that have no first-level node are looked for, their support over the current
    window is computed (a single additional Spark job, only if there are new items) and they are inserted as first-level
    nodes. If they turn out to be frequent, their descendants are unfolded and they are incorporated as a pair with
    their frequent left siblings, like any other itemset that becomes frequent.

    This step is executed at the end of FIMoTS_Main: the supports of the new nodes are already computed over the
    current window, so they must not go through the bound update due to the shift of the window.

    Args:
        algStructure (FIMoTS_Structure, required): current structure of the FIMoTS algorithm
        slidingWindow (WindowData, required): already updated time window
        minRelSupportElems (DecimalFraction, required): minimum relative support
        sparkContext(SparkContext): Spark distributed context
    """
    nodeMap = algStructure.itemsetsTree.nodeMap
    minRelSupport = minRelSupportElems.numeratorFraction / float(minRelSupportElems.denominatorFraction)
    newestBatch = slidingWindow.transactionIntervals[-1][0]

    batchItems = set(newestBatch.flatMap(lambda transaction: transaction.split()).distinct().collect())
    newItems = sorted(item for item in batchItems if item not in nodeMap)
    if not newItems:
        return

    newItemsSet = set(newItems)
    transactions = slidingWindow.getUnionTransactions(sparkContext)
    transacTotal = slidingWindow.totalWindowTransacs
    counts = dict(transactions.flatMap(lambda transaction: [(item, 1) for item in set(transaction.split())
                                                            if item in newItemsSet])
                  .reduceByKey(lambda x, y: x + y).collect())

    for item in newItems:
        count = counts.get(item, 0)
        nodeMap[item] = FIMoTS_Node([item], count / float(transacTotal), DecimalFraction(count, transacTotal), '-1', 0,
                                    slidingWindow.timeWindow)
    nodeMap['-1'].totalChildNodes += len(newItems)

    for item in newItems:
        node = nodeMap[item]
        FIMoTS_Initial(item, node, algStructure, slidingWindow, minRelSupportElems, True, False, sparkContext)
        if node.relativeSupport >= minRelSupport:
            FIMoTS_ExtendLeftSiblings(item, node, algStructure, slidingWindow, minRelSupportElems, sparkContext)


def FIMoTS_UpdateBoundsList(currentItemNode, algStructure, minSupportElems, windowSize):
    """
    Update the lists with the transformation bounds of a node/itemset in the solution structure of the algorithm.

    Args:
        currentItemNode (FIMoTS_Node, required): node whose bounds, as well as its location in the linked lists of
            itemset bounds, must be recomputed
        algStructure (FIMoTS_Structure, required): current structure for the execution of the FIMoTS algorithm
        minSupportElems (DecimalFraction, required): the minimum support with which an itemset can be considered
            frequent, expressed as a fraction of integers
        windowSize (int, required): total number of transactions that make up the current time window
    """
    # Compute new transformation limits of the node
    nodeSupportElems = currentItemNode.fractionElems
    minSupport = float(minSupportElems.numeratorFraction) / float(minSupportElems.denominatorFraction)
    nodeItemsetKey = currentItemNode.getItemsetKey()
    nodeNewBounds = FIMoTS_Bounds()

    # If the itemset is an infrequent itemset
    if currentItemNode.relativeSupport < minSupport:
        nodeUpperBound = FIMoTS_CalculateUpperBound(minSupportElems, nodeSupportElems, windowSize, False)
        nodeLowerBound = FIMoTS_CalculateLowerBound(minSupportElems, nodeSupportElems, windowSize, False)
        nodeNewBounds = FIMoTS_Bounds(nodeUpperBound, nodeLowerBound, [nodeItemsetKey])

    # If the itemset is frequent
    else:
        nodeUpperBound = FIMoTS_CalculateUpperBound(minSupportElems, nodeSupportElems, windowSize, True)
        nodeLowerBound = FIMoTS_CalculateLowerBound(minSupportElems, nodeSupportElems, windowSize, True)
        nodeNewBounds = FIMoTS_Bounds(nodeUpperBound, nodeLowerBound, [nodeItemsetKey])

    currentItemList = FIMoTS_List()

    # Generar elemento correspondiente en lista de cotas (ajustar apuntadores)
    # If the itemset remains an infrequent itemset
    if currentItemNode.relativeSupport < minSupport:
        currentItemList = algStructure.infrequentItemsetsBounds

    # Si el itemset infrecuente ha cambiado a frecuente
    else:
        currentItemList = algStructure.frequentItemsetsBounds

    equalBounds = None
    findEqualBounds = False

    # Look for another element in the list of transformation bounds with the same combination
    # of upper - lower limit
    for bounds in currentItemList.itemsBounds:

        if nodeNewBounds.upperBound == bounds.upperBound and nodeNewBounds.lowerBound == bounds.lowerBound:
            # If an equivalent element is found
            equalBounds = bounds
            findEqualBounds = True
            break

    # If an equivalent combination of bounds already exists in the list
    if findEqualBounds:

        # Add the current node to the end of the list of
        # pointers of the equivalent bound
        equalBounds.itemNodes.append(nodeItemsetKey)

    # If it is a new upper/lower pair of bounds
    else:

        # Add a new element to the list of distinct bounds of
        # itemsets frecuentes
        currentItemList.itemsBounds.append(nodeNewBounds)


def FIMoTS_CalculateUpperBound(relSupportElems, currentSupportElems, windowSize, isFrequent):
    """
    Compute the new value of the Upper Transformation Bound of an itemset

    relSupportElems (DecimalFraction, required): minimum relative support, specified as a fraction of integers
    currentSupportElems (DecimalFraction, required): current support of the itemset, specified as a fraction of integers
    windowSize (int, required): number of transactions in the current sliding window
    isFrequent (Boolean, required): indicates whether the itemset on which the computation is applied is frequent or not

    Returns:
        The new value of the Upper Transformation Bound of the itemset
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
    Compute the new value of the Lower Transformation Bound of an itemset

    relSupportElems (DecimalFraction, required): minimum relative support, specified as a fraction of integers
    currentSupportElems (DecimalFraction, required): current support of the itemset, specified as a fraction of integers
    windowSize (int, required): number of transactions in the current sliding window
    isFrequent (Boolean, required): indicates whether the itemset on which the computation is applied is frequent or not

    Returns:
        The new value of the Lower Transformation Bound of the itemset
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
    Return the decimal value that is closest to the received argument and is equal to an integer.

    Args:
        num (float, required): The numeric value to round to an integer

    Returns:
        The decimal value that is closest to the received argument and is equal to an integer.
    """
    return round(num + (num % 2 - 1 if (num % 1 == 0.5) else 0))


def FIMoTS_CalculateSupportLevelsTree(itemsetsCandidates, transactions):
    """
    Compute the relative supports of a set of candidate itemsets, over a group of transactions that make up a time
    window.

    Args:
        itemsetsCandidates(Array(FIMoTS_Node), required): list of nodes of the tree whose support is to be updated or
            computed over a set of transactions
        transactions (RDD, required): the set of transactions that make up the time window

    Returns:
        A list with the (key, count) pairs associated with the candidate itemsets received as arguments.
    """
    # Split each line of the RDD into words
    transactionsAsList = transactions.map(lambda transaction: transaction.split())

    # Count each word in each subset
    itemsetPairs = transactionsAsList.flatMap(
        lambda transaction: getItemsetsInTransaction(transaction, itemsetsCandidates))
    itemsetCounts = itemsetPairs.reduceByKey(lambda x, y: x + y).collect()

    return itemsetCounts


def getItemsetsInTransaction(transaction, itemsetsCandidates):
    """
    Auxiliary function to determine whether an itemset is contained in a given transaction of items.

    Args:
        itemsetsCandidates(Array(FIMoTS_Node), required): list of nodes of the tree for which it is to be determined
            whether they are present in the transaction
        transaction (Array(String), required): a transaction of items

    Returns:
        A list with the (key, 1) pairs for each itemset contained in the transaction, or (key, 0) if the itemset is not
        contained in the transaction
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
    Main program of the FIMoTS algorithm, in which the state of the frequent and infrequent itemsets is updated based
    on the update of a data window sliding in time. The update depends on the changes that have occurred in the upper
    and lower transformation bounds of each identified itemset.

    deletedTransacSize (int, required): total number of transactions removed from the time window
    addedTransacSize (int, required): total number of transactions added to the time window
    algStructure (FIMoTS_Structure, required): current structure with the execution of the FIMoTS algorithm
    slidingWindow (WindowData, required): Updated sliding time window
    minRelSupportElems (DecimalFraction, required): Minimum relative support, specified as a fraction of integers
    sparkContext(SparkContext): distributed context to apply Big Data techniques on an RDD with Spark
    """
    relSupportElems = minRelSupportElems
    minRelSupport = float(minRelSupportElems.numeratorFraction) / float(minRelSupportElems.denominatorFraction)

    transactions = slidingWindow.getUnionTransactions(sparkContext)
    windowSize = slidingWindow.totalWindowTransacs

    # Estudiar lista de itemset frecuentes
    for currentBounds in algStructure.frequentItemsetsBounds.itemsBounds:
        # Update Lower Transformation Bound
        currentLowerBound = currentBounds.lowerBound
        currentBounds.lowerBound = FIMoTS_UpdateLowerBound(currentLowerBound, relSupportElems, deletedTransacSize,
                                                           addedTransacSize, True)

        # Update Upper Transformation Bound
        currentUpperBound = currentBounds.upperBound
        currentBounds.upperBound = FIMoTS_UpdateUpperBound(currentUpperBound, relSupportElems, deletedTransacSize,
                                                           addedTransacSize, True)

    # Estudiar lista de itemset infrecuentes
    for currentBounds in algStructure.infrequentItemsetsBounds.itemsBounds:
        # Update Lower Transformation Bound
        currentLowerBound = currentBounds.lowerBound
        currentBounds.lowerBound = FIMoTS_UpdateLowerBound(currentLowerBound, relSupportElems, deletedTransacSize,
                                                           addedTransacSize, False)

        # Update Upper Transformation Bound
        currentUpperBound = currentBounds.upperBound
        currentBounds.upperBound = FIMoTS_UpdateUpperBound(currentUpperBound, relSupportElems, deletedTransacSize,
                                                           addedTransacSize, False)

    nodesUpdated = []
    itemsetsCounts = []

    # Update frequent itemsets with transformation bounds equal to or lower than zero
    for currentBounds in algStructure.frequentItemsetsBounds.itemsBounds:

        if (currentBounds.lowerBound <= 0) or (currentBounds.upperBound <= 0):

            # While there is another node with the same upper - lower bound combination
            for currentFreqItemKey in currentBounds.itemNodes:
                freqNodeUpdated = algStructure.itemsetsTree.nodeMap.get(currentFreqItemKey)
                nodesUpdated.append(freqNodeUpdated)

            # Remove all the already studied itemsets from the list of frequent ones (with their original bounds)
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
                # Prune the child subtrees of the node
                algStructure.deleteChildNodes(currentFreqItemKey)

        nodesUpdated = []

    # Update infrequent itemsets with transformation bounds equal to or lower than zero
    for currentBounds in algStructure.infrequentItemsetsBounds.itemsBounds:

        if (currentBounds.lowerBound <= 0) or (currentBounds.upperBound <= 0):

            # While there is another node with the same upper - lower bound combination
            for currentInfreqItemKey in currentBounds.itemNodes:
                infreqNodeUpdated = algStructure.itemsetsTree.nodeMap.get(currentInfreqItemKey)
                nodesUpdated.append(infreqNodeUpdated)

            # Remove all the already studied itemsets from the list of infrequent ones (with their original bounds)
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

    # Generate the descendants of all the new frequent nodes
    for newFreqItemKey in newFrequentItemsKeys:
        newFreqItemNode = algStructure.itemsetsTree.nodeMap.get(newFreqItemKey)
        FIMoTS_Initial(newFreqItemKey, newFreqItemNode, algStructure, slidingWindow, minRelSupportElems, False, False,
                       sparkContext)
        FIMoTS_ExtendLeftSiblings(newFreqItemKey, newFreqItemNode, algStructure, slidingWindow, minRelSupportElems,
                                  sparkContext)

    FIMoTS_AddNewItems(algStructure, slidingWindow, minRelSupportElems, sparkContext)

    algStructure.frequentItemsetsBounds.clearEmptyBounds()
    algStructure.infrequentItemsetsBounds.clearEmptyBounds()


def FIMoTS_UpdateLowerBound(currentLowerBound, relSupportElems, deletedTransacsSize, addedTransacsSize, isFrequent):
    """
    Update the value of the Lower Transformation Bound of an itemset.

    currentLowerBound (int, required): the value of the current lower transformation bound
    relSupportElems (DecimalFraction, required): minimum relative support, specified as a fraction of integers
    deletedTransacsSize (int, required): number of transactions removed from the time window
    addedTransacsSize (int, required): total number of transactions added to the sliding window
    isFrequent (Boolean, required): indicates whether the itemset on which the update is applied is frequent or not

    Returns:
        The updated value of the Lower Transformation Bound of the itemset
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
    Update the value of the Upper Transformation Bound of an itemset.

    currentUpperBound (int, required): the value of the current upper transformation bound
    relSupportElems (DecimalFraction, required): minimum relative support, specified as a fraction of integers
    deletedTransacsSize (int, required): number of transactions removed from the time window
    addedTransacsSize (int, required): total number of transactions added to the sliding window
    isFrequent (Boolean, required): indicates whether the itemset on which the update is applied is frequent or not

    Returns:
        The updated value of the Upper Transformation Bound of the itemset
    """
    upperBoundUpdate = 0.0

    if isFrequent:
        upperBoundUpdate = (float(relSupportElems.denominatorFraction - relSupportElems.numeratorFraction) / float(
            relSupportElems.numeratorFraction)) * deletedTransacsSize
    else:
        upperBoundUpdate = (float(relSupportElems.numeratorFraction) / float(
            relSupportElems.denominatorFraction - relSupportElems.numeratorFraction)) * deletedTransacsSize

    return int(floor(currentUpperBound - upperBoundUpdate - addedTransacsSize))
