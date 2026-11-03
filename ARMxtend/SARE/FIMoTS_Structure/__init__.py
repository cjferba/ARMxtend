#!/usr/bin/env python
# -*- coding: utf-8 -*-


class FIMoTS_Structure(object):
    """
    Utility class to define a structure that groups all the components needed to run the algorithm for obtaining
    frequent itemsets in time-sensitive time windows, FIMoTS.

    Attributes:
        itemsetsTree (FIMoTS_Tree, optional, default None): Lexicographically ordered tree with the frequent
            nodes/itemsets identified
        frequentItemsetsBounds (FIMoTS_List, optional, default None): List with the distinct transformation bounds of
            the frequent itemsets
        infrequentItemsetsBounds (FIMoTS_List, optional, default None): List with the distinct transformation bounds of
            the infrequent itemsets

    @author Abel Francisco (2016)
    """

    def __init__(self, itemsetsTree=None, frequentItemsetsBounds=None, infrequentItemsetsBounds=None):
        """
        Initialization of a new structure for the resolution and execution of the FIMoTS algorithm for the
        identification of frequent itemsets

        Args:
            itemsetsTree (FIMoTS_Tree, optional, default None): New lexicographically ordered tree with the frequent
                nodes/itemsets identified
            frequentItemsetsBounds (FIMoTS_List, optional, default None): New list with the transformation bounds of the
                frequent itemsets
            infrequentItemsetsBounds (FIMoTS_List, optional, default None): New list with the transformation bounds of
                the infrequent itemsets
        """
        self.itemsetsTree = itemsetsTree
        self.frequentItemsetsBounds = frequentItemsetsBounds
        self.infrequentItemsetsBounds = infrequentItemsetsBounds

    def __repr__(self):
        """
        Mechanism to display or print the execution structure of the FIMoTS algorithm

        Returns:
            A string that visually represents (text) an execution structure of the FIMoTS algorithm
        """
        treeFormat = '{}: [ ARBOL: {}, FRECUENTES: {}, INFRECUENTES: {} ]'
        return treeFormat.format(self.__class__.__name__, self.itemsetsTree,
                                 self.frequentItemsetsBounds,
                                 self.infrequentItemsetsBounds)

    def deleteChildNodes(self, keyParentNode):
        """
        Remove from the tree and the bound lists all the nodes/itemsets that descend from a particular node.

        Args:
            keyParentNode(string): key, in the dictionary/tree, of the parent node whose descendants are to be removed
        """
        deletedNodesKeys = self.itemsetsTree.deleteChildNodes(keyParentNode)
        for deletedKeyNode in deletedNodesKeys:
            isFrequent = self.frequentItemsetsBounds.deleteBounds(deletedKeyNode)
            if not isFrequent:
                self.infrequentItemsetsBounds.deleteBounds(deletedKeyNode)
