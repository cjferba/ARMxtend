#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from ..FIMoTS_Bounds import FIMoTS_Bounds

class FIMoTS_List(object):
    """
    Utility class that specifies the structure of a list of triples, with the distinct combinations of transformation
    bounds (upper - lower) that the itemsets have in a solution of the FIMoTS algorithm.

    Attributes:
        itemsBounds (Array(FIMoTS_Bounds), optional, default []): The list of distinct combinations of transformation
            bounds

    @author Abel Francisco (2016)
    """

    def __init__(self, itemsBounds=None):
        """
        Initialization of the structure with the list of transformation bounds (upper - lower) that the itemsets have in
        a solution of the FIMoTS algorithm.

        Attributes:
            itemsBounds (Array(FIMoTS_Bounds), optional, default []): The new list of distinct combinations of
                transformation bounds
        """
        self.itemsBounds = itemsBounds if itemsBounds is not None else []

    def __repr__(self):
        """
        Mechanism to display or print a list of transformation bounds

        Returns:
            A string that visually represents (text) a list of transformation bounds
        """
        boundListFormat = '{}: [ COTAS: {} ]'
        return boundListFormat.format(self.__class__.__name__, self.itemsBounds)

    def clearEmptyBounds(self):
        """
        Remove those pairs of bounds not associated with any itemset of the tree
        """
        newBoundList = []
        for bounds in self.itemsBounds:
            if bounds.itemNodes:
                newBoundList.append(bounds)
        self.itemsBounds = newBoundList

    def deleteBounds(self, keyNode):
        """
        Remove from the list of bounds the references to a particular node.

        Args:
            keyNode(string): key, in the dictionary/tree, of the node to remove from the list of bounds

        Returns:
            True, if the key of the node was found in the list, False if not
        """
        keyRemoved = False
        for bounds in self.itemsBounds:
            if keyNode in bounds.itemNodes:
                bounds.itemNodes.remove(keyNode)
                keyRemoved = True
                break

        return keyRemoved

    def getItemsetsCount(self):
        """
        Get the total number of itemsets included in the list, referenced by each of the bounds that make it up.

        Returns:
            Integer with the total count of itemsets that are referenced from the bounds that make up the list
        """
        totalItemsets = 0
        for bounds in self.itemsBounds:
            totalItemsets += bounds.getItemsetsCount()

        return totalItemsets

    def getItemsetsKeys(self):
        """
        Get the list of keys in the dictionary of all the itemsets/nodes referenced in the list.

        Returns:
            List of strings that represent the keys of each of the nodes referenced by the current list of bounds
        """
        itemsetsKeys = []
        for bounds in self.itemsBounds:
            itemsetsKeys.extend(bounds.itemNodes)

        return itemsetsKeys
