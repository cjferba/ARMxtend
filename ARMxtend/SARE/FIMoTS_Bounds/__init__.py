#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from ..FIMoTS_Node import FIMoTS_Node

class FIMoTS_Bounds(object):
    """
    Utility class to define a triple that contains the transformation bounds of an itemset (FIMoTS algorithm), and a
    list with the keys, in the Dictionary/Lexicographic Tree, of the nodes that share those bounds.

    Attributes:
        upperBound (int, optional, default 0): Upper transformation bound of an itemset
        lowerBound (int, optional, default 0): Lower transformation bound of an itemset
        itemNodes (Array(String), optional, default []): list of keys in the Dictionary that identify the nodes of the
            tree that share the combination of bounds specified in the triple

    @author Abel Francisco (2016)
    """

    def __init__(self, upperBound = 0, lowerBound = 0, itemNodes = []):
        """
        Initialization of a triple that contains the transformation bounds of an itemset (FIMoTS algorithm), and the
        identifiers of the itemsets that share them.

        Attributes:
            upperBound (int, optional, default 0): New upper transformation bound of an itemset
            lowerBound (int, optional, default 0): New lower transformation bound of an itemset
            itemNodes (Array(String), optional, default []): New list of keys in the Dictionary that identify the nodes
                of the tree that share the combination of bounds specified in the triple

        @author Abel Francisco (2016)
        """
        self.upperBound	= upperBound
        self.lowerBound = lowerBound
        self.itemNodes  = itemNodes
    	
    def __repr__(self):
        """
        Mechanism to display or print a triple of transformation bounds and their itemsets.

        Returns:
            A string that visually represents (text) a combination of FIMoTS transformation bounds
        """
        boundFormat = '{}: [ COTA SUPERIOR: {}, COTA INFERIOR: {}, ITEMS: {} ]'
        return boundFormat.format(self.__class__.__name__, self.upperBound, 
    							  self.lowerBound, self.itemNodes)

    def getItemsetsCount(self):
    	"""
        Get the total number of itemsets that share the current pair of transformation bounds

        Returns:
            Integer with the total count of itemsets that share the pair of transformation bounds
        """
    	return len(self.itemNodes)