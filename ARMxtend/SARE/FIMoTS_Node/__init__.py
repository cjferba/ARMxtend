#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from ..DecimalFraction import DecimalFraction

class FIMoTS_Node(object):
    """
    Utility class to define the structure of a node of a lexicographic tree. Each node of the tree represents a
    distinct itemset. A parent node represents a "prefix" (subset) of each of its child nodes. A child node is the union
    of the prefixes of its parent and one of its "right uncles" (siblings located to the right of its parent node).

    Attributes:
        itemPrefix (Array(String), optional, default []): List of items that make up the itemset identified in the node
        relativeSupport (float, optional, default 0.0): Relative support associated with the itemset. Frequency with
            which that prefix appears in the set of transactions studied
        fractionElems (DecimalFraction, optional, default None): Numerator and Denominator that define the relative
            support of an itemset.
        parentNode (String, optional, default ''): Key in the Dictionary/Tree of nodes that identifies the parent node
            of the current element
        totalChildNodes (int, optional, default 0): Total number of child nodes that the current node has in the
            lexicographic tree
        lastUpdate (int, optional, default -1): Temporal indicator that specifies the moment, in the sliding time
            window, in which the relative support of this itemset was last updated

    @author Abel Francisco (2016)
    """

    def __init__(self, itemPrefix=None, relativeSupport=0.0, fractionElems=None,
                 parentNode='', totalChildNodes=0, lastUpdate=-1):
        """
        Initialization of a new node of the lexicographic tree that represents a frequent itemset identified in a set
        of transactions.

        Args:
            itemPrefix (Array(String), optional, default []): List of items that make up the itemset identified in the
                node
            relativeSupport (float, optional, default 0.0): Relative support associated with the itemset. Frequency with
                which that prefix appears in the set of transactions studied
            fractionElems (DecimalFraction, optional, default None): Numerator and Denominator that define the relative
                support of an itemset.
            parentNode (String, optional, default ''): Key in the Dictionary/Tree of nodes that identifies the parent
                node of the current element
            totalChildNodes (int, optional, default 0): Total number of child nodes that the current node has in the
                lexicographic tree
            lastUpdate (int, optional, default -1): Temporal indicator that specifies the moment, in the sliding time
                window, in which the relative support of this itemset was last updated
        """
        self.itemPrefix = itemPrefix if itemPrefix is not None else []
        self.relativeSupport = relativeSupport
        self.fractionElems = fractionElems
        self.parentNode = parentNode
        self.totalChildNodes = totalChildNodes
        self.lastUpdate = lastUpdate

    def orderItemPrefix(self):
        """
        Sort alphabetically the elements of the itemset associated with the node.
        """
        self.itemPrefix = sorted(self.itemPrefix)

    def getItemPrefixConcat(self):
        """
        Get a string that identifies the itemset, concatenating the items of the prefix.

        Returns:
            A string that concatenates the items of the prefix of the itemset directly (without separators).
        """
        return ''.join(self.itemPrefix).lower()

    def getItemsetKey(self):
        """
        Get a string that identifies the itemset, concatenating the items of the prefix with underscores ('_'). This
        resulting string is the format of the keys that identify the nodes in the Dictionary/Lexicographic Tree.

        Returns:
            A string that concatenates the items of the prefix of the itemset, separating them by an underscore ('_').
        """
        return '_'.join(self.itemPrefix)

    def __repr__(self):
        """
        Mechanism to display or print a node of the lexicographic tree

        Returns:
            A string that visually represents (text) a node of the lexicographic tree
        """
        nodeFormat = '{}: [ PREFIJO: {}, SOPORTE: {}, FRACCION: {}, PADRE: {}, TOTAL HIJOS: {}, ACTUALIZACION: {} ]'
        return nodeFormat.format(self.__class__.__name__, '_'.join(self.itemPrefix),
                                 self.relativeSupport, self.fractionElems,
                                 self.parentNode, self.totalChildNodes, self.lastUpdate)

    def __eq__(self, other):
        """
        Mechanism to define whether two (2) nodes of the lexicographic tree are equivalent. Two (2) nodes are
        equivalent if their concatenated prefixes are exactly equal.

        Args:
            other (FIMoTS_Node, required): Node to compare the current node with

        Returns:
            True if the nodes are equivalent, False if not
        """
        if hasattr(other, 'itemPrefix'):
            return (self.getItemPrefixConcat() == other.getItemPrefixConcat())

    def __lt__(self, other):
        """
        Mechanism to define whether one (1) node of the lexicographic tree is "less" than another. A node is "earlier"
        or "less" than another if its written representation precedes, lexicographically, the concatenated prefix of
        the other.

        Args:
            other (FIMoTS_Node, required): Node to compare the current node with

        Returns:
            True if the current node is earlier than the other, False if not
        """
        return (self.getItemPrefixConcat() < other.getItemPrefixConcat())
