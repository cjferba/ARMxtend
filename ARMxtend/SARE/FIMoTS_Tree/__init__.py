#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from ..FIMoTS_Node import FIMoTS_Node

class FIMoTS_Tree(object):
    """
    Utility class to define the general structure of a lexicographic tree (FP-Tree), used repeatedly in the computation
    of frequent itemsets in sliding time windows. By definition, the structure of the tree is recursive, that is, the
    children of a node of the tree are other lexicographic trees. However, in order to optimize real execution in Big
    Data environments, the current implementation of the tree is not handled recursively. Instead, a Dictionary or
    HashMap (set of Key-Value pairs) with the nodes is used.

    Each node represents an itemset. A parent node is a subset of its child nodes. The first level of the tree is
    itemsets of size one (1), the second level of size two (2) and so on until there are no frequent itemsets (an
    infrequent itemset has no descendants in the tree). If a parent node is not frequent, its children will not be
    either. The leaves of the tree are infrequent itemsets.

    The keys of each node have the form <code1>_<code2>_..._<codeN>, where <codei> is an identifier associated with a
    particular item of the language of the transactions. The codes that make up the key define the prefix corresponding
    to the current itemset/node. The level at which the node is located is the number of '_' in the key of the itemset
    plus one (1).

    Attributes:
        nodeMap (HashMap(string,FIMoTS_Node), optional, default {}): Dictionary with the set of frequent
            nodes/itemsets identified

    @author Abel Francisco (2016)
    """

    def __init__(self, nodeMap={}):
        """
        Initialization of a new lexicographic FP-Tree

        Args:
            nodeMap (HashMap(string,FIMoTS_Node), optional, default {}): New dictionary with the set of frequent
                nodes/itemsets identified
        """
        self.nodeMap = nodeMap

    def __repr__(self):
        """
        Mechanism to display or print the lexicographic FP-Tree

        Returns:
            A string that visually represents (text) a lexicographic FP-Tree
        """
        treeFormat = '{}: [ NODOS: {} ]'
        return treeFormat.format(self.__class__.__name__,
                                 self.nodeMap)

    def getLevelTree(self, levelNumber):
        """
        Get all the nodes that make up a specific level of the tree. Each FP-Tree starts at level zero (0) or root, in
        which there is a node not associated with any itemset, but which serves as the original 'Parent' of all the
        nodes of the tree. This level zero (0) node has key equal to '-1'

        The level of a node is determined according to the number of '_' characters in its associated key in the
        Dictionary (Node Level = Total '_' + 1)

        Args:
            levelnumber(int): numeric identifier of the level of the tree whose nodes are to be obtained.

        Returns:
            A Dictionary with all the nodes of the tree located at the indicated level.
        """
        levelSelected = {k: v for k, v
                         in self.nodeMap.items()
                         if k.count('_') == (levelNumber - 1) and
                         k != '-1'
                         }
        return levelSelected

    @staticmethod
    def getParentKey(keyNode):
        """
        Key of the parent of a node: the key without its last item ('a_b_c' -> 'a_b'). The nodes of the first level hang
        from the root, with key '-1'.
        """
        return keyNode.rsplit('_', 1)[0] if '_' in keyNode else '-1'

    def _getSiblings(self, keyNode, isRight):
        parentKey = self.getParentKey(keyNode)
        lastElement = keyNode.split('_')[-1]
        level = keyNode.count('_')
        # The parent is compared EXACTLY (not with startswith): with items such as 'milk' and
        # 'milkshake', 'milkshake_x' starts with 'milk' but is not a child of 'milk'.
        return {k: v for k, v in self.nodeMap.items()
                if k != '-1' and k.count('_') == level and self.getParentKey(k) == parentKey and
                ((k.split('_')[-1] > lastElement) if isRight else (k.split('_')[-1] < lastElement))}

    def getRightSiblings(self, keyNode, node=None):
        """
        Get the right sibling nodes of a particular itemset. The nodes of the same level of the tree are lexicographically
        ordered according to their itemsets. Thus, the right siblings of a node are those nodes of the same level and
        with the same parent that have as the last component of their prefix an element greater (lexicographically) than
        the node in question.

        Args:
            keyNode(string): key of the node whose right siblings are to be found
            node(FIMoTS_Node): kept for compatibility; the parent is deduced from the key

        Returns:
            A Dictionary with the right siblings of a particular node of the tree.
        """
        return self._getSiblings(keyNode, True)

    def getLeftSiblings(self, keyNode, node=None):
        """
        Get the left sibling nodes of an itemset: those of its same level and same parent whose last item is
        lexicographically smaller. They are the nodes that, to form their children, must consider this node as a "right
        sibling" when it is frequent.

        Returns:
            A Dictionary with the left siblings of a particular node of the tree.
        """
        return self._getSiblings(keyNode, False)

    def deleteChildNodes(self, keyParentNode):
        """
        Remove from the tree all the descendant nodes/itemsets (children, grandchildren, ...) of a particular node. The
        descendants of 'a_b' are all the keys that start with 'a_b_'.

        Args:
            keyParentNode(string): key, in the dictionary/tree, of the parent node whose descendants are to be removed

        Returns:
            A list with the keys of the elements/nodes removed
        """
        prefix = keyParentNode + '_'
        deletedNodes = [k for k in self.nodeMap if k.startswith(prefix)]
        for k in deletedNodes:
            del self.nodeMap[k]
        return deletedNodes

    def getItemsetsCount(self):
        """
        Get the total number of itemsets/nodes that make up the current lexicographic tree. The root node (which does not
        represent any real itemset) is excluded from the count

        Returns:
            Integer with the total count of valid itemsets/nodes included in the lexicographic tree
        """
        return (len(self.nodeMap) - 1)
