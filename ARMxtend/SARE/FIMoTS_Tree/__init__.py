#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from ..FIMoTS_Node import FIMoTS_Node

class FIMoTS_Tree(object):
    """
	Clase utilitaria para definir la estructura general de un árbol lexicográfico (FP-Tree), utilizado
	de forma recurrente en el cálculo de itemsets frecuentes en ventanas de tiempo deslizantes. Por
	definición, la estructura del árbol es recursiva, es decir, los hijos de un nodo del árbol son
	otros árboles lexicográficos. Sin embargo, a fin de optimizar la ejecución real
	en ambientes de Big Data, la implementación actual del árbol no se maneja recursivamente.
	En cambio, se utiliza un Diccionario o HashMap (conjunto de Clave-Valor) con los nodos.
  
	Cada nodo representa un itemset. Un nodo padre es un subconjunto de sus nodos hijos. El primer nivel
	del árbol son itemsets de tamaño uno (1), el segundo nivel de tamaño dos (2) y así sucesivamente hasta
	que no existan itemsets frecuentes (un itemset infrecuente no tiene descendencia en el árbol). Si un
	nodo padre no es frecuente, sus hijos tampoco lo serán. Las hojas del árbol son itemsets infrecuentes.
	 
	Las claves de cada nodo son de la forma <codigo1>_<codigo2>_..._<codigoN>, donde <codigoi> es
	un identificador asociado a un item en particular del lenguaje de las transacciones. Los códigos
	que integran la clave definen el prefijo correspondiente al itemset/nodo actual. El nivel
	en el que se ubica el nodo es la cantidad de '_' que integran la clave del itemset más uno (1).
	
	Atributos:
    	nodeMap (HashMap(string,FIMoTS_Node), opcional, por defecto {}): Diccionario con el conjunto 
    		de nodos/itemsets frecuentes identificados
        
	@author Abel Francisco (2016)
    """

    def __init__(self, nodeMap={}):
        """
    	Inicialización de un nuevo árbol lexicográfico FP-Tree
    	
    	Argumentos:
    		nodeMap (HashMap(string,FIMoTS_Node), opcional, por defecto {}): Nuevo diccionario con el conjunto
    			de nodos/itemsets frecuentes identificados
        """
        self.nodeMap = nodeMap

    def __repr__(self):
        """
    	Mecanismo de presentación o escritura en pantalla del árbol lexicográfico FP-Tree
    	
    	Retorna:
    		Una cadena de caracteres que representa visualmente (texto) a un árbol
    		lexicográfico FP-Tree
        """
        treeFormat = '{}: [ NODOS: {} ]'
        return treeFormat.format(self.__class__.__name__,
                                 self.nodeMap)

    def getLevelTree(self, levelNumber):
        """
    	Obtener todos los nodos que integran un nivel específico del árbol. Cada árbol FP-Tree
    	inicia en el nivel cero (0) o raíz, en el cual hay un nodo no asociado a ningún itemset,
    	pero que sirve de 'Padre' original de todos los nodos del árbol. Este nodo de nivel
    	cero (0) tiene clave igual a '-1'
    	
    	El nivel de un nodo se determina de acuerdo al número de caracteres '_' que tenga su
    	clave asociada en el Diccionario (Nivel Nodo = Total de '_' + 1)
    	
    	Argumentos:
    		levelnumber(int): identificador numérico del nivel del árbol del cual se quieren
    			obtener los nodos.
    			
    	Retorna:
    		Un Diccionario con todos los nodos del árbol que se ubican en el nivel indicado.
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
        Clave del padre de un nodo: la clave sin su ultimo item ('a_b_c' -> 'a_b'). Los nodos
        del primer nivel cuelgan de la raiz, de clave '-1'.
        """
        return keyNode.rsplit('_', 1)[0] if '_' in keyNode else '-1'

    def _getSiblings(self, keyNode, isRight):
        parentKey = self.getParentKey(keyNode)
        lastElement = keyNode.split('_')[-1]
        level = keyNode.count('_')
        # Se compara el padre de forma EXACTA (no con startswith): con items como 'milk' y
        # 'milkshake', 'milkshake_x' empieza por 'milk' pero no es hijo de 'milk'.
        return {k: v for k, v in self.nodeMap.items()
                if k != '-1' and k.count('_') == level and self.getParentKey(k) == parentKey and
                ((k.split('_')[-1] > lastElement) if isRight else (k.split('_')[-1] < lastElement))}

    def getRightSiblings(self, keyNode, node=None):
        """
    	Obtener los nodos hermanos derechos de un itemset en particular. Los nodos de un
    	mismo nivel del árbol se encuentran ordenados lexicográficamente de acuerdo a sus
    	itemsets. Así, los hermanos derechos de un nodo son aquellos nodos de un mismo nivel
    	y con el mismo padre que tengan como último componente de su prefijo un elemento mayor
    	(lexicográficamente) que el nodo en cuestión.

    	Argumentos:
    		keyNode(string): clave del nodo a buscar sus hermanos derechos
    		node(FIMoTS_Node): se mantiene por compatibilidad; el padre se deduce de la clave

    	Retorna:
    		Un Diccionario con los hermanos derechos de un nodo en particular del árbol.
        """
        return self._getSiblings(keyNode, True)

    def getLeftSiblings(self, keyNode, node=None):
        """
    	Obtener los nodos hermanos izquierdos de un itemset: los de su mismo nivel y mismo
    	padre cuyo último item es lexicográficamente menor. Son los nodos que, para formar
    	sus hijos, deben considerar a este nodo como "hermano derecho" cuando es frecuente.

    	Retorna:
    		Un Diccionario con los hermanos izquierdos de un nodo en particular del árbol.
        """
        return self._getSiblings(keyNode, False)

    def deleteChildNodes(self, keyParentNode):
        """
		Eliminar del árbol todos los nodos/itemsets descendientes (hijos, nietos, ...) de un
		nodo en particular. Los descendientes de 'a_b' son todas las claves que empiezan por
		'a_b_'.

		Argumentos:
			keyParentNode(string): clave, en el diccionario/árbol, del nodo padre cuya
				descendencia se desea eliminar

		Retorna:
			Una lista con las claves de los elementos/nodos eliminados
        """
        prefix = keyParentNode + '_'
        deletedNodes = [k for k in self.nodeMap if k.startswith(prefix)]
        for k in deletedNodes:
            del self.nodeMap[k]
        return deletedNodes

    def getItemsetsCount(self):
        """
    	Obtiene el total de itemsets/nodos que integran el árbol lexicográfico actual. Se
    	excluye el nodo raíz (que no representa ningún itemset real) del conteo
    	
    	Retorna:
    		Entero con el conteo total de itemsets/nodos válidos incluidos el árbol lexicográfico
    	"""
        return (len(self.nodeMap) - 1)
