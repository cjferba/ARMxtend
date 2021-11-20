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
                         in self.nodeMap.iteritems()
                         if k.count('_') == (levelNumber - 1) and
                         k != '-1'
                         }
        return levelSelected

    def getRightSiblings(self, keyNode, node):
        """
    	Obtener los nodos hermanos derechos de un itemset en particular. Los nodos de un
    	mismo nivel del árbol se encuentran ordenados lexicográficamente de acuerdo a sus
    	itemsets. Así, los hermanos derechos de un nodo son aquellos nodos de un mismo nivel 
    	que tengan como último componente de su prefijo un elemento mayor (lexicográficamente)
    	que el nodo en cuestión.
    	
    	Argumentos:
    		keyNode(string): clave del nodo a buscar sus hermanos derechos
    		node(FIMoTS_Node): contenido del nodo del árbol a buscar sus hermanos derechos
    	
    	Retorna:
    		Un Diccionario con los hermanos derechos de un nodo en particular del árbol.
        """
        currentNodeElements = keyNode.split('_')
        lastNodeElement = currentNodeElements[-1]

        if len(currentNodeElements) == 1:
            rightSiblingsMap = {k: v for k, v
                                in self.nodeMap.iteritems()
                                if keyNode.count('_') == k.count('_') and
                                k.split('_')[-1] > lastNodeElement
                                }
        else:
            rightSiblingsMap = {k: v for k, v
                                in self.nodeMap.iteritems()
                                if keyNode.count('_') == k.count('_') and
                                k.split('_')[-1] > lastNodeElement and
                                k.startswith(node.parentNode)
                                }
        return rightSiblingsMap

    def deleteChildNodes(self, keyParentNode):
        """
		Eliminar del árbol todos los nodos/itemsets descendientes de un nodo en particular.
		
		Argumentos:
			keyParentNode(string): clave, en el diccionario/árbol, del nodo padre cuya 
				descendencia se desea eliminar
		
		Retorna:
			Una lista con las claves de los elementos/nodos eliminados
        """
        deletedNodes = {k: v for k, v
                        in self.nodeMap.iteritems()
                        if keyParentNode.count('_') == (k.count('_') - 1) and
                        k.startswith(keyParentNode)
                        }
        self.nodeMap = {k: v for k, v
                        in self.nodeMap.iteritems()
                        if keyParentNode.count('_') != (k.count('_') - 1) or
                        not k.startswith(keyParentNode)
                        }
        return deletedNodes.keys()

    def getItemsetsCount(self):
        """
    	Obtiene el total de itemsets/nodos que integran el árbol lexicográfico actual. Se
    	excluye el nodo raíz (que no representa ningún itemset real) del conteo
    	
    	Retorna:
    		Entero con el conteo total de itemsets/nodos válidos incluidos el árbol lexicográfico
    	"""
        return (len(self.nodeMap) - 1)
