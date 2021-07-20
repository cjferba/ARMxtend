#!/usr/bin/env python
# -*- coding: utf-8 -*-


class FIMoTS_Structure(object):
    """
	Clase utilitaria pare definir una estructura que agrupe todos los componentes necesarios
	para ejecutar el algoritmo de obtención de itemsets frecuentes en ventanas de tiempo sensibles
	al tiempo, FIMoTS. 
		
	Atributos:
    	itemsetsTree (FIMoTS_Tree, opcional, por defecto None): Árbol ordenado lexicográficamente 
    		con los nodos/itemsets frecuentes identificados
    	frequentItemsetsBounds (FIMoTS_List, opcional, por defecto None): Lista con las cotas
    		transformadoras distintas de los itemsets frecuentes
    	infrequentItemsetsBounds (FIMoTS_List, opcional, por defecto None): Lista con las cotas
    		transformadoras distintas de los itemsets infrecuentes
        
	@author Abel Francisco (2016)
	"""

    def __init__(self, itemsetsTree=None, frequentItemsetsBounds=None, infrequentItemsetsBounds=None):
        """
    	Inicialización de una nueva estructura para la resolución y ejecución del algoritmo
    	FIMoTS para la identificación de itemsets frecuentes
    	
    	Argumentos:
    		itemsetsTree (FIMoTS_Tree, opcional, por defecto None): Nuevo árbol ordenado 
    			lexicográficamente con los nodos/itemsets frecuentes identificados
    		frequentItemsetsBounds (FIMoTS_List, opcional, por defecto None): Nueva lista con 
    			las cotas transformadoras de los itemsets frecuentes
    		infrequentItemsetsBounds (FIMoTS_List, opcional, por defecto None): Nueva lista 
    			con las cotas transformadoras de los itemsets infrecuentes
    	"""
        self.itemsetsTree = itemsetsTree
        self.frequentItemsetsBounds = frequentItemsetsBounds
        self.infrequentItemsetsBounds = infrequentItemsetsBounds

    def __repr__(self):
        """
    	Mecanismo de presentación o escritura en pantalla de la estructura de ejecución del
    	algoritmo FIMoTS
    	
    	Retorna:
    		Una cadena de caracteres que representa visualmente (texto) a una estructura de ejecución del
    		algoritmo FIMoTS
    	"""
        treeFormat = '{}: [ ARBOL: {}, FRECUENTES: {}, INFRECUENTES: {} ]'
        return treeFormat.format(self.__class__.__name__, self.itemsetsTree,
                                 self.frequentItemsetsBounds,
                                 self.infrequentItemsetsBounds)

    def deleteChildNodes(self, keyParentNode):
        """
		Eliminar del árbol y las listas de cotas todos los nodos/itemsets descendientes 
		de un nodo en particular.
		
		Argumentos:
			keyParentNode(string): clave, en el diccionario/árbol, del nodo padre cuya 
				descendencia se desea eliminar
        """
        deletedNodesKeys = self.itemsetsTree.deleteChildNodes(keyParentNode)
        for deletedKeyNode in deletedNodesKeys:
            isFrequent = self.frequentItemsetsBounds.deleteBounds(deletedKeyNode)
            if not isFrequent:
                self.infrequentItemsetsBounds.deleteBounds(deletedKeyNode)
