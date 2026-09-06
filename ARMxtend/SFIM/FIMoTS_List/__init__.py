#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from ..FIMoTS_Bounds import FIMoTS_Bounds

class FIMoTS_List(object):
    """
 	Clase utilitaria que especifica la estructura de una lista de ternas, con las combinaciones distintas
	de cotas transformadoras (superior - inferior) que tienen los itemsets en una solución del algoritmo
	FIMoTS.
			
	Atributos:
    	itemsBounds (Array(FIMoTS_Bounds), opcional, por defecto []): La lista de 
    		combinaciones distintas de cotas transformadoras
        
	@author Abel Francisco (2016)
    """

    def __init__(self, itemsBounds=None):
        """
		Inicialización la estructura con la lista de cotas transformadoras (superior - inferior) 
		que tienen los itemsets en una solución del algoritmo FIMoTS.
			
		Atributos:
			itemsBounds (Array(FIMoTS_Bounds), opcional, por defecto []): La nueva lista de 
				combinaciones distintas de cotas transformadoras        
        """
        self.itemsBounds = itemsBounds if itemsBounds is not None else []

    def __repr__(self):
        """
    	Mecanismo de presentación o escritura en pantalla de una lista de cotas transformadoras
    	
    	Retorna:
    		Una cadena de caracteres que representa visualmente (texto) a una lista de cotas
    		transformadoras
        """
        boundListFormat = '{}: [ COTAS: {} ]'
        return boundListFormat.format(self.__class__.__name__, self.itemsBounds)

    def clearEmptyBounds(self):
        """
    	Eliminar aquellos pares de cotas no asociados a ningún itemset del árbol
    	"""
        newBoundList = []
        for bounds in self.itemsBounds:
            if bounds.itemNodes:
                newBoundList.append(bounds)
        self.itemsBounds = newBoundList

    def deleteBounds(self, keyNode):
        """
		Eliminar de la lista de cotas aquellas referencias a un nodo en particular.
		
		Argumentos:
			keyNode(string): clave, en el diccionario/árbol, del nodo que se desea eliminar
				de la lista de cotas
		
		Retorna:
			True, si se ha encontrado la clave del nodo en la lista, False si no
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
    	Obtiene el total de itemsets incluidos en la lista, referenciados por cada una de las
    	cotas que la integran.
    	
    	Retorna:
    		Entero con el conteo total de itemsets que son referenciados desde las cotas que
    		integran a la lista
    	"""
        totalItemsets = 0
        for bounds in self.itemsBounds:
            totalItemsets += bounds.getItemsetsCount()

        return totalItemsets

    def getItemsetsKeys(self):
        """
    	Obtiene la lista de claves en el diccionario de todos los itemsets/nodos referenciados
    	en la lista.
    	
    	Retorna:
    		Lista de cadenas de caracteres que representan las claves de cada uno de los nodos
    		referenciados por la lista de cotas actual
    	"""
        itemsetsKeys = []
        for bounds in self.itemsBounds:
            itemsetsKeys.extend(bounds.itemNodes)

        return itemsetsKeys
