#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from ..FIMoTS_Node import FIMoTS_Node

class FIMoTS_Bounds(object):
    """
	Clase utilitaria para definir una terna que contiene las cotas transformadoras de un 
	itemset (algoritmo FIMoTS), y una lista con las claves en el Diccionario/Árbol 
	Lexicográfico de los nodos que comparten dichas cotas.

	Atributos:
    	upperBound (int, opcional, por defecto 0): Cota superior de transformación de un itemset
    	lowerBound (int, opcional, por defecto 0): Cota inferior de transformación de un itemset
    	itemNodes (Array(String), opcional, por defecto []): lista de claves en el Diccionario
    		que identifican a los nodos del árbol que comparten la combinación de cotas 
    		especificada en la terna
    		
	@author Abel Francisco (2016)
    """

    def __init__(self, upperBound = 0, lowerBound = 0, itemNodes = []):
        """
		Inicialización de una terna que contiene las cotas transformadoras de un 
		itemset (algoritmo FIMoTS), y los identificadores de los itemsets que las comparten.

		Atributos:
			upperBound (int, opcional, por defecto 0): Nueva cota superior de transformación de un itemset
			lowerBound (int, opcional, por defecto 0): Nueva cota inferior de transformación de un itemset
			itemNodes (Array(String), opcional, por defecto []): Nueva lista de claves en el Diccionario
				que identifican a los nodos del árbol que comparten la combinación de cotas 
				especificada en la terna
			
		@author Abel Francisco (2016)
        """
        self.upperBound	= upperBound
        self.lowerBound = lowerBound
        self.itemNodes  = itemNodes
    	
    def __repr__(self):
        """
    	Mecanismo de presentación o escritura en pantalla de una terna de cotas transformadoras
    	y sus itemsets.
    	
    	Retorna:
    		Una cadena de caracteres que representa visualmente (texto) a una combinación
    		de cotas transformadoras FIMoTS
        """
        boundFormat = '{}: [ COTA SUPERIOR: {}, COTA INFERIOR: {}, ITEMS: {} ]'
        return boundFormat.format(self.__class__.__name__, self.upperBound, 
    							  self.lowerBound, self.itemNodes)

    def getItemsetsCount(self):
    	"""
    	Obtiene el total de itemsets que comparten el par de cotas transformadoras actual
    	
    	Retorna:
    		Entero con el conteo total de itemsets que comparten el par de cotas transformadoras
    	"""
    	return len(self.itemNodes)