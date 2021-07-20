#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from ..DecimalFraction import DecimalFraction

class FIMoTS_Node(object):
    """
	Clase utilitaria para definir la estructura de un nodo de un árbol lexicográfico. Cada nodo del árbol
	representa un itemset distinto. Un nodo padre representa un "prefijo" (subconjunto) de cada uno de sus
 	nodos hijos. Un nodo hijo es la union de los prefijos de su padre y uno de sus "tíos derechos"
 	(hermanos ubicados a la deracha de su nodo padre).
 		
 	Atributos:
    	itemPrefix (Array(String), opcional, por defecto []): Lista de items que integran al itemset
    		identificado en el nodo
    	relativeSupport (float, opcional, por defecto 0.0): Soporte relativo asociado al
    		itemset. Frecuencia con que aparece dicho prefijo en el conjunto de transacciones
    		estudiado
    	fractionElems (DecimalFraction, opcional, por defecto None): Numerador y Denominador
    		que definen el soporte relativo de un itemset.
    	parentNode (String, opcional, por defecto ''): Clave en el Diccionario/Árbol de nodos
    		que identifica al nodo padre del elemento actual
    	totalChildNodes (int, opcional, por defecto 0): Cantidad total de nodos hijos que
    		presenta el nodo actual en el árbol lexicográfico
    	lastUpdate (int, opcional, por defecto -1): Indicador temporal que especifica el 
    		momento, en la ventana de tiempo deslizante, en que el soporte relativo del
    		presente itemset ha sido actualizado por última vez 
    		
 	@author Abel Francisco (2016)
    """

    def __init__(self, itemPrefix=[], relativeSupport=0.0, fractionElems=None,
                 parentNode='', totalChildNodes=0, lastUpdate=-1):
        """
    	Inicialización de un nuevo nodo del árbol lexicográfico que representa a un itemset
    	frecuente identificado en un conjunto de transacciones.
    	
    	Argumentos:
    		itemPrefix (Array(String), opcional, por defecto []): Lista de items que integran al itemset
    			identificado en el nodo
    		relativeSupport (float, opcional, por defecto 0.0): Soporte relativo asociado al
    			itemset. Frecuencia con que aparece dicho prefijo en el conjunto de transacciones
    			estudiado
    		fractionElems (DecimalFraction, opcional, por defecto None): Numerador y Denominador
    			que definen el soporte relativo de un itemset.
    		parentNode (String, opcional, por defecto ''): Clave en el Diccionario/Árbol de nodos
    			que identifica al nodo padre del elemento actual
    		totalChildNodes (int, opcional, por defecto 0): Cantidad total de nodos hijos que
    			presenta el nodo actual en el árbol lexicográfico
   		 	lastUpdate (int, opcional, por defecto -1): Indicador temporal que especifica el 
    			momento, en la ventana de tiempo deslizante, en que el soporte relativo del
    			presente itemset ha sido actualizado por última vez 
        """
        self.itemPrefix = itemPrefix
        self.relativeSupport = relativeSupport
        self.fractionElems = fractionElems
        self.parentNode = parentNode
        self.totalChildNodes = totalChildNodes
        self.lastUpdate = lastUpdate

    def orderItemPrefix(self):
        """
    	Ordenar alfabéticamente los elementos del itemset asociado al nodo.
        """
        self.itemPrefix = sorted(self.itemPrefix)

    def getItemPrefixConcat(self):
        """
    	Obtener una cadena de caracteres que identifica al itemset, concatenando los items
    	del prefijo.
    	
    	Retorna:
    		Una cadena de caracteres que concatena los items del prefijo del itemset de forma
    		directa (sin separadores).
        """
        return ''.join(self.itemPrefix).lower()

    def getItemsetKey(self):
        """
    	Obtener una cadena de caracteres que identifica al itemset, concatenando los items
    	del prefijo con guiones bajos ('_'). Esta cadena resultante es el formato de las claves
    	que identifican a los nodos en el Diccionario/Árbol Lexicográfico.
    	
    	Retorna:
    		Una cadena de caracteres que concatena los items del prefijo del itemset, 
    		separándolos por un guión bajo ('_').
        """
        return '_'.join(self.itemPrefix)

    def __repr__(self):
        """
    	Mecanismo de presentación o escritura en pantalla de un nodo del árbol lexicográfico
    	
    	Retorna:
    		Una cadena de caracteres que representa visualmente (texto) a un nodo del árbol
    		lexicográfico
        """
        nodeFormat = '{}: [ PREFIJO: {}, SOPORTE: {}, FRACCION: {}, PADRE: {}, TOTAL HIJOS: {}, ACTUALIZACION: {} ]'
        return nodeFormat.format(self.__class__.__name__, '_'.join(self.itemPrefix),
                                 self.relativeSupport, self.fractionElems,
                                 self.parentNode, self.totalChildNodes, self.lastUpdate)

    def __eq__(self, other):
        """
    	Mecanismo para definir si dos (2) nodos del árbol lexicográfico son equivalentes.
    	Dos (2) nodos son equivalentes, si sus prefijos concatenados son exactamente
    	iguales.
    	
    	Argumentos:
    		other (FIMoTS_Node, obligatorio): Nodo con el que se quiere comparar al nodo
    			actual
    			
    	Retorna:
    		True si los nodos son equivalentes, False si no
        """
        if hasattr(other, 'itemPrefix'):
            return (self.getItemPrefixConcat() == other.getItemPrefixConcat())

    def __lt__(self, other):
        """
    	Mecanismo para definir si un (1) nodo del árbol lexicográfico es "menor" que otro.
    	Un nodo es "anterior" o "menor" que otro, si su representación escrita es previa,
    	lexicográficamente al prefijo concatenado del otro.
    	
    	Argumentos:
    		other (FIMoTS_Node, obligatorio): Nodo con el que se quiere comparar al nodo
    			actual
    			
    	Retorna:
    	    True si el nodo actual es anterior que el otro, False si no
    	"""
        return (self.getItemPrefixConcat() < other.getItemPrefixConcat())
