#!/usr/bin/env python
# -*- coding: utf-8 -*-

class WindowData(object):
    """
	Clase utilitaria para definir de forma genérica una ventana de tiempo deslizante conformada por
	transacciones de datos.

	Atributos:
    	transactionIntervals (Array(RDD, int), obligatorio): Lista de intervalos de tiempo que 
    		incluye la ventana de tiempo. Cada intervalo es un conjunto de transacciones que
    		son recibidas en tiempo real, y un valor que indica el total de transacciones que existen
    		en dicho instante
    	timeWindow (int, obligatorio): Identificador del punto temporal en el que termina 
    		la ventana de tiempo
    	totalWindowTransacs (int, opcional): cantidad total de transacciones contenidas 
    		en la ventana deslizante de tiempo
    		
	@author Abel Francisco (2016)
    """

    def __init__(self, transactionIntervals=None, timeWindow=-1):
        """
		Inicialización de una nueva ventana de tiempo deslizante.

		Attributos:
    		transactionIntervals (Array(RDD, int), obligatorio): Nueva lista de intervalos de tiempo que 
    			incluye la ventana de tiempo
			timeWindow (int, obligatorio): Nuevo identificador del punto temporal en el que termina 
				la ventana de tiempo
        """
        self.transactionIntervals = transactionIntervals if transactionIntervals is not None else []
        self.timeWindow = timeWindow
        self.totalWindowTransacs = self.countTotalWindowTransacs()

    def addTransacInterval(self, transacInterval):
        """
		Agregar un conjunto de transacciones (intervalo temporal) a la lista de intervalos
		que integran actualmente la ventana deslizante (simular un avance en el tiempo)
		
		Argumentos:
		
			transacInterval(RDD,int): la lista de transacciones, definida como un intervalo de tiempo,
				que deben ser agregadas a la ventan de tiempo actual
        """
        self.transactionIntervals.append(transacInterval)
        self.timeWindow += 1
        self.totalWindowTransacs += transacInterval[0].count()
		
    def deleteTransacInterval(self):
        """
		Eliminar un conjunto de transacciones de la lista de transacciones que integran actualmente un
		intervalo temporal de la ventana deslizante (simular un avance en el tiempo), se elimina el 
		primer bloque de transacciones en la ventana (el más antiguo).

		Retorna:
			Las cantidad de transacciones que han sido eliminadas de la ventana
        """
        deletedInterval = self.transactionIntervals.pop(0)
        deletedInterval[0].unpersist()
        self.totalWindowTransacs -= deletedInterval[1]
        return deletedInterval[1]
		
    def countTotalWindowTransacs(self):
        """
		Calcular el total de transacciones que integran la ventana de tiempo, las cuales
		se distribuyen entre todos los intervalos de tiempo que la conforman.

		Retorna:
			La cantidad total de transacciones que integran a la ventana de tiempo actual
        """
        totalWindowTransacs = 0
        for intervals in self.transactionIntervals:
	        totalWindowTransacs += intervals[1]
        return totalWindowTransacs
        
    def getUnionTransactions(self, sparkContext):
        """
        Obtener la unión de todos los archivos RDD asociados a los intervalos de transacciones
        que integran una ventana de tiempo.
        
         Argumentos:
		
			sparkContext(SparkContext): contexto distribuido para aplicar técnicas de BigData
			    en un RDD con Spark
		
		Retorna:
		    La unión de los RDD de una ventana de tiempo (intervalos) como un único RDD
        """
        windowRDDList = []
        for intervals in self.transactionIntervals:
            windowRDDList.extend([intervals[0]])
        return sparkContext.union(windowRDDList)