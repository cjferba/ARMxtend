#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Programa principal para configurar y ejecutar la activación de un algoritmo FIMoTS,
utilizando el ambiente de programación BigData Spark. Se puede emplear en conjuntos
de datos estáticos (Dataset) y, principalmente, en ambientes de Tiempo Real (Streaming).

@author Abel Francisco (2016)
"""

import sys
import os
import time
import datetime
import sh

from os import walk

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.streaming import StreamingContext

from .DecimalFraction import DecimalFraction
from .WindowData import WindowData
from .FIMoTS_Structure import FIMoTS_Structure

from .FIMoTS_Algorithm import FIMoTS_TreeInitialization
from .FIMoTS_Algorithm import FIMoTS_Main


def runStreamingFIMoTSAlgorithm(addedTransactions, slidingWindow, intervals, Actual_FIMoTS_Structure,
                                minSuppRelElements, sc):
    """
    Método para aplicar el manejo de intervalos de tiempo en la ventana deslizante, durante
    la ejecución del algoritmo FIMoTS en ambientes de flujos de datos (streaming), en
    tiempo real. Cada RDD procesado es considerado como un conjunto de transacciones recibidas
    en un instante de tiempo, y son agregadas a la ventana deslizante actual, eliminando a
    su vez las transacciones más antiguas (intervalo de tiempo más antiguo).

    Argumentos:
        addedTransactions (RDD, obligatorio): Conjunto de transacciones recibidas en el
            instante de tiempo actual, y que deben ser agregadas a la ventana deslizante
        slidingWindow (WindowData, obligatorio): Ventana deslizante de tiempo actual
        intervals (int, opcional): Total de intervalos temporales, o instantes
            de tiempo (grupos de transacciones) que integran a una ventana de tiempo
        Actual_FIMoTS_Structure (FIMoTS_Structure, obligatorio): estructura actual con la
            ejecución del algoritmo FIMoTS
        minRelSupportElems (DecimalFraction, obligatorio): mínimo Soporte relativo,
            especificado como una fracción de enteros
        sparkContext(SparkContext): contexto distribuido para aplicar técnicas de BigData
            en un RDD con Spark
    """
    isFirstWindow = (len(slidingWindow.transactionIntervals) < intervals)

    if len(slidingWindow.transactionIntervals) == intervals:
        deletedTransacsSize = slidingWindow.deleteTransacInterval()

    addedTransacsSize = addedTransactions.count()
    print("Transacciones Recibidas: {}".format(addedTransacsSize))

    slidingWindow.addTransacInterval([addedTransactions, addedTransacsSize])

    if isFirstWindow or (Actual_FIMoTS_Structure.itemsetsTree.getItemsetsCount() == 0):
        if len(slidingWindow.transactionIntervals) == intervals:
            # Inicializar el árbol FP-Tree para el algoritmo FIMoTS (conteo inicial de itemsets
            # en la primera ventana de tiempo)
            FIMoTS_TreeInitialization(Actual_FIMoTS_Structure, slidingWindow, minSuppRelElements, sc)
            isFirstWindow = False
    else:
        FIMoTS_Main(deletedTransacsSize, addedTransacsSize, Actual_FIMoTS_Structure, slidingWindow, minSuppRelElements,
                    sc)

    if not isFirstWindow:
        FIMoTS_PrintIterationResults(Actual_FIMoTS_Structure, slidingWindow)


def FIMoTS_PrintIterationResults(Actual_FIMoTS_Structure, slidingWindow):
    """
    Muestra en pantalla los resultados de una iteración del algoritmo FIMoTS aplicado sobre
    una ventana de tiempo de transacciones.

    Argumentos:
        Actual_FIMoTS_Structure (FIMoTS_Structure, obligatorio): estructura actual con la
            ejecución del algoritmo FIMoTS
        slidingWindow (WindowData, obligatorio): Ventana deslizante de tiempo actual
    """
    current_time = time.time()
    current_datetime = datetime.datetime.fromtimestamp(current_time).strftime('%d/%m/%Y %H:%M:%S')
    print("")
    print("Estado Actual - Intervalo de tiempo {} ({}):".format(slidingWindow.timeWindow, current_datetime))

    # Solo presentar total de itemsets identificados
    print("Total de Itemsets Frecuentes: {}".format(Actual_FIMoTS_Structure.frequentItemsetsBounds.getItemsetsCount()))
    print("Total de Itemsets Infrecuentes: {}".format(
        Actual_FIMoTS_Structure.infrequentItemsetsBounds.getItemsetsCount()))
    print("Total de Itemsets: {}".format(Actual_FIMoTS_Structure.itemsetsTree.getItemsetsCount()))

    # Presentar detalles sobre todos los itemsets identificados
    # print('Árbol Actual de Itemsets - {}:'.format(slidingWindow.timeWindow))
    # print Actual_FIMoTS_Structure
    # print("")
    # print("Itemsets Frecuentes: ")
    # print(Actual_FIMoTS_Structure.frequentItemsetsBounds.getItemsetsKeys())
    # print("")
    # print("Itemsets Infrecuentes: ")
    # print(Actual_FIMoTS_Structure.infrequentItemsetsBounds.getItemsetsKeys())

    # Presentar solo itemsets frecuentes en formato simple
    print("Itemsets Frecuentes:")
    freqItemsetsKeys = Actual_FIMoTS_Structure.frequentItemsetsBounds.getItemsetsKeys()
    for freqItemsetKey in freqItemsetsKeys:
        print(' '.join(freqItemsetKey.split("_")))

    # Presentar solo itemsets infrecuentes en formato simple
    # infreqItemsetsKeys = Actual_FIMoTS_Structure.infrequentItemsetsBounds.getItemsetsKeys()
    # for infreqItemsetKey in infreqItemsetsKeys:
    #        print(' '.join(infreqItemsetKey.split("_")))

    print("")


def main(minSuppNum=1, minSuppDen=3, mode='TEXTFILE', intervals=5, filesDir='/',
         threads=2, seconds=5, hostname='localhost', port=9999):
    """
    Programa Principal para ejecutar el algoritmo FIMoTS en BigData con Spark (Estático o
    en Streaming).

    Argumentos:
    	minSuppNum (int, opcional, por defecto 1): Numerador del soporte mínimo relativo
    		para que un itemset sea frecuente. Mayor que cero (0)
    	minSuppDen (int, opcional, por defecto 3): Denominador del soporte mínimo relativo
    		para que un itemset sea frecuente. Mayor que cero (0)
    	mode (String, opcional, por defecto 'TEXTFILE'): Modo de recepción de transacciones
    		sobre las cuales aplicar el algoritmo FIMoTS. 'TEXTFILE' para un conjunto de
    		datos estáticos, con las transacciones especificadas en un archivo de texto (cada
    		transacción en una línea, con los ítems separados por espacios en blanco);
    		'STREAMING' para que las transacciones sean enviadas en tiempo real, a través
    		de un contexto de Spark Streaming
    	intervals (int, opcional, por defecto 5): Total de intervalos temporales, o instantes
    		de tiempo (grupos de transacciones) que integran a una ventana de tiempo
    	filesDir (string, opcional, por defecto '/'): Dirección absoluta del directorio
    		en el que se ubican los archivos de texto con las transacciones a procesar.
    		Una (1) transacción por cada línea en el archivo.
    	threads (int, opcional, por defecto 2): Total de hilos de procesamiento empleados
    		para definir el contexto de spark
    	seconds (int, opcional, por defecto 2): Cantidad de segundos de acumulación de
    		transacciones, para generar un nuevo bloque de procesamiento
    	hostname (string, opcional, por defecto 'localhost'): Identificador del servidor
    		o host desde el que se enviarán las transacciones a estudiar
    	port (int, opcional, por defecto 9999): Número de puerto desde el que se recibirán
    		las transacciones a procesar
    """
    print("Parámetros de entrada Recibidos: ")
    print("")
    print("MINSUPPORT_NUMERATOR: '%s'" % minSuppNum)
    print("MINSUPPORT_DENOMINATOR: '%s'" % minSuppDen)
    print("MODE: '%s'" % mode)
    print("INTERVALS: %s" % intervals)
    print("THREADS: '%s'" % threads)
    print("FILES_DIRECTORY: '%s'" % filesDir)
    print("SECONDS: '%s'" % seconds)
    print("HOSTNAME: '%s'" % hostname)
    print("PORT: '%s'" % port)

    print("")

    current_time = time.time()
    current_datetime = datetime.datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H%M%S')

    minSupp = minSuppNum / float(minSuppDen)
    minSuppRelElements = DecimalFraction(minSuppNum, minSuppDen)

    # Inicializar nueva ventana de tiempo
    slidingWindow = WindowData([], 0)

    # Crear un contexto de Spark local con 'threads' hilos de ejecución
    threadsConfig = "local[%s]" % threads
    algExecName = "FIMoTS_Algorithm_" + current_datetime
    scConf = SparkConf().setAppName(algExecName).setMaster(threadsConfig).set("spark.streaming.unpersist", "false")
    sc = SparkContext(conf=scConf)

    # Solo mostrar mensajes en pantalla de Advertencia o Error (No informativos)
    sc.setLogLevel("ERROR")

    Actual_FIMoTS_Structure = FIMoTS_Structure()

    deletedTransacsSize = 0
    isFirstWindow = True

    if mode == 'STREAMING':
        # Crear un contexto de Streaming local con una periodicidad de 'seconds' segundos
        ssc = StreamingContext(sc, seconds)

        # Crear un DStream que se conectará con el host y puerto indicado
        # transactions = ssc.socketTextStream(hostname, port)

        # Definir directorio del cual tomar los archivos de datos
        transactions = ssc.textFileStream(filesDir)

        # Procesar RDDs con transacciones recibidas
        # desde el streaming en cada instante de tiempo
        transactions.foreachRDD(
            lambda rdd: runStreamingFIMoTSAlgorithm(rdd, slidingWindow, intervals, Actual_FIMoTS_Structure,
                                                    minSuppRelElements, sc))

        # Iniciar la computación en streaming
        ssc.start()
    else:
        filesNames = []

        # fileList = sc.wholeTextFiles(filesDir).collect()
        # for (filename, fileContent) in fileList:
        #	filesNames.append(filename)

        # Listar archivos de texto que especifican las
        # transacciones recibidas en cada instante de tiempo

        # Buscar en Directorio Local
        fileList = os.walk(filesDir)
        for (dirPath, dirNames, fileNames) in fileList:
            filesNames.extend([(dirPath + fileName) for fileName in fileNames])
            break

        # Buscar en directorio HDFS
        if not filesNames:
            filesNames = [line.rsplit(None, 1)[-1] for line in sh.hdfs('dfs', '-ls', filesDir).split('\n') if
                          len(line.rsplit(None, 1))][1:]
        # filesNames = esutil.hdfs.ls(hdfs_url=filesDir, recurse=False, full=False)

        # Procesar cada archivo de texto con transacciones
        # como un instante de tiempo que integrará a la
        # ventana deslizante de tiempo
        for fileName in filesNames:
            if len(slidingWindow.transactionIntervals) == intervals:
                deletedTransacsSize = slidingWindow.deleteTransacInterval()

            addedTransactions = sc.textFile(fileName)
            addedTransacsSize = addedTransactions.count()
            slidingWindow.addTransacInterval([addedTransactions, addedTransacsSize])

            if isFirstWindow:
                if len(slidingWindow.transactionIntervals) == intervals:
                    # Inicializar el árbol FP-Tree para el algoritmo FIMoTS (conteo inicial de itemsets
                    # en la primera ventana de tiempo)
                    FIMoTS_TreeInitialization(Actual_FIMoTS_Structure, slidingWindow, minSuppRelElements, sc)
                    isFirstWindow = False
            else:
                FIMoTS_Main(deletedTransacsSize, addedTransacsSize, Actual_FIMoTS_Structure, slidingWindow,
                            minSuppRelElements, sc)

            if not isFirstWindow:
                FIMoTS_PrintIterationResults(Actual_FIMoTS_Structure, slidingWindow)

    if mode == 'STREAMING':
        # Esperar a que la computación termine
        ssc.awaitTermination()


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Ejecutar algoritmo FIMoTS para obtención de Itemsets Frecuentes en " +
              "Ventanas de Tiempo Deslizante de tamaño Variable")
        print("")
        print("Uso: %s [MINSUPPORT_NUMERATOR] [MINSUPPORT_DENOMINATOR] " % sys.argv[0] +
              "[MODE] [THREADS] [SECONDS] [FILES_DIRECTORY] [HOSTNAME] [PORT]")
        print("")
        print("MINSUPPORT_NUMERATOR: Numerador entero correspondiente al mínimo " +
              "soporte aceptado para considerar a un itemset como frecuente. El " +
              "soporte mínimo debe ser expresado como una fracción de números enteros " +
              "(por defecto: 1)")
        print("MINSUPPORT_DENOMINATOR: Denominador entero correspondiente al mínimo " +
              "soporte aceptado para considerar a un itemset como frecuente (por " +
              " defecto: 3)")
        print("MODE: 'TEXTFILE' para procesamiento sobre transacciones en un archivo " +
              "de texto, 'STREAMING' para procesamiento de transacciones en tiempo " +
              "real (por defecto: TEXTFILE)")
        print("INTERVALS: Total de intervalos de tiempo (instantes) que integran a " +
              "una ventana temporal deslizante de transacciones.")
        print("FILES_DIRECTORY: dirección absoluta del directorio con los archivos de " +
              "texto con las transacciones a procesar. Una (1) transacción por cada " +
              "línea en los archivos. Solo aplica para modo de procesamiento (MODE) de " +
              "tipo 'TEXTFILE'")
        print("THREADS: Total de hilos de procesamiento empleados para definir el " +
              "contexto de streaming (por defecto: 2).")
        print("SECONDS: Cantidad de segundos de acumulación de transacciones, para " +
              "generar un nuevo bloque de procesamiento (por defecto: 5). Solo " +
              "aplica para modo de procesamiento (MODE) de tipo 'STREAMING'")
        print("HOSTNAME: Identificador del servidor o host desde el que se enviarán " +
              "las transacciones a estudiar (por defecto: localhost). Solo aplica " +
              "para modo de procesamiento (MODE) de tipo 'STREAMING'")
        print("PORT: Número de puerto desde el que se recibirán las transacciones " +
              "a procesar (por defecto: 9999). Solo aplica para modo de procesamiento " +
              "(MODE) de tipo 'STREAMING'")
        print("")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['minSuppNum'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['minSuppDen'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['mode'] = sys.argv[3]
        if len(sys.argv) > 4:
            kwargs['intervals'] = int(sys.argv[4])
        if len(sys.argv) > 5:
            kwargs['filesDir'] = sys.argv[5]
        if len(sys.argv) > 6:
            kwargs['threads'] = int(sys.argv[6])
        if len(sys.argv) > 7:
            kwargs['seconds'] = int(sys.argv[7])
        if len(sys.argv) > 8:
            kwargs['hostname'] = sys.argv[8]
        if len(sys.argv) > 9:
            kwargs['port'] = int(sys.argv[9])
        main(**kwargs)
