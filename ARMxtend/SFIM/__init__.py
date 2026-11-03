#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main program to configure and run the activation of a FIMoTS algorithm, using the BigData Spark programming
environment. It can be used on static datasets (Dataset) and, mainly, in Real-Time (Streaming) environments.

@author Abel Francisco (2016)
"""

import sys
import os
import time
import datetime

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
    Method to apply the handling of time intervals in the sliding window, during the execution of the FIMoTS algorithm in
    data stream (streaming) environments, in real time. Each processed RDD is considered as a set of transactions
    received at an instant of time, and is added to the current sliding window, while removing the oldest transactions
    (oldest time interval).

    Args:
        addedTransactions (RDD, required): Set of transactions received at the current instant of time, which must be
            added to the sliding window
        slidingWindow (WindowData, required): Current sliding time window
        intervals (int, optional): Total number of time intervals, or instants of time (groups of transactions) that
            make up a time window
        Actual_FIMoTS_Structure (FIMoTS_Structure, required): current structure with the execution of the FIMoTS
            algorithm
        minRelSupportElems (DecimalFraction, required): minimum relative support, specified as a fraction of integers
        sparkContext(SparkContext): distributed context to apply BigData techniques on an RDD with Spark
    """
    isFirstWindow = (len(slidingWindow.transactionIntervals) < intervals)

    if len(slidingWindow.transactionIntervals) == intervals:
        deletedTransacsSize = slidingWindow.deleteTransacInterval()

    addedTransacsSize = addedTransactions.count()
    print("Transacciones Recibidas: {}".format(addedTransacsSize))

    slidingWindow.addTransacInterval([addedTransactions, addedTransacsSize])

    if isFirstWindow or (Actual_FIMoTS_Structure.itemsetsTree.getItemsetsCount() == 0):
        if len(slidingWindow.transactionIntervals) == intervals:
            # Initialize the FP-Tree for the FIMoTS algorithm (initial count of itemsets
            # in the first time window)
            FIMoTS_TreeInitialization(Actual_FIMoTS_Structure, slidingWindow, minSuppRelElements, sc)
            isFirstWindow = False
    else:
        FIMoTS_Main(deletedTransacsSize, addedTransacsSize, Actual_FIMoTS_Structure, slidingWindow, minSuppRelElements,
                    sc)

    if not isFirstWindow:
        FIMoTS_PrintIterationResults(Actual_FIMoTS_Structure, slidingWindow)


def FIMoTS_PrintIterationResults(Actual_FIMoTS_Structure, slidingWindow):
    """
    Display on screen the results of an iteration of the FIMoTS algorithm applied over a time window of transactions.

    Args:
        Actual_FIMoTS_Structure (FIMoTS_Structure, required): current structure with the execution of the FIMoTS
            algorithm
        slidingWindow (WindowData, required): Current sliding time window
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

    # Show details about all the identified itemsets
    # print('Current Itemset Tree - {}:'.format(slidingWindow.timeWindow))
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
    Main program to run the FIMoTS algorithm on BigData with Spark (Static or Streaming).

    Args:
    	minSuppNum (int, optional, default 1): Numerator of the minimum relative support for an itemset to be frequent.
    		Greater than zero (0)
    	minSuppDen (int, optional, default 3): Denominator of the minimum relative support for an itemset to be
    		frequent. Greater than zero (0)
    	mode (String, optional, default 'TEXTFILE'): Mode of reception of the transactions on which to apply the FIMoTS
    		algorithm. 'TEXTFILE' for a static dataset, with the transactions specified in a text file (each
    		transaction on a line, with the items separated by blanks); 'STREAMING' for the transactions to be sent in
    		real time, through a Spark Streaming context
    	intervals (int, optional, default 5): Total number of time intervals, or instants of time (groups of
    		transactions) that make up a time window
    	filesDir (string, optional, default '/'): Absolute path of the directory where the text files with the
    		transactions to process are located. One (1) transaction per line in the file.
    	threads (int, optional, default 2): Total number of processing threads used to define the spark context
    	seconds (int, optional, default 2): Number of seconds of accumulation of transactions, to generate a new
    		processing block
    	hostname (string, optional, default 'localhost'): Identifier of the server or host from which the transactions
    		to study will be sent
    	port (int, optional, default 9999): Port number from which the transactions to process will be received
    """
    print("Input parameters received: ")
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

    minSuppRelElements = DecimalFraction(minSuppNum, minSuppDen)

    # Inicializar nueva ventana de tiempo
    slidingWindow = WindowData([], 0)

    # Create a local Spark context with 'threads' execution threads
    threadsConfig = "local[%s]" % threads
    algExecName = "FIMoTS_Algorithm_" + current_datetime
    scConf = SparkConf().setAppName(algExecName).setMaster(threadsConfig).set("spark.streaming.unpersist", "false")
    sc = SparkContext(conf=scConf)

    # Only show Warning or Error messages on screen (not informative ones)
    sc.setLogLevel("ERROR")

    Actual_FIMoTS_Structure = FIMoTS_Structure()

    deletedTransacsSize = 0
    isFirstWindow = True

    if mode == 'STREAMING':
        # Create a local Streaming context with a period of 'seconds' seconds
        ssc = StreamingContext(sc, seconds)

        # Create a DStream that will connect to the indicated host and port
        # transactions = ssc.socketTextStream(hostname, port)

        # Define the directory from which to take the data files
        transactions = ssc.textFileStream(filesDir)

        # Procesar RDDs con transacciones recibidas
        # from the streaming at each instant of time
        transactions.foreachRDD(
            lambda rdd: runStreamingFIMoTSAlgorithm(rdd, slidingWindow, intervals, Actual_FIMoTS_Structure,
                                                    minSuppRelElements, sc))

        # Start the streaming computation
        ssc.start()
    else:
        filesNames = []

        # fileList = sc.wholeTextFiles(filesDir).collect()
        # for (filename, fileContent) in fileList:
        #	filesNames.append(filename)

        # List the text files that specify the
        # transactions received at each instant of time

        # Buscar en Directorio Local
        fileList = os.walk(filesDir)
        for (dirPath, dirNames, fileNames) in fileList:
            filesNames.extend([os.path.join(dirPath, fileName) for fileName in sorted(fileNames)])
            break

        # Search in an HDFS directory (deferred import: 'sh' is only needed in this case)
        if not filesNames:
            import sh
            filesNames = [line.rsplit(None, 1)[-1] for line in sh.hdfs('dfs', '-ls', filesDir).split('\n') if
                          len(line.rsplit(None, 1))][1:]
        # filesNames = esutil.hdfs.ls(hdfs_url=filesDir, recurse=False, full=False)

        # Process each text file with transactions
        # as an instant of time that will be part of the
        # ventana deslizante de tiempo
        for fileName in filesNames:
            if len(slidingWindow.transactionIntervals) == intervals:
                deletedTransacsSize = slidingWindow.deleteTransacInterval()

            addedTransactions = sc.textFile(fileName)
            addedTransacsSize = addedTransactions.count()
            slidingWindow.addTransacInterval([addedTransactions, addedTransacsSize])

            if isFirstWindow:
                if len(slidingWindow.transactionIntervals) == intervals:
                    # Initialize the FP-Tree for the FIMoTS algorithm (initial count of itemsets
                    # in the first time window)
                    FIMoTS_TreeInitialization(Actual_FIMoTS_Structure, slidingWindow, minSuppRelElements, sc)
                    isFirstWindow = False
            else:
                FIMoTS_Main(deletedTransacsSize, addedTransacsSize, Actual_FIMoTS_Structure, slidingWindow,
                            minSuppRelElements, sc)

            if not isFirstWindow:
                FIMoTS_PrintIterationResults(Actual_FIMoTS_Structure, slidingWindow)

    if mode == 'STREAMING':
        # Wait for the computation to finish
        ssc.awaitTermination()


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Run the FIMoTS algorithm to obtain Frequent Itemsets in " +
              "Sliding Time Windows of Variable size")
        print("")
        print("Usage: %s [MINSUPPORT_NUMERATOR] [MINSUPPORT_DENOMINATOR] " % sys.argv[0] +
              "[MODE] [THREADS] [SECONDS] [FILES_DIRECTORY] [HOSTNAME] [PORT]")
        print("")
        print("MINSUPPORT_NUMERATOR: Integer numerator of the minimum support " +
              "accepted to consider an itemset as frequent. The minimum " +
              "support must be expressed as a fraction of integers " +
              "(default: 1)")
        print("MINSUPPORT_DENOMINATOR: Integer denominator of the minimum support " +
              "accepted to consider an itemset as frequent (default: 3)")
        print("MODE: 'TEXTFILE' to process transactions in a text file, " +
              "'STREAMING' to process transactions in real time " +
              "(default: TEXTFILE)")
        print("INTERVALS: Total number of time intervals (instants) that make up " +
              "a sliding time window of transactions.")
        print("FILES_DIRECTORY: absolute path of the directory with the text files " +
              "with the transactions to process. One (1) transaction per " +
              "line in the files. Only applies to processing mode (MODE) " +
              "'TEXTFILE'")
        print("THREADS: Total number of processing threads used to define the " +
              "streaming context (default: 2).")
        print("SECONDS: Number of seconds of accumulation of transactions, to " +
              "generate a new processing block (default: 5). Only " +
              "applies to processing mode (MODE) 'STREAMING'")
        print("HOSTNAME: Identifier of the server or host from which the " +
              "transactions to study will be sent (default: localhost). Only applies " +
              "to processing mode (MODE) 'STREAMING'")
        print("PORT: Port number from which the transactions to process will be " +
              "received (default: 9999). Only applies to processing mode " +
              "(MODE) 'STREAMING'")
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
