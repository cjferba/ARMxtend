#!/usr/bin/env python
# -*- coding: utf-8 -*-

class WindowData(object):
    """
    Utility class to define in a generic way a sliding time window made up of data transactions.

    Attributes:
        transactionIntervals (Array(RDD, int), required): List of time intervals included in the time window. Each
            interval is a set of transactions that are received in real time, and a value that indicates the total
            number of transactions that exist at that instant
        timeWindow (int, required): Identifier of the time point at which the time window ends
        totalWindowTransacs (int, optional): total number of transactions contained in the sliding time window

    @author Abel Francisco (2016)
    """

    def __init__(self, transactionIntervals=None, timeWindow=-1):
        """
        Initialization of a new sliding time window.

        Attributes:
            transactionIntervals (Array(RDD, int), required): New list of time intervals included in the time window
            timeWindow (int, required): New identifier of the time point at which the time window ends
        """
        self.transactionIntervals = transactionIntervals if transactionIntervals is not None else []
        self.timeWindow = timeWindow
        self.totalWindowTransacs = self.countTotalWindowTransacs()

    def addTransacInterval(self, transacInterval):
        """
        Add a set of transactions (time interval) to the list of intervals that currently make up the sliding window
        (simulate an advance in time)

        Args:

            transacInterval(RDD,int): the list of transactions, defined as a time interval, that must be added to the
                current time window
        """
        self.transactionIntervals.append(transacInterval)
        self.timeWindow += 1
        self.totalWindowTransacs += transacInterval[0].count()
		
    def deleteTransacInterval(self):
        """
        Remove a set of transactions from the list of transactions that currently make up a time interval of the sliding
        window (simulate an advance in time); the first block of transactions in the window (the oldest) is removed.

        Returns:
            The number of transactions that have been removed from the window
        """
        deletedInterval = self.transactionIntervals.pop(0)
        deletedInterval[0].unpersist()
        self.totalWindowTransacs -= deletedInterval[1]
        return deletedInterval[1]
		
    def countTotalWindowTransacs(self):
        """
        Compute the total number of transactions that make up the time window, which are distributed among all the time
        intervals that make it up.

        Returns:
            The total number of transactions that make up the current time window
        """
        totalWindowTransacs = 0
        for intervals in self.transactionIntervals:
	        totalWindowTransacs += intervals[1]
        return totalWindowTransacs
        
    def getUnionTransactions(self, sparkContext):
        """
        Get the union of all the RDD files associated with the transaction intervals that make up a time window.

        Args:

            sparkContext(SparkContext): distributed context to apply Big Data techniques on an RDD with Spark

        Returns:
            The union of the RDDs of a time window (intervals) as a single RDD
        """
        windowRDDList = []
        for intervals in self.transactionIntervals:
            windowRDDList.extend([intervals[0]])
        return sparkContext.union(windowRDDList)