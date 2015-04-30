# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:30:38 2015

@author: Ren
"""
from numpy import *
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
import os
import sys
from copy import *
import random
import cPickle

#############################################################################

def interpolation(data):
    for i in range(len(data[:,0])):
        for j in range(len(data[i,:])):
            if data[i,j] == -1 and j == 0:
                data[i,j] = 0
            elif data[i,j] == -1:
                data[i,j] = data[i,j-1]    
    return data

##############################################################################


if '__main__' == __name__ :
    
    yearSetTrain = [2011, 2012]
    kFoldTotal = 5
    kFoldTrain = 4
    trainSetNumber = 5


    ## step 1: combine all the training data(year 2011 and 2012)

    ## get the trainID set
    year = yearSetTrain[0]
    yearDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/amtrakData/'+str(year)+'/'
    trainIDSet = [f for f in os.listdir(yearDirectory) if (1-os.path.isfile(os.path.join(yearDirectory,f)))]    
    ## remove data if not all years have record for the train
    trainIDSetCopy = deepcopy(trainIDSet)
    for trainID in trainIDSetCopy:
        for year in yearSetTrain:
            extractedDataDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/extractedData/'+str(year)+'/'
            if not os.path.exists(extractedDataDirectory+str(trainID)+'.npy'):
                try:
                    trainIDSet.remove(str(trainID))
                except ValueError:
                    pass


    saveDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/trainingData/'+str(yearSetTrain[0])+str(yearSetTrain[-1])+str(kFoldTotal)+str(kFoldTrain)+str(trainSetNumber)+'/'
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)
            
    trainIDSetCopySecond = deepcopy(trainIDSet)          
    for trainID in trainIDSetCopySecond:
        sizeCheck = True
        for year in yearSetTrain:
            extractedDataDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/extractedData/'+str(year)+'/'
            if year == yearSetTrain[0]:
                data = load(extractedDataDirectory+str(trainID)+'.npy')
                size = len(data[0])
            else:
                dataCurrent = load(extractedDataDirectory+str(trainID)+'.npy')
                sizeCurrent = len(dataCurrent[0])
                if sizeCurrent == size:
                    data = vstack((data, dataCurrent))
                else:
                    trainIDSet.remove(str(trainID))
                    sizeCheck = False
        if sizeCheck:
            data = interpolation(data)
            save(saveDirectory+str(trainID), data)
        
            ## step2: generate train data sets, using k fold crossing validation   
            totalSample = len(data[:,0])
            individualSample = totalSample/kFoldTotal
            for k in range(1, trainSetNumber+1):
                # random generate numberss
                indexList = []
                indexSet = range(kFoldTotal)
                for j in range(kFoldTrain):
                    value = random.choice(indexSet)
                    indexList.append(value)
                    indexSet.remove(value)
                n = 0
                for m in indexList:
                    if n == 0:
                        dataTraining = data[int(m*individualSample):int((m+1)*individualSample)]
                        n = n+1
                    else:
                        dataTrainingCurrent = data[int(m*individualSample):int((m+1)*individualSample)]
                        dataTraining = vstack((dataTraining, dataTrainingCurrent))
                        n = n+1
                save(saveDirectory+str(trainID)+'_'+str(k), dataTraining)
    cPickle.dump(trainIDSet, open(saveDirectory+'TrainIDSet.p','wb'))



