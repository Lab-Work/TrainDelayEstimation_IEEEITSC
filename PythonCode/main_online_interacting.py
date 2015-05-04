# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 15:31:26 2014

@author: Ren

"""

from numpy import *
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
import os
import sys
from statsmodels.tsa.api import *
from statsmodels.tsa.ar_model import *
import scipy.optimize as optimization
import statsmodels.api as sm
from copy import *
import random
import cPickle


##############################################################################################################################

    
def convert_time(scheduledDepart):
    if scheduledDepart[-1] == ' ' or scheduledDepart[-1] == '\r':
        if scheduledDepart[-2] == 'A':
            convertedDepart = int(scheduledDepart[0:-2])
        elif scheduledDepart[-2] == 'P':
            convertedDepart = int(scheduledDepart[0:-2])+1200
        else:
            print 'check, undefined scenario found', scheduledDepart
    elif scheduledDepart[-1] == 'A' or scheduledDepart[-1] == 'N':
        convertedDepart = int(scheduledDepart[0:-1])
    elif scheduledDepart[-1] == 'P':
        if scheduledDepart[0:2] == '12':
            convertedDepart = int(scheduledDepart[0:-1])
        elif scheduledDepart[0:2] == '11' or scheduledDepart[0:2] == '10':
            convertedDepart = int(scheduledDepart[0:-1])+1200
        else:
            print 'check, undefined scenario found', scheduledDepart
    else:
        print 'check, undefined scenario found', scheduledDepart
    return convertedDepart


def generate_trainIDSet(yearSet, yearSetTrain, yearForecast, kFoldTotal, kFoldTrain, trainSetNumber, forecastStep):    
    
    ## get the trainID set
    year = yearSetTrain[0]
    trainingDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/trainingData/'+str(yearSetTrain[0])+str(yearSetTrain[-1])+str(kFoldTotal)+str(kFoldTrain)+str(trainSetNumber)+'/'
    trainIDSet = cPickle.load(open(trainingDirectory+'TrainIDSet.p', 'rb'))
    ## remove data if not all years have record for the train
    trainIDSetCopy = deepcopy(trainIDSet)
    for trainID in trainIDSetCopy:
        for year in yearSet:
            extractedDataDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/extractedData/'+str(year)+'/'
            if not os.path.exists(extractedDataDirectory+str(trainID)+'.npy'):
                try:
                    trainIDSet.remove(str(trainID))
                except ValueError:
                    pass
    ## remove data if the data records have different dimensions among years                
    loadDirectoryYearForecast = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/extractedData/'+str(yearForecast)+'/'    
    loadDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/trainingData/'+str(yearSetTrain[0])+str(yearSetTrain[-1])+str(kFoldTotal)+str(kFoldTrain)+str(trainSetNumber)+'/'            
    trainIDSetSecondCopy = deepcopy(trainIDSet)
    for trainID in trainIDSetSecondCopy:
        currentData = load(loadDirectoryYearForecast+str(trainID)+'.npy')
        trainingData = load(loadDirectory+str(trainID)+'.npy')
        if len(trainingData[0]) == len(currentData[0]):
            pass
        else:
            trainIDSet.remove(str(trainID))
    return trainIDSet
    
def generate_dicts(trainIDSet, yearForecast):

    ## generate a dictionary of trains of their stations
    ## key: trainID     value: a list of stations associated with the train
    ## generate another dictionary of stations
    ## key: StationCode     value: (trainID; stationID index,  scheduled departure time)


    dictTrainStation = dict()
    dictStationTrainDepart = dict()
    for trainID in trainIDSet:
        if trainID not in dictTrainStation:
            dictTrainStation[trainID] = []
            
        
        
        fileDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/amtrakData/'+str(yearForecast)+'/'+str(trainID)+'/'
        fileList = [f for f in os.listdir(fileDirectory)]
        
        filename = fileList[0]
        offset = 10
        with open(fileDirectory+filename,'r') as currentFile:
            count = 0
            for line in currentFile:
                if count == 0 and (line=='* THIS TRAIN HAS EXPERIENCED CANCELLATIONS.\r\n' or line =='* THIS TRAIN HAS EXPERIENCED CANCELLATIONS.' or line == '* THIS TRAIN EXPERIENCED A SERVICE DISRUPTION.' or line == '* THIS TRAIN EXPERIENCED A SERVICE DISRUPTION.\r\n'):                                    
                    pass
                else:
                    count = count+1
                    if count > offset:
                        if line != '\r\n':
                            if line[2] != 'V' and line[3] != ' ':
                                stationCode = line[2:5]
                                dictTrainStation[trainID].append(stationCode)
                                if line[19] != '*':
                                    scheduledDepart = line[19:24]
                                    convertedScheduledDepart = convert_time(scheduledDepart)
                                    
                                    if stationCode not in dictStationTrainDepart:
                                        dictStationTrainDepart[stationCode] = zeros(3)
                                        dictStationTrainDepart[stationCode][0] = int(trainID) ## trainID
                                        dictStationTrainDepart[stationCode][1] = len(dictTrainStation[trainID])-1 ## stationID index
                                        dictStationTrainDepart[stationCode][2] = convertedScheduledDepart ## scheduled departure     
                                    else:
                                        stationData = zeros(3)
                                        stationData[0] = int(trainID)
                                        stationData[1] = len(dictTrainStation[trainID])-1
                                        stationData[2] = convertedScheduledDepart  
                                        dictStationTrainDepart[stationCode] = vstack((dictStationTrainDepart[stationCode], stationData))                                        
            del dictTrainStation[trainID][-1]
    return dictTrainStation, dictStationTrainDepart    
    
    
def generate_trainIDSetRelate(trainID, dictTrainStations, dictStationTrainDeparts, sc, tr):    

    trainIDSetRelate = []
    
    for stationID in range(1, len(dictTrainStations[trainID])):
        ## compute scheduled departure for the station
        stationCode=dictTrainStations[trainID][stationID]
        try:
            indexStationID = where(dictStationTrainDeparts[stationCode][:,0]==int(trainID))[0][0]
            scheduledDepartureStation = int(dictStationTrainDeparts[stationCode][indexStationID,2])
        except IndexError:
            scheduledDepartureStation = int(dictStationTrainDeparts[stationCode][2])
            
        ## determine related stations
        stationList = []        
        for scIndex in range(sc):
            if (scIndex+stationID) < len(dictTrainStations[trainID]):
                stationCode = dictTrainStations[trainID][stationID+scIndex]
                stationList.append(stationCode)
                
        ## extract related trainID
        for stationCode in stationList:
            try:                
                relatedDataRecordNumber = len(dictStationTrainDeparts[stationCode][:,2])
            except IndexError:
                relatedDataRecordNumber = 1

            if relatedDataRecordNumber == 1:
                if (scheduledDepartureStation-tr*100) >= 0:
                    thresholdLB = scheduledDepartureStation-tr*100
                    thresholdUB = scheduledDepartureStation
                    thresholdLB2 = -1
                    thresholdUB2 = -1                        
                else:
                    thresholdLB = 0
                    thresholdUB = scheduledDepartureStation
                    thresholdLB2 = scheduledDepartureStation-tr*100+2400
                    thresholdUB2 = 2400                       

                sd = dictStationTrainDeparts[stationCode][2]
                if thresholdLB <= sd <= thresholdUB or thresholdLB2 <= sd <= thresholdUB2: 
                    ## stationCodeRalate = stationCode
                    trainIDRelate = int(dictStationTrainDeparts[stationCode][0])
                    indexCurrent = dictTrainStations[str(trainIDRelate)].index(stationCode)
                    if str(dictTrainStations[str(trainIDRelate)][indexCurrent-1]) in dictTrainStations[trainID]:
                        if str(trainIDRelate) not in trainIDSetRelate:
                            trainIDSetRelate.append(str(trainIDRelate))
            else:
                for m in range(len(dictStationTrainDeparts[stationCode][:,2])):
                    if (scheduledDepartureStation-tr*100) >= 0:
                        thresholdLB = scheduledDepartureStation-tr*100
                        thresholdUB = scheduledDepartureStation
                        thresholdLB2 = -1
                        thresholdUB2 = -1                        
                    else:
                        thresholdLB = 0
                        thresholdUB = scheduledDepartureStation
                        thresholdLB2 = scheduledDepartureStation-tr*100+2400
                        thresholdUB2 = 2400                       
    
                    sd = dictStationTrainDeparts[stationCode][m,2]
                    if thresholdLB <= sd <= thresholdUB or thresholdLB2 <= sd <= thresholdUB2: 
                        ## stationCodeRalate = stationCode
                        trainIDRelate = int(dictStationTrainDeparts[stationCode][m,0])
                        indexCurrent = dictTrainStations[str(trainIDRelate)].index(stationCode)
                        if str(dictTrainStations[str(trainIDRelate)][indexCurrent-1]) in dictTrainStations[trainID]:                        
                            if str(trainIDRelate) not in trainIDSetRelate:
                                trainIDSetRelate.append(str(trainIDRelate))
    return trainIDSetRelate    
    
    
def date_list():    
    dateList = []
    for monthID in range(1,13):
        if monthID<10:
            monthIDStr = str(0)+str(monthID)
        else:
            monthIDStr = str(monthID)
        for dayID in range(1,32):
            if dayID<10:
                dayIDStr = str(0)+str(dayID)
            else:
                dayIDStr = str(dayID)
            index = str(monthIDStr)+str(dayIDStr)
            dateList.append(index)
    return dateList    
    
    
def generate_trainSetTheDay(yearTheDay, dayTheDay, trainIDSet):
    ## for a specific day, generate a trainIDSet that contains trainID operate on the day    
    trainIDSetTheDay = []
    theDay = str(yearTheDay)+str(dayTheDay)
    for trainIDTheDay in trainIDSet:
        directoryToCheck = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/amtrakData/'+str(yearTheDay)+'/'+str(trainIDTheDay)+'/'
        fileToCheck = trainIDTheDay+'_'+theDay+'.txt'
        if os.path.exists(directoryToCheck+fileToCheck):
            trainIDSetTheDay.append(trainIDTheDay)
    return trainIDSetTheDay
    
    
def finalize_trainIDSet(trainIDSetTheDay, trainIDSetRelated):
    trainIDSetTheDayCopy = deepcopy(trainIDSetTheDay)
    for ID in deepcopy(trainIDSetTheDay):
        if ID not in trainIDSetRelated:
            trainIDSetTheDayCopy.remove(ID)
    return trainIDSetTheDayCopy    
    
    
def compute_delay(scheduledDeparture, actualDeparture):
        if scheduledDeparture == actualDeparture:
                delay = 0
        else:
                dayS = scheduledDeparture[-1]
                dayA = actualDeparture[-1]
                lengthS = len(scheduledDeparture[0:-1])
                lengthA = len(actualDeparture[0:-1])
                if dayS == 'N':
                    dayS = 'P'
                if dayA == 'N':
                    dayA = 'P'                

                if lengthS == 4:
                        hourS = int(scheduledDeparture[0:2])
                        minuteS = int(scheduledDeparture[2:4])
                elif lengthS == 3:
                        hourS = int(scheduledDeparture[0:1])
                        minuteS = int(scheduledDeparture[1:3])

                if lengthA == 4:
                        hourA = int(actualDeparture[0:2])
                        minuteA = int(actualDeparture[2:4])
                elif lengthA == 3:
                        hourA = int(actualDeparture[0:1])
                        minuteA = int(actualDeparture[1:3])
                
                if dayS == dayA:
                        if hourA == hourS and minuteA <= minuteS: # early departure
                                delay = 0
                        elif hourA == hourS and minuteA > minuteS: # minutes delay
                                delay = minuteA-minuteS
                        elif hourA > hourS:
                                delay = (hourA-hourS)*60+minuteA-minuteS
                        elif hourA < hourS:
                                if hourS == 12:
                                        delay = (60-minuteS)+ (hourA-1)*60+ minuteA
                                else:
                                        delay = 0 # early departure
                if dayS != dayA:                    
                        if dayS == 'A' and dayA == 'P':
                                if hourS == 12:
                                        delaySS = (60-minuteS)+(23-hourS)*60
                                else:
                                        delaySS = (60-minuteS)+(11-hourS)*60
                                if hourA == 12:
                                        delayAA = 60 - minuteA
                                else:
                                        delayAA = hourA*60 + minuteA
                                delay = delaySS+delayAA

                        elif dayS =='P' and dayA == 'A':
                                if hourS == 12:
                                        delaySS = (60-minuteS)+(23-hourS)*60
                                else:
                                        delaySS = (60-minuteS)+(11-hourS)*60
                                if hourA == 12:
                                        delayAA = 60 - minuteA
                                else:
                                        delayAA = hourA*60 + minuteA
                                delay = delaySS+delayAA
                        else:
#                                print 'error in data record, set delay value as nan'
                                delay = -1
        return delay
        
def generate_trainingData_oneTrain(yearSetTrain, trainIDSetFinal, trainID):    
    
    ## generate training set
    ## get a list of days that contains data for every trainID
    dateList = date_list()
    dayList = []
    for year in yearSetTrain:    
        for dayID in dateList:
            existAll = True
            for trainIDFinal in trainIDSetFinal:                
                directoryToCheck = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/amtrakData/'+str(year)+'/'+str(trainIDFinal)+'/'
                fileToCheck = str(trainIDFinal)+'_'+str(year)+str(dayID)+'.txt'
                if os.path.exists(directoryToCheck+fileToCheck):
                    pass
                else:
                    existAll = False
                    break
            if existAll:
                dayList.append(str(year)+str(dayID))
    
    ## extract delay
    trips = len(dayList)
    if trips<=10:
        return trips
    else:
        saveDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/onlineTraining/'
        for trainIDFinal in trainIDSetFinal:
            
            delayStore = -1*ones((trips, 50))
            tripNumber = 0        
            
            for dayID in dayList:
                year = dayID[0:4]
                day = dayID[4:]
                loadDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/amtrakData/'+str(year)+'/'+str(trainIDFinal)+'/'
                filename = str(trainIDFinal)+'_'+str(year)+str(day)+'.txt'
    
                element = 0
                with open(loadDirectory+filename,'r') as currentFile:
                        count = 0
                        for line in currentFile:
                            if count == 0 and (line=='* THIS TRAIN HAS EXPERIENCED CANCELLATIONS.\r\n' or line =='* THIS TRAIN HAS EXPERIENCED CANCELLATIONS.' or line == '* THIS TRAIN EXPERIENCED A SERVICE DISRUPTION.' or line == '* THIS TRAIN EXPERIENCED A SERVICE DISRUPTION.\r\n'):                                    
                                break
                            else:
                                count = count+1
                                if count > 10:
                                    if line != '\r\n':
                                        if line[2] != 'V' and line[3] != ' ':
                                                exist = line[0:1]
                                                if exist == '*':
                                                        lineInfo = line[0:37].split()
                                                        scheduledDeparture = lineInfo[5]
                                                        actualDeparture = lineInfo[7]
                                                        if scheduledDeparture != '*' and actualDeparture != '*': 
                                                            delay = compute_delay(scheduledDeparture, actualDeparture)
                                                            delayStore[tripNumber,count-11-element] = delay
                                        else:
                                                element = element+1
                tripNumber = tripNumber+1
    
            for i in reversed(range(len(delayStore[0,:]))):
                if any(delayStore[:,i] != -1): 
                        delayStore = delayStore[:,0:i+1].copy()
                        break
    
            save(saveDirectory+str(trainIDFinal)+'_'+str(trainID),delayStore)   
    return trips  
        
        
def generate_trainingData_oneTrain_kfold(trainIDSetFinal, trainID, kFoldTotal, trainSetNumber, kFoldTrain):
    loadDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/onlineTraining/'
    data = load(loadDirectory+str(trainIDSetFinal[0])+'_'+str(trainID)+'.npy')
    totalSample = len(data[:,0])
    individualSample = totalSample/kFoldTotal
    for k in range(1, trainSetNumber+1):
        # random generate numbers
        indexList = []
        indexSet = range(kFoldTotal)
        for j in range(kFoldTrain):
            value = random.choice(indexSet)
            indexList.append(value)
            indexSet.remove(value)
        
        for trainIDFinal in trainIDSetFinal:
            data = load(loadDirectory+str(trainIDFinal)+'_'+str(trainID)+'.npy')            
            ## intepolation
            for i in range(len(data[:,0])):
                for j in range(len(data[i,:])):
                    if data[i,j] == -1 and j == 0:
                        data[i,j] = 0
                    elif data[i,j] == -1:
                        data[i,j] = data[i,j-1]
            n = 0
            for m in indexList:
                if n == 0:
                    dataTraining = data[int(m*individualSample):int((m+1)*individualSample)]
                    n = n+1
                else:
                    dataTrainingCurrent = data[int(m*individualSample):int((m+1)*individualSample)]
                    dataTraining = vstack((dataTraining, dataTrainingCurrent))
                    n = n+1
            save(loadDirectory+str(trainIDFinal)+'_'+str(k)+'_'+str(trainID), dataTraining)   
    return 
    


def extract_relatedStation(stationID, trainID, dictTrainStationsTheTrain, dictStationTrainDepartsTheTrain, sc, tr):

    stationExtractList = []
    ## compute scheduled departure for the station
    stationCode = dictTrainStationsTheTrain[trainID][stationID]
    try:
        indexStationID = where(dictStationTrainDepartsTheTrain[stationCode][:,0]==int(trainID))[0][0]
        scheduledDepartureStation = int(dictStationTrainDepartsTheTrain[stationCode][indexStationID,2])
    except IndexError:
        scheduledDepartureStation = int(dictStationTrainDepartsTheTrain[stationCode][2])

#        print '==================================='
#        print 'station ID: ', stationID, scheduledDepartureStation
        
    ## determine related stations for stationID
    stationList = []        
    for scIndex in range(1,sc):
        if (scIndex+stationID) < len(dictTrainStationsTheTrain[trainID]):
            stationCode = dictTrainStationsTheTrain[trainID][stationID+scIndex]
            stationList.append(stationCode)

    ## extract related trainID
    for stationCode in stationList:
        try:                
            relatedDataRecordNumber = len(dictStationTrainDepartsTheTrain[stationCode][:,2])
        except IndexError:
            relatedDataRecordNumber = 1

        if relatedDataRecordNumber == 1:
            if (scheduledDepartureStation-tr*100) >= 0:
                thresholdLB = scheduledDepartureStation-tr*100
                thresholdUB = scheduledDepartureStation
                thresholdLB2 = -1
                thresholdUB2 = -1                        
            else:
                thresholdLB = 0
                thresholdUB = scheduledDepartureStation
                thresholdLB2 = scheduledDepartureStation-tr*100+2400
                thresholdUB2 = 2400                       

            sd = dictStationTrainDepartsTheTrain[stationCode][2]
            if thresholdLB <= sd <= thresholdUB or thresholdLB2 <= sd <= thresholdUB2: 
                ## stationCodeRalate = stationCode
                trainIDRelate = int(dictStationTrainDepartsTheTrain[stationCode][0])
                trainIDRelateStationIndex = int(dictStationTrainDepartsTheTrain[stationCode][1])
                indexCurrent = dictTrainStations[str(trainIDRelate)].index(stationCode)
                if str(dictTrainStations[str(trainIDRelate)][indexCurrent-1]) in dictTrainStations[trainID]:                        
                    stationExtractList.append((trainIDRelate, trainIDRelateStationIndex))


        else:
            for m in range(len(dictStationTrainDepartsTheTrain[stationCode][:,2])):
                if (scheduledDepartureStation-tr*100) >= 0:
                    thresholdLB = scheduledDepartureStation-tr*100
                    thresholdUB = scheduledDepartureStation
                    thresholdLB2 = -1
                    thresholdUB2 = -1                        
                else:
                    thresholdLB = 0
                    thresholdUB = scheduledDepartureStation
                    thresholdLB2 = scheduledDepartureStation-tr*100+2400
                    thresholdUB2 = 2400                       
                    
                sd = dictStationTrainDepartsTheTrain[stationCode][m,2]
#                    if stationCode == 'SDY':
#                        print thresholdLB, thresholdUB, sd
#                        print thresholdLB <= sd <= thresholdUB

                if thresholdLB <= sd <= thresholdUB or thresholdLB2 <= sd <= thresholdUB2: 
                    trainIDRelate = int(dictStationTrainDepartsTheTrain[stationCode][m,0])
                    trainIDRelateStationIndex = int(dictStationTrainDepartsTheTrain[stationCode][m,1])
                    indexCurrent = dictTrainStations[str(trainIDRelate)].index(stationCode)
                    if str(dictTrainStations[str(trainIDRelate)][indexCurrent-1]) in dictTrainStations[trainID]:                                            
                        stationExtractList.append((trainIDRelate, trainIDRelateStationIndex))
    return stationExtractList




def generate_dateListForecast_data(trainIDSetFinal):
    ## forecast dateList
    dateListForecast = []
    dateList = date_list()
    for dateID in dateList:
        for trainIndex in trainIDSetFinal:
            existAll = True
            directoryToCheck = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/amtrakData/'+str(yearForecast)+'/'+str(trainIndex)+'/'
            fileToCheck = str(trainIndex)+'_'+str(yearForecast)+str(dateID)+'.txt'
            if os.path.exists(directoryToCheck+fileToCheck):
                pass
            else:
                existAll = False
                break
        if existAll:
            dateListForecast.append(str(yearForecast)+str(dateID))
    return dateListForecast
    
    
def generate_onlineData_oneTrain(yearForecast, trainIDSetFinal, trainID, dateListForecast):    
    
    dayList = dateListForecast

    ## extract delay
    trips = len(dayList)

    saveDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/onlineData/'

    for trainIDFinal in trainIDSetFinal:
        delayStore = -1*ones((trips, 50))
        tripNumber = 0        
        
        for dayID in dayList:
            year = dayID[0:4]
            day = dayID[4:]
            loadDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/amtrakData/'+str(year)+'/'+str(trainIDFinal)+'/'
            filename = str(trainIDFinal)+'_'+str(year)+str(day)+'.txt'

            element = 0
            with open(loadDirectory+filename,'r') as currentFile:
                    count = 0
                    for line in currentFile:
                        if count == 0 and (line=='* THIS TRAIN HAS EXPERIENCED CANCELLATIONS.\r\n' or line =='* THIS TRAIN HAS EXPERIENCED CANCELLATIONS.' or line == '* THIS TRAIN EXPERIENCED A SERVICE DISRUPTION.' or line == '* THIS TRAIN EXPERIENCED A SERVICE DISRUPTION.\r\n'):                                    
                            break
                        else:
                            count = count+1
                            if count > 10:
                                if line != '\r\n':
                                    if line[2] != 'V' and line[3] != ' ':
                                            exist = line[0:1]
                                            if exist == '*':
                                                    lineInfo = line[0:37].split()
                                                    scheduledDeparture = lineInfo[5]
                                                    actualDeparture = lineInfo[7]
                                                    if scheduledDeparture != '*' and actualDeparture != '*': 
                                                        delay = compute_delay(scheduledDeparture, actualDeparture)
                                                        delayStore[tripNumber,count-11-element] = delay
                                    else:
                                            element = element+1
            tripNumber = tripNumber+1

        for i in reversed(range(len(delayStore[0,:]))):
            if any(delayStore[:,i] != -1): 
                    delayStore = delayStore[:,0:i+1].copy()
                    break


        ## intepolation
        for i in range(len(delayStore[:,0])):
            for j in range(len(delayStore[i,:])):
                if delayStore[i,j] == -1 and j == 0:
                    delayStore[i,j] = 0
                elif delayStore[i,j] == -1:
                    delayStore[i,j] = delayStore[i,j-1]

        save(saveDirectory+str(trainIDFinal)+'_'+str(trainID),delayStore)   
    return trips  
    
    
    
def online_interact(trainID, dictTrainStationsTheTrain, dictStationTrainDepartsTheTrain, sc, tr, yearForecast,kfold):
    ## given: trainID
    trainingDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/onlineTraining/'
    trainingData = load(trainingDirectory+str(trainID)+'_'+str(kfold)+'_'+str(trainID)+'.npy')
    ## intepolation
    for i in range(len(trainingData[:,0])):
        for j in range(len(trainingData[i,:])):
            if trainingData[i,j] == -1 and j == 0:
                trainingData[i,j] = 0
            elif trainingData[i,j] == -1:
                trainingData[i,j] = trainingData[i,j-1]


    trainedParameter = dict()
    trainedParameter_sim = dict()
    for stationID in range(1, len(dictTrainStationsTheTrain[trainID])):
        stationExtractList = extract_relatedStation(stationID, trainID, dictTrainStationsTheTrain, dictStationTrainDepartsTheTrain, sc, tr)
#        print stationExtractList
        y = trainingData[:,stationID]
        X = trainingData[:,stationID-1]        
        X = sm.add_constant(X)
        
        X_sim = deepcopy(X)        
        
        lengthList = len(stationExtractList)
        
        if lengthList == 0:
            pass
        elif lengthList == 1:
            trainCode = stationExtractList[0][0]
            trainCodeStationIndex = stationExtractList[0][1]
            trainingDataRelated = load(trainingDirectory+str(trainCode)+'_'+str(kfold)+'_'+str(trainID)+'.npy')
            ## intepolation
            for i in range(len(trainingDataRelated[:,0])):
                for j in range(len(trainingDataRelated[i,:])):
                    if trainingDataRelated[i,j] == -1 and j == 0:
                        trainingDataRelated[i,j] = 0
                    elif trainingDataRelated[i,j] == -1:
                        trainingDataRelated[i,j] = trainingDataRelated[i,j-1]            
            X = column_stack((X,trainingDataRelated[:,trainCodeStationIndex]))
        else:
            for n in range(lengthList):
                trainCode = stationExtractList[n][0]
                trainCodeStationIndex = stationExtractList[n][1]
                trainingDataRelated = load(trainingDirectory+str(trainCode)+'_'+str(kfold)+'_'+str(trainID)+'.npy')
                ## intepolation
                for i in range(len(trainingDataRelated[:,0])):
                    for j in range(len(trainingDataRelated[i,:])):
                        if trainingDataRelated[i,j] == -1 and j == 0:
                            trainingDataRelated[i,j] = 0
                        elif trainingDataRelated[i,j] == -1:
                            trainingDataRelated[i,j] = trainingDataRelated[i,j-1]            
                X = column_stack((X,trainingDataRelated[:,trainCodeStationIndex]))

        model = sm.OLS(y, X)
        results = model.fit() #(alpha = 0.0, L1_wt = 1)
        
        model_sim = sm.OLS(y, X_sim)
        results_sim = model_sim.fit()
        
        if stationID not in trainedParameter:
            trainedParameter[stationID] = results.params
            trainedParameter_sim[stationID] = results_sim.params  
        else:
            print 'check'

    ## forecast

    ## for each k 
    est_interact = []
    est_sim = []
    est_true = []

    est_interact_station = []
    est_sim_station = []
    est_true_station = []

    MSE_est = []
    MSE_est_sim = []
    
    MSE_est_station = []
    MSE_est_sim_station = []

    
    for k in range(forecastStep):
        listStationInterEst = []
        listStationInterEst_sim = []
        listStationInterTrue = []

        true = load(os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/onlineData/'+str(trainID)+'_'+str(trainID)+'.npy')[k]

        estDelay = zeros(len(dictTrainStationsTheTrain[trainID]))
        estDelay_sim = zeros(len(dictTrainStationsTheTrain[trainID]))

        for stationID in range(1, len(dictTrainStationsTheTrain[trainID])):
            stationExtractList = extract_relatedStation(stationID, trainID, dictTrainStationsTheTrain, dictStationTrainDepartsTheTrain, sc, tr)
            estDelay[stationID] = trainedParameter[stationID][0]+trainedParameter[stationID][1]*true[stationID-1]
            estDelay_sim[stationID] = trainedParameter_sim[stationID][0]+trainedParameter_sim[stationID][1]*true[stationID-1]
            if len(trainedParameter[stationID]) > 2:
                for m in range(2,len(trainedParameter[stationID])):
                    trainCode = stationExtractList[m-2][0]
                    trainCodeStationIndex = stationExtractList[m-2][1]
                    trueRelate = load(os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/onlineData/'+str(trainCode)+'_'+str(trainID)+'.npy')[k]
#                    print trainedParameter[stationID][m], trueRelate[trainCodeStationIndex], trainedParameter[stationID][m]*trueRelate[trainCodeStationIndex]    
                    estDelay[stationID] = estDelay[stationID]+ trainedParameter[stationID][m]*trueRelate[trainCodeStationIndex]
                    
                listStationInterEst.append(estDelay[stationID])
                listStationInterEst_sim.append(estDelay_sim[stationID])
                listStationInterTrue.append(true[stationID])
#                print 'est: ', estDelay[stationID]
#                print 'est_sim: ', estDelay_sim[stationID]
#                print 'true: ', true[stationID]
        try:
            est_interact.append(average(estDelay))
            est_sim.append(average(estDelay_sim))
            est_true.append(average(true))

            MSE_est.append((estDelay-true).mean())
            MSE_est_sim.append((estDelay_sim-true).mean())

        
            est_interact_station.append(average(listStationInterEst))
            est_sim_station.append(average(listStationInterEst_sim))
            est_true_station.append(average(listStationInterTrue))
            
            MSE_est_station.append(((array(listStationInterEst)-array(listStationInterTrue))**2).mean())
            MSE_est_sim_station.append(((array(listStationInterEst_sim)-array(listStationInterTrue))**2).mean())            

        except:
            print 'shoud not happen a lot'
            return 0,0,0,0,0,0,0,0,0,0


    return average(est_interact), average(est_sim), average(est_true), average(MSE_est), average(MSE_est_sim), average(est_interact_station), average(est_sim_station), average(est_true_station), average(MSE_est_station), average(MSE_est_sim_station)
    

    
    
    
    
##############################################################################################################################

if '__main__' == __name__ :

    yearSet = [2011, 2012, 2013]    
    yearSetTrain = [2011, 2012]
    yearForecast = 2013

    kFoldTotal = 5
    kFoldTrain = 4
    trainSetNumber = 5
    forecastStep = 30
    
    sc = 2 # station
    tr = 1 # hour

##############################################################################################################################
    ## get trainIDSet for all the trains modeled in the system    ALL
    trainIDSet = generate_trainIDSet(yearSet, yearSetTrain, yearForecast, kFoldTotal, kFoldTrain, trainSetNumber, forecastStep)
    ## generate dicts    ALL
    dictTrainStations, dictStationTrainDeparts = generate_dicts(trainIDSet, yearForecast)

    ## generate trainIDSet for a specific day, this is the day we want to test
    yearTheDay = '2013'
    dayTheDay = '0101'
    trainIDSetTheDay = generate_trainSetTheDay(yearTheDay, dayTheDay, trainIDSet)

    trainID = '64' ## this train should be in the trainIDSetTheDay set. 
    if trainID not in trainIDSetTheDay:
        print 'Change another train'
        

    listEst = []
    listEstSim = []
    listEstMSE = []
    listEstSimMSE = []
    listTrue = []
        
    listEstR = []
    listEstSimR = []
    listEstMSER = []
    listEstSimMSER = []
    listTrueR = []


#    trainIDSetTheDay2 = ['156']
    trainIDSetTheDay.remove('156')    
    
    for trainID in trainIDSetTheDay:
        ## generate trainIDSet related to the current modeled train given sc and tr
        trainIDSetRelated = generate_trainIDSetRelate(trainID, dictTrainStations, dictStationTrainDeparts, sc, tr)
        ## this is final set of trains modeled for trainID    
        trainIDSetFinal = finalize_trainIDSet(trainIDSetTheDay, trainIDSetRelated)
#        print 'trainIDSet final: ',trainIDSetFinal
        if len(trainIDSetFinal) > 0:
        #    trainIDSetFinal = ['68','64']
            ## generate trainingData for this trainID
            trainingDataNumber = generate_trainingData_oneTrain(yearSetTrain, trainIDSetFinal, trainID)
            
            if trainingDataNumber > 40:
    #            print 'STOP not enough training data'
    
                ## generate trainingData for cross validation
                generate_trainingData_oneTrain_kfold(trainIDSetFinal, trainID, kFoldTotal, trainSetNumber, kFoldTrain)
                ## update dicts
                dictTrainStationsTheTrain, dictStationTrainDepartsTheTrain = generate_dicts(trainIDSetFinal, yearForecast)
            
                dateListForecast = generate_dateListForecast_data(trainIDSetFinal)
                generate_onlineData_oneTrain(yearForecast, trainIDSetFinal, trainID, dateListForecast)
            
                listEstSample = []
                listEstSimSample = []
                listEstMSESample = []
                listEstSimMSESample = []
                listTrueSample = []
                    
                listEstRSample = []
                listEstSimRSample = []
                listEstMSERSample = []
                listEstSimMSERSample = []
                listTrueRSample = []

                
                for kfold in range(1, trainSetNumber+1):

                    estInter, estSim, estTrue, MSEest, MSEestSim, estInterR, estSimR, estTrueR, MSEestR, MSEestSimR = online_interact(trainID, dictTrainStationsTheTrain, dictStationTrainDepartsTheTrain, sc, tr, yearForecast, kfold)
    
                    if len(trainIDSetFinal) > 1 and (1-isnan(estInterR)) and (1-isnan(MSEestR)) and MSEestR<600 and MSEestSimR<600:
                        try:
                            listEstSample.append(estInter)
                            listEstSimSample.append(estSim)
                            listEstMSESample.append(MSEest)
                            listEstSimMSESample.append(MSEestSim)
                            listTrueSample.append(estTrue)
                                
                            listEstRSample.append(estInterR)
                            listEstSimRSample.append(estSimR)
                            listEstMSERSample.append(MSEestR)
                            listEstSimMSERSample.append(MSEestSimR)
                            listTrueRSample.append(estTrueR)       
                        except:
                            pass

                try:
                    listEst.append(average(listEstSample))
                    listEstSim.append(average(listEstSim))
                    listEstMSE.append(average(listEstMSE))
                    listEstSimMSE.append(average(listEstSimMSE))
                    listTrue.append(average(listTrue))
                        
                    listEstR.append(average(listEstRSample))
                    listEstSimR.append(average(listEstSimRSample))
                    listEstMSER.append(average(listEstMSERSample))
                    listEstSimMSER.append(average(listEstSimMSERSample))
                    listTrueR.append(average(listTrueRSample))
                    
                    print 'train: ', trainID
                    print 'error: ', estInterR, estSimR               
                except:
                    pass
                



    print 'average delay: ', average(listEstR), average(listEstSimR), average(listTrueR)
    print 'average MSE: ', average(listEstMSER), average(listEstSimMSER)


    saveDirectoryFig = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/results/'


    vals = array(listEstMSER)
    sort_index = list(reversed(array(argsort(vals))))
    listEstMSERRank = (array(listEstMSER))[sort_index]
    listEstSimMSERRank = (array(listEstSimMSER))[sort_index]

    counter = 0
    for i in listEstMSERRank:
        if (isnan(i)):
            counter = counter+1
        else:
            pass

    plt.figure()
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.plot(range(len(listEstMSERRank)-counter), listEstMSERRank[counter:],'rx', label = 'Online regression interacting model (3)')                 
    plt.plot(range(len(listEstSimMSERRank)-counter), listEstMSERRank[counter:],'b+', label = 'Online regression model (7)')                    
    plt.xlabel('Train Index',fontsize=20)
    plt.ylabel('MSE',fontsize=20)
    plt.ylim([0,700])
    plt.legend()
    plt.savefig(saveDirectoryFig+'fig_mse_inter.pdf',bbox_inches='tight')  
    plt.show()
    plt.clf()

    


