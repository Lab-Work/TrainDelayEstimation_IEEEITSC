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
import cPickle

##############################################################################################################################

def hist_var(trainingData, currentData, forecastStep, trainID, trainID_example, index):
    saveDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/results/'
    model = VAR(trainingData)
    results = model.fit(1)
    predict = results.forecast(trainingData, forecastStep)
    true = currentData[0:forecastStep]
    predDelay = average(predict)
    trueDelay = average(true)        
    mse = ((predict-true)**2).mean() 
    mseBase = ((true-0)**2).mean()
    mae = (abs(predict-true)).mean()

    if str(trainID) == str(trainID_example) and index == 1:
        print 'var mse of train 68 is: ', mse
        print 'averge predict delay 68: ', predDelay
        print 'averge true delay 68: ', trueDelay
        print 'mse of timetable ', mseBase
        station = range(len(true[0,:]))    
            
        plt.figure(1, figsize(6,10))
        plt.rc('xtick',labelsize=15)
        plt.rc('ytick',labelsize=15)

        for k in range(5):
            plt.subplot(int(str(5)+'1'+str(k+1)))
            plt.plot(station, zeros(len(predict[k])), 'g', label = 'Scheduled timetable')
            plt.plot(station, predict[k], 'b', label = 'Historical regression model (1)')
            plt.plot(station, true[k], 'r', label = 'True delay')
            plt.ylim([-20,200])
            plt.xlabel('station code',fontsize=15)
            plt.ylabel('delay (min)',fontsize=15)
            plt.yticks([0, 50, 100, 150])
            plt.xticks(range(0,17,3), ['MTR','PLB','FTC','SAR','HUD','CRT'])
            if k == 0:
                plt.title('Five trips predicted travel time delay of train 68',fontsize=15)
                plt.legend()
        plt.savefig(saveDirectory+'fig_'+str(trainID)+'_prediction_vector.pdf',bbox_inches='tight')  
        plt.show()
        plt.clf()

    return predDelay, trueDelay, mse, mseBase, mae, true[0], predict[0] 




def online_var(trainingData, currentData, forecastStep, trainID, trainID_example,index):
    ## training
    saveDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/results/'
    stationNumber = len(trainingData[0,:])        
    trainedParameter = zeros((stationNumber,2))        
    for stationID in range(stationNumber):
        if stationID == 0:
            pass
        else:
            X = trainingData[:,stationID-1]
            X = sm.add_constant(X)
            y = trainingData[:,stationID]
            model = sm.OLS(y, X)
            results = model.fit()
            trainedParameter[stationID,0] =  results.params[0]
            trainedParameter[stationID,1] =  results.params[1]

    ## forecast
    true = currentData[0:forecastStep]            
    estDelay = zeros((forecastStep, stationNumber))            
    for k in range(forecastStep):
        for n in range(stationNumber):
            if n != 0:
                estDelay[k,n] = trainedParameter[n,0]+trainedParameter[n,1]*true[k,n-1]


    ## compute est baseline delay
    estDelayBase = zeros((forecastStep, stationNumber)) 
    estDelayBase[:,1:] = true[:,0:-1]                   

    mseEst = ((estDelay-true)**2).mean() 
    mseBase = ((estDelayBase-true)**2).mean() 
    maeEst = (abs(estDelay-true)).mean()
    maeBase = (abs(estDelayBase -true)).mean()

    if trainID == trainID_example and index == 1:
        print 'averge est delay: ', average(estDelay)        
        print 'averge est delay base: ', average(estDelayBase)        
        print 'online mse of train 68 is ', mseEst
        print 'online msebaseline of train 68 is ', mseBase

        station = range(len(true[0,:]))    
            
        plt.figure(1, figsize(6,10))
        plt.rc('xtick',labelsize=15)
        plt.rc('ytick',labelsize=15)                
        
        for k in range(5):
            plt.subplot(int(str(5)+'1'+str(k+1)))
            plt.plot(station, estDelayBase[k], 'g', label = 'Online baseline model (6)')
            plt.plot(station, estDelay[k], 'b', label = 'Online regression model (7)')
            plt.plot(station, true[k], 'r', label = 'True delay')
            plt.ylim([-20,200])
            plt.xlabel('station code',fontsize=15)
            plt.ylabel('delay (min)',fontsize=15)
            plt.yticks([0, 50, 100, 150])
            plt.xticks(range(0,17,3), ['MTR','PLB','FTC','SAR','HUD','CRT'])
            if k == 0:
                plt.title('Five trips estimated travel time delay of train 68', fontsize = 15)
                plt.legend()
        plt.savefig(saveDirectory+'fig_'+str(trainID)+'_RT_prediction_vector.pdf',bbox_inches='tight')  
        plt.show()
        plt.clf()

    return average(estDelay), average(estDelayBase), mseEst, mseBase, maeEst, maeBase, estDelay[0], estDelayBase[0]
            
def interpolation(data):
    for i in range(len(data[:,0])):
        for j in range(len(data[i,:])):
            if data[i,j] == -1 and j == 0:
                data[i,j] = 0
            elif data[i,j] == -1:
                data[i,j] = data[i,j-1]    
    return data

##############################################################################################################################

if '__main__' == __name__ :
    
    ## input

    yearSetTrain = [2011, 2012]
    yearForecast = 2013
    yearSet = [2011, 2012, 2013]
    kFoldTotal = 5
    kFoldTrain = 4
    trainSetNumber = 5

    forecastStep = 30
    trainID_example = '68'

    ## get the trainID set
    loadDirectoryYearForecast = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/extractedData/'+str(yearForecast)+'/'
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
    ## removed because there are only two stations in a trip, not suitable for vector augoregressive model
    trainIDSet.remove('52')
    trainIDSet.remove('53')


    loadDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/trainingData/'+str(yearSetTrain[0])+str(yearSetTrain[-1])+str(kFoldTotal)+str(kFoldTrain)+str(trainSetNumber)+'/'
    
    listTrainID = []

    listTrueDelay = []    
    listPredDelay = []
    listEstDelay = []
    listEstDelayBase = []

    listMSE = []    
    listMSEEst = []
    listMSEEstBase = []
    listMSETimeTable = []    
    
    listMAE = []
    listMAEEst = []
    listMAEBase = []
    
    trueT1List = []
    predictT1List = []
    estT1List = []
    estBaseT1List = []


    trainIDSetSecondCopy = deepcopy(trainIDSet)
    for trainID in trainIDSetSecondCopy:
        currentData = load(loadDirectoryYearForecast+str(trainID)+'.npy')
        currentData = interpolation(currentData)
        
        listPredDelaySample = []
        listEstDelaySample = []        
        
        listMSESample = []
        listMSEEstSample = []
        listMSEEstBaseSample = []

        listMAESample = []
        listMAEEstSample = []
        listMAEBaseSample = []

        
        saveTrigger = False
        for k in range(1, trainSetNumber+1):
            trainingData = load(loadDirectory+str(trainID)+'_'+str(k)+'.npy')
            if len(trainingData[0]) == len(currentData[0]):            
                ## historical
                predDelay, trueDelay, mse, mseTimeTable , mae, trueT1, predictT1 = hist_var(trainingData, currentData, forecastStep, trainID, trainID_example, k)
                estDelay, estDelayBase, mseEst, mseBase, maeEst, maeBase, estT1, estBaseT1 = online_var(trainingData, currentData, forecastStep, trainID, trainID_example, k)              
                
                listPredDelaySample.append(predDelay)
                listEstDelaySample.append(estDelay)

                listMSESample.append(mse)
                listMSEEstSample.append(mseEst)
                listMSEEstBaseSample.append(mseBase)

                listMAESample.append(mae)
                listMAEEstSample.append(maeEst)
                listMAEBaseSample.append(maeBase)
                
                saveTrigger = True
            else:
                trainIDSet.remove(str(trainID))
##                print 'trainID removed:', trainID
                break
            
            if k == 1:
                trueT1List = trueT1List + list(trueT1)
                predictT1List = predictT1List + list(predictT1)
                estT1List = estT1List + list(estT1)
                estBaseT1List = estBaseT1List + list(estBaseT1)
            

        if saveTrigger and average(listPredDelaySample)<60 and average(listMSEEstSample)<600 and average(listMSESample)<4000:
            listTrainID.append(trainID)
            listTrueDelay.append(trueDelay)

            listPredDelay.append(average(listPredDelaySample))
            listEstDelay.append(average(listEstDelaySample))
            listEstDelayBase.append(estDelayBase)

            listMSE.append(average(listMSESample))
            listMSEEst.append(average(listMSEEstSample))
            listMSEEstBase.append(average(listMSEEstBaseSample))

            listMAE.append(average(listMAESample))
            listMAEEst.append(average(listMAEEstSample))
            listMAEBase.append(average(listMAEBaseSample))
            
            listMSETimeTable.append(mseTimeTable)    
#        else:
#            print trainID, average(listMSESample)

    saveDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/results/'

    print 'var mse ', average(listMSE)
    print 'est mse ', average(listMSEEst)
    print 'est base mse ', average(listMSEEstBase)
    
    print 'var mae ', average(listMAE)
    print 'est mae ', average(listMAEEst)
    print 'est base mae ', average(listMAEBase)
#    
#    print average(abs(array(listPredDelay)-array(listTrueDelay)))
#    print average(abs(array(listEstDelay)-array(listTrueDelay)))
#    print average(abs(array(listEstDelayBase)-array(listTrueDelay)))

    vals = array(listMSE)
    sort_index = list(reversed(array(argsort(vals))))
    listMSERank = (array(listMSE))[sort_index]
    listMSEEstBaseRank = (array(listMSEEstBase))[sort_index]
    listMSEEstRank = (array(listMSEEst))[sort_index]


    plt.figure(1,figsize(6,4))
    plt.figure()
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.plot(range(len(listMSE)), listMSERank, label = 'Historical regression model (2)')                 
    plt.plot(range(len(listMSE)), listMSEEstBaseRank, label = 'Online baseline model (6)')                 
    plt.plot(range(len(listMSE)), listMSEEstRank, label = 'Online regression model (7)')                 
    plt.xlabel('Train Index',fontsize=20)
    plt.ylabel('MSE',fontsize=20)
    plt.ylim([0,5000])
#        plt.xlim([105,135])
    plt.legend()
    plt.savefig(saveDirectory+'fig_mse.pdf',bbox_inches='tight')  
    plt.show()
    plt.clf()


    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.plot(predictT1List,trueT1List,'o', markersize = 3)
    plt.plot(range(61),range(61),'g')
    plt.xlabel('Predicted delay',fontsize=20)
    plt.ylabel('True delay',fontsize=20)   
    plt.xlim([0,120])
    plt.ylim([0,120])     
    plt.savefig(saveDirectory+'fig_prediction_sample.pdf',bbox_inches='tight')  
    plt.show()
    plt.clf()

    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.plot(estT1List,trueT1List,'o', markersize = 3)
    plt.plot(range(61),range(61),'g')
    plt.xlabel('Estimated delay',fontsize=20)
    plt.ylabel('True delay',fontsize=20)  
    plt.xlim([0,120])
    plt.ylim([0,120])           
    plt.savefig(saveDirectory+'fig_estimation_sample.pdf',bbox_inches='tight')  
    plt.show()
    plt.clf()


    plt.plot(estBaseT1List,trueT1List,'o', markersize = 3)
    plt.plot(range(61),range(61),'g')
    plt.xlabel('Estimated delay',fontsize=20)
    plt.ylabel('True delay',fontsize=20)  
    plt.xlim([0,120])
    plt.ylim([0,120])           
    plt.savefig(saveDirectory+'fig_estimation_base_sample.pdf',bbox_inches='tight')  
    plt.show()
    plt.clf()






#
#    plt.rc('xtick',labelsize=20)
#    plt.rc('ytick',labelsize=20)
#    plt.plot(listPredDelay,listTrueDelay,'o', markersize = 3)
#    plt.plot(range(61),range(61),'g')
#    plt.xlabel('Predicted delay',fontsize=20)
#    plt.ylabel('True delay',fontsize=20)   
#    plt.xlim([0,60])
#    plt.ylim([0,60])     
#    plt.savefig(saveDirectory+'fig_prediction_sample.pdf',bbox_inches='tight')  
#    plt.show()
#    plt.clf()
#
#    plt.rc('xtick',labelsize=20)
#    plt.rc('ytick',labelsize=20)
#    plt.plot(listEstDelay,listTrueDelay,'o', markersize = 3)
#    plt.plot(range(61),range(61),'g')
#    plt.xlabel('Estimated delay',fontsize=20)
#    plt.ylabel('True delay',fontsize=20)  
#    plt.xlim([0,60])
#    plt.ylim([0,60])           
#    plt.savefig(saveDirectory+'fig_estimation_sample.pdf',bbox_inches='tight')  
#    plt.show()
#    plt.clf()
#
#
#    plt.plot(listEstDelayBase,listTrueDelay,'o', markersize = 3)
#    plt.plot(range(61),range(61),'g')
#    plt.xlabel('Estimated delay',fontsize=20)
#    plt.ylabel('True delay',fontsize=20)  
#    plt.xlim([0,60])
#    plt.ylim([0,60])           
#    plt.savefig(saveDirectory+'fig_estimation_base_sample.pdf',bbox_inches='tight')  
#    plt.show()
#    plt.clf()



