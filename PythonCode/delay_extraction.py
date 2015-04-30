from numpy import *
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
import os
import sys

################################################################################################

def plot_image(delayStore, saveDirectory, filename, year, trainID):
    fig = plt.figure()
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    delayStore = ma.masked_where(delayStore== -1, delayStore)
    cmap=plt.cm.jet_r
    cmap.set_bad(color='white')
    plt.imshow(delayStore,aspect='auto',origin='lower',interpolation='none')
    plt.ylabel('Trip number',fontsize=15)
    plt.xlabel('Station ID',fontsize=15)
#    plt.xlabel('Station code',fontsize=15)
#    plt.title('Travel time delay (mins)',fontsize = 15)
#    plt.xticks(range(0,17,3), ['MTR','PLB','FTC','SAR','HUD','CRT'])
    plt.title('Travel time delay (mins) for train '+str(trainID)+' in year '+str(year),fontsize = 15)
    plt.colorbar()
    plt.clim(0, 120)
    plt.savefig(saveDirectory+'fig_'+str(trainID)+'.pdf',bbox_inches='tight')
    plt.clf()

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
                print 'error in data record, set delay value as missing data'
                delay = -1
    return delay
    
################################################################################################

if '__main__' == __name__ :
    
    yearSet = [2011,2012,2013]
#    yearSet = [2013]

    for year in yearSet:
        saveDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/extractedData/'+str(year)+'/'
        yearDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/amtrakData/'+str(year)+'/'
        trainIDSet = [f for f in os.listdir(yearDirectory) if (1-os.path.isfile(os.path.join(yearDirectory,f)))]

        ## the data of the following trains are removed because the data are coarse.
        trainIDSet.remove('21')
        trainIDSet.remove('89')
        trainIDSet.remove('90')
        trainIDSet.remove('2175')

#        trainIDSet = ['68']
        for trainID in trainIDSet:
            loadDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/amtrakData/'+str(year)+'/'+str(trainID)+'/'
            f = os.listdir(loadDirectory)
            trips = len(f)
            ## the data of trains with less than 48 trips per year are removed.
            if trips>48:
        
                delayStore = -1*ones((trips, 50))
                tripNumber = 0
                for filename in f:
                    element = 0
                    with open(loadDirectory+filename,'r') as currentFile:
                            count = 0
                            for line in currentFile:
                                if count == 0 and (line=='* THIS TRAIN HAS EXPERIENCED CANCELLATIONS.\r\n' or line =='* THIS TRAIN HAS EXPERIENCED CANCELLATIONS.' or line == '* THIS TRAIN EXPERIENCED A SERVICE DISRUPTION.' or line == '* THIS TRAIN EXPERIENCED A SERVICE DISRUPTION.\r\n'):                                    
                                    tripNumber = tripNumber-1
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
    
                for j in reversed(range(len(delayStore[:,0]))):
                    if any(delayStore[j,:] != -1): 
                            delayStore = delayStore[0:j+1,:].copy()
                            break    
    
                save(saveDirectory+str(trainID),delayStore)
    
                plot_image(delayStore, saveDirectory, trainID, year, trainID)                    
                    

        
