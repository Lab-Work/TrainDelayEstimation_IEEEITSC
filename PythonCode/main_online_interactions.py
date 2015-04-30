# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:04:47 2015

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

if '__main__' == __name__ :

    yearSet = [2011, 2012, 2013]    
    yearSetTrain = [2011, 2012]
    
    kFoldTotal = 5
    kFoldTrain = 4
    trainSetNumber = 5
    
    forecastStep = 7    
    sc = 2 # station
    tr = 1 # hour
    
    yearToEst = '2013' 
    dayToEst = '0101'
    
    
    trainToEst = '64'



