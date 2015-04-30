# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 14:06:16 2014

@author: Ren
"""

import sys  
from os import *
import zipfile




if '__main__' == __name__ :
    ## unzip all the files
    yearSet = [2011,2012,2013,2014]
    saveDirectory = path.abspath(path.join(getcwd(), pardir))+'/extractedData/'
        
    for year in yearSet:        
        loadDirectory = path.abspath(path.join(getcwd(), pardir))+'/amtrakData/'+str(year)+'/'
        fileList = [f for f in listdir(loadDirectory) if path.isfile(path.join(loadDirectory,f))]  
        for zipedFile in fileList:
            if zipedFile[-3:] == 'zip':
                zipHandler = zipfile.ZipFile(loadDirectory+zipedFile)
                zipHandler.extractall(loadDirectory)


        