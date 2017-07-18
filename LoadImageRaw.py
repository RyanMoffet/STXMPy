# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:10:04 2017

@author: Ryan
"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import data_struct
import data_stack
import analyze
import file_plugins

def LoadImageRaw(filedir):
    os.chdir(filedir)
    with open('532_110204013_a005.xim') as file:
        array2d=[[int(digit) for digit in line.split()] for line in file]
    imgplot=plt.imshow(array2d)

datastruct = data_struct.h5() # initialize data structure
stk = data_stack.data(datastruct) # define stack variable
FileInternalSelection = [(0,0)] # i think this is for selecting ROIs
filepath = os.path.join("C:\\Dropbox\\Ryan\\PythonStuff\\STXMCodes\\TestData\\532_110204013","532_110204013.hdr")
plugin = file_plugins.identify(filepath) # dont quite know what this is for ...

file_plugins.load(filepath, stk, plugin=plugin,selection=FileInternalSelection)
    


LoadImageRaw("C:\\Dropbox\\Ryan\\PythonStuff\\STXMCodes\\TestData\\532_110204013")
