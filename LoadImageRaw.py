# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:10:04 2017

@author: Ryan
"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial 

This example shows an icon
in the titlebar of the window.

author: Jan Bodnar
website: zetcode.com 
last edited: January 2015
"""
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import data_struct

def LoadImageRaw(filedir):
    os.chdir(filedir)
    with open('532_110204013_a005.xim') as file:
        array2d=[[int(digit) for digit in line.split()] for line in file]
    imgplot=plt.imshow(array2d)

def LoadStack(filedir):
    datastruct = data_struct.h5()
    stk = data_stack.data(datastruct)
    
    
LoadImageRaw("C:\\Dropbox\\Ryan\\PythonStuff\\STXMCodes\\TestData\\532_110204013")