# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:10:04 2017

@author: Ryan
"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os

import numpy as np

import matplotlib 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation

import data_struct
import data_stack
import analyze
import file_plugins


PlotH = 4.0
PlotW = PlotH*1.61803
#----------------------------------------------------------------------        
def show_image(iev, stk):
    
    imdat = stk.absdata[:,:,int(iev)].copy() 
    #plt.imshow(imdat, cmap=matplotlib.cm.get_cmap("gray"), animated=True)
    
    return imdat

#------------------------------------------------------------------------- 
   
def deglitch_stack(stack, iev):
# from mantis_qt   def OnRemoveImage(self, event):
        
        stack.absdata = np.delete(stack.absdata, iev, axis=2)  
                   
        stack.n_ev = stack.n_ev - 1
        stack.ev = np.delete(stack.ev, iev) 

        stack.data_struct.exchange.data = stack.absdata

        stack.data_struct.exchange.energy = stack.ev
       
        iev = iev-1
        if iev < 0:
            iev = 0

    
datastruct = data_struct.h5() # initialize data structure
stk = data_stack.data(datastruct) # define stack variable
FileInternalSelection = [(0,0)] # i think this is for selecting ROIs
filepath = os.path.join("C:\\Dropbox\\Ryan\\PythonStuff\\STXMCodes\\TestData\\532_110204013","532_110204013.hdr")
plugin = file_plugins.identify(filepath) # dont quite know what this is for ...
stack = stk
file_plugins.load(filepath, stk, plugin=plugin,selection=FileInternalSelection)

show_image(20, stk)
deglitch_stack(stk,3)

#------------------------------------------------------------------------------
# work on animating images...

fig2 = plt.figure()
# slices = np.array(range(stk.n_ev))
ims = []
for i in range(stk.n_ev):
    im = plt.imshow(show_image(i, stk), animated=True)
    ims.append([im])
ani=animation.ArtistAnimation(fig2, ims, interval=50, blit=True, 
                              repeat_delay=1000)
plt.show()


# file_dataexch_hdf5.write_h5(filepath, self.data_struct) 
#fig = plt.figure()
#
#
#def f(x, y):
#    return np.sin(x) + np.cos(y)
#
#x = np.linspace(0, 2 * np.pi, 120)
#y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
#
#im = plt.imshow(f(x, y), animated=True)
#
#
#def updatefig(*args):
#    global x, y
#    x += np.pi / 15.
#    y += np.pi / 20.
#    im.set_array(f(x, y))
#    return im,
#
#ani = FuncAnimation(fig, updatefig, interval=50, blit=True)
#plt.show()

