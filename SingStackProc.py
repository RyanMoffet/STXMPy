# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:10:04 2017

@author: Ryan
"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import math
import cmath

import numpy as np

import matplotlib 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation

from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

import data_struct
import data_stack
import analyze
import file_plugins


PlotH = 4.0
PlotW = PlotH*1.61803

datastruct = data_struct.h5() # initialize data structure
stk = data_stack.data(datastruct) # define stack variable
FileInternalSelection = [(0,0)] # i think this is for selecting ROIs
filepath = os.path.join("C:\\Dropbox\\Ryan\\PythonStuff\\STXMCodes\\TestData\\532_110204013","532_110204013.hdr")
plugin = file_plugins.identify(filepath) # dont quite know what this is for ...
stack = stk
file_plugins.load(filepath, stk, plugin=plugin,selection=FileInternalSelection)
#------------------------------------------------------------------------------

    
def ft_matrix_shift(A,dy,dx):
    # shifts matrix elements (not-integer shifts dx,dy are possible)
    # follows from TR Henn's original code...
    Ny, Nx = np.shape(A)
    rx = np.floor(Nx/2)
    fx = ((range(Nx)-rx)/(Nx/2))
    ry = np.floor(Ny/2)+1
    fy = ((range(Ny)-ry)/(Ny/2))

    px = np.fft.ifftshift(np.exp(-1j*dx*cmath.pi*fx))
    py = np.fft.ifftshift(np.exp(-1j*dy*cmath.pi*fy))
    
    yphase, xphase = np.meshgrid(py,px)
    
    yphase = np.transpose(yphase)
    xphase = np.rot90(xphase)
    
    B = abs(np.fft.ifft2(np.fft.fft2(A))*yphase*xphase)
    
    return B
#----------------------------------------------------------------------        
def show_image(iev, stk):
    
    imdat = stk.absdata[:,:,int(iev)].copy() 
    plt.imshow(imdat, cmap=matplotlib.cm.get_cmap("gray"), animated=True)

#---------------------------------------------------------------------------
def update_frame(iev, stk):
        imdat = stk.absdata[:,:,int(iev)].copy()
        return imdat
    
#---------------------------------------------------------------------------
def stack_movie(stk):
    fig2 = plt.figure()
    ims = []
    for i in range(stk.n_ev):
        im = plt.imshow(update_frame(i, stk), animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig2, ims, interval=50, blit=True, 
                                  repeat_delay=1000)
    return ani
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
            
            
# def align_stack(stack):
stackcontainer = stack.absdata

dims = np.shape(stackcontainer)

ymax = dims[0]
xmax = dims[1]
emax = dims[2]

xresloution = np.mean(np.diff(stack.x_dist))
yresolution = np.mean(np.diff(stack.x_dist))
center = np.ceil(stack.n_ev/4*3)

spectr = np.zeros(dims)

shifts = np.zeros((dims[2], 2))

for k in range(dims[2]):   
    shifts[k,:], err, phasediff = register_translation(stackcontainer[:,:,int(center)],
          stackcontainer[:,:,k], 50)
    spectr[:,:,k] = ft_matrix_shift(stackcontainer[:,:,k],-shifts[k,0],-shifts[k,1])            

shiftymax = np.ceil(np.max(shifts[:,0]))
shiftxmax = np.ceil(np.max(shifts[:,1]))
shiftymin = np.ceil(np.abs(np.min(shifts[:,0])))
shiftxmin = np.ceil(np.abs(np.min(shifts[:,1])))

shiftmatrix = np.zeros((int(ymax-shiftymin-shiftymax),int(xmax-shiftxmax-shiftxmin)
,int(emax)))

#---------------------------------------------------------------------------
# http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html#sphx-glr-auto-examples-transform-plot-register-translation-py





show_image(20, stk)
deglitch_stack(stk,3)

    
    
ani = stack_movie(stk)
plt.show()


#------------------------------------------------------------------------------
# work on animating images...




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
#--------------------------------------------------------------------------------
# save stack data as hdf5 ... 
# file_dataexch_hdf5.write_h5(filepath, self.data_struct) 