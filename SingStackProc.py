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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage import data
from skimage.feature import register_translation
from skimage import exposure
from skimage.feature.register_translation import _upsampled_dft
from skimage.filters import threshold_otsu
from scipy.ndimage import fourier_shift
from scipy.ndimage.filters import median_filter

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
def align_stack(stk):
    stackcontainer = stk.absdata
    
    dims = np.shape(stackcontainer)
    
    ymax = dims[0]
    xmax = dims[1]
    emax = dims[2]
    
    center = np.ceil(stk.n_ev/4*3)
    
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
    
    shiftmatrix[:,:,:] = spectr[int(shiftymax):int(ymax-shiftymin),int(shiftxmax):int(xmax-shiftxmin),:]
    
    stk.absdata=shiftmatrix
    
    return stk
    
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
    fig = plt.figure()
    cax = plt.imshow(imdat, cmap=matplotlib.cm.get_cmap("gray"), animated=True)
    fig.colorbar(cax,ticks=[np.min(imdat),np.max(imdat)])
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

            
def mat2_gray(inmat):
    
    limits = [np.min(inmat), np.max(inmat)]
    delta = limits[1] - limits[0]
    outmat = (inmat - limits[0])/delta
    return outmat
def od_part_stack(stk,method):
    #particle masking & thresholding with constant threshold condition
    # method='O' for Otsu thresholding
    # method='C' for contstant threshold (percentage of pixels above thershold)
    Xval=np.max(stk.x_dist)-np.min(stk.x_dist)
    Xvec = np.linspace(0,Xval,np.size(stk.x_dist))
    
    Yval=np.max(stk.y_dist)-np.min(stk.y_dist)
    Yvec = np.linspace(0,Xval,np.size(stk.x_dist))
    
    if method=='C':
        imagebuffer = np.mean(stk,2)
        imagebuffer = median_filter(imagebuffer,(3,3))
        GrayImage = mat2_gray(imagebuffer)
        Mask = np.zeros(np.shape(imagebuffer))
        Mask[GrayImage>=0.90] = 1
        
        plt.figure()
        plt.imshow(GrayImage, cmap=matplotlib.cm.get_cmap("gray"))        
        plt.figure()
        plt.imshow(Mask, cmap=matplotlib.cm.get_cmap("gray"))
    
    
    elif method=='O':
        imagebuffer = np.mean(stack.absdata,2)
        GrayImage = mat2_gray(imagebuffer)            
        GrayImage = exposure.adjust_gamma(GrayImage, 15)        
        Thresh = threshold_otsu(GrayImage)
        Mask = np.zeros(np.shape(imagebuffer))
        Mask[GrayImage>=Thresh] = 1
        
    Izero_Otsu = np.zeros((2, stk.n_ev))
    Izero_Otsu[0,:] = stk.ev
    for i in range(stk.n_ev):
        tempmat = stk.absdata[:,:,i]
        Izero_Otsu[1,i] = np.mean(tempmat[Mask==1])

    for i in range(stk.n_ev):
        stk.absdata[:,:,i] = -np.log(stk.absdata[:,:,i]/Izero_Otsu[1,i])
    stk.binmap = Mask

    # plotting... 
    fig, axarr = plt.subplots(2,2)
    fig.suptitle('od_part_stack Output', fontsize=14)   
    axarr[0,0].set_title('Raw Intensity Stack Mean')
    im0=axarr[0,0].imshow(imagebuffer, cmap=matplotlib.cm.get_cmap("gray"),extent=(0,Xval,0,Yval))
    divider0 = make_axes_locatable(axarr[0,0])
    cax0 = divider0.append_axes("right", size="20%", pad=0.05)
    cbar1=plt.colorbar(im0, cax=cax0, ticks=[np.min(imagebuffer), np.max(imagebuffer)])

    axarr[0,1].set_title('Optical Density Stack Mean')
    ODMean = np.mean(stk.absdata,2)
    im1=axarr[0,1].imshow(ODMean, cmap=matplotlib.cm.get_cmap("gray"))
    divider1 = make_axes_locatable(axarr[0,1])
    cax1 = divider1.append_axes("right", size="20%", pad=0.05)
    cbar1=plt.colorbar(im1, cax=cax1, ticks=[np.min(ODMean), np.max(ODMean)])
    
    axarr[1,0].set_title('I0 Region Mask')
    im1=axarr[1,0].imshow(Mask, cmap=matplotlib.cm.get_cmap("gray"))
    
    axarr[1,1].set_title('I0')
    xyplt0=axarr[1,1].plot(Izero_Otsu[0,:],Izero_Otsu[1,:],'-')
    axarr[1,1].set_xlabel('Energy (eV)')
    axarr[1,1].set_ylabel('Counts')
    
    return stk
    # plotting...

#def od_stack(stk,method)    
# create temporary variables    
stk=align_stack(stk)
stk=od_part_stack(stk, 'O')    
show_image(80, stk)
#fig = plt.figure()
#cax = plt.imshow(stk.absdata[:,:,20], cmap=matplotlib.cm.get_cmap("gray"))
#fig.colorbar(cax,ticks=[0,.5,1])


xAxisLabel = [0,np.max(stk.x_dist)-np.min(stk.x_dist)]
yAxisLabel = [0,np.max(stk.y_dist)-np.min(stk.y_dist)]



ani = stack_movie(stk)  
plt.show()          

    
#file_plugins.file_dataexch_hdf5.write_h5(filepath, data_struct)  
#file_dataexch_hdf5.write_results_h5
   
##=======
##def od_stack(stk,method)    
## create temporary variables    
#
#method='C'
#    
#stack = stk.absdata
#eVlength = stk.n_ev
#
#xAxisLabel = [0,np.max(stk.x_dist)-np.min(stk.x_dist)]
#yAxisLabel = [0,np.max(stk.y_dist)-np.min(stk.y_dist)]
#


#---------------------------------------------------------------------------
# http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html#sphx-glr-auto-examples-transform-plot-register-translation-py




#ani = stack_movie(stk)  
#plt.show()    
#
#show_image(20, stk)
#deglitch_stack(stk,3)
#    
#ani2 = stack_movie(stk)  
#plt.show()   



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