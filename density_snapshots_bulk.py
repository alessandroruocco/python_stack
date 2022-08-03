#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.constants as const
from scipy.ndimage.filters import gaussian_filter1d
import time
import sdf
import pyfftw
import gc
#import multiprocessing as mp
# import scipyWavelet
import textwrap
import pathos.multiprocessing as mp

import srsUtils
from srsUtils import misc
import sdfUtils
import sdf

axis_font = {'fontname':'Arial', 'size':'22'}


#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.rc('figure', autolayout=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
#plt.rc('axes.formatter',limits=(-3,3))

# Output directory
_outputDir = 'processed'
_nCropx = (6,6)
_nCropy = (6,6)
_nProc = mp.cpu_count()

def f(x,y,k):
    Y,X = np.meshgrid(y,x)
    return np.sin(k[0]*X + k[1]*Y)

def cropArray(arr,nCropx,nCropy):
    if nCropx[1] == 0:
        xSlice = np.s_[nCropx[0]:]
    else:
        xSlice = np.s_[nCropx[0]:-nCropx[1]]

    if nCropy[1] == 0:
        ySlice = np.s_[nCropy[0]:]
    else:
        ySlice = np.s_[nCropy[0]:-nCropy[1]]

    #print("before crop {:}\nafter crop {:}".format(arr.shape,arr[xSlice,ySlice].shape))
    return arr[xSlice,ySlice]

def getData(fileName,dataName,log=False):
    if log:
        print("Reading and processing data from file \"" + fileName + "\"...")
        startTime = time.time()
    data = sdf.read(fileName)

    t = data.Header['time']
    field = cropArray(data.__dict__[dataName].data,_nCropx,_nCropy)

    if log: print("Finished reading and processing data from file, took " + str(time.time()-startTime) + "s.")
    return t,field

def readCachedData(cacheFile):
    npFile = np.load(cacheFile)
    data = npFile['processedData']
    xVar = npFile['xVar']
    yVar = npFile['yVar']

    return data,xVar,yVar

def saveCacheData(t,cacheFile,processedData,xVar,yVar):
    np.savez_compressed(cacheFile,processedData=processedData,xVar=xVar,
                        yVar=yVar,time=t)

    return

def genCacheFileName(fileName):
    cacheFileName = os.path.basename(fileName)
    cacheFileName = ''.join(cacheFileName.split('.')[:-1] + ['.npz'])

    return cacheFileName

def cachedDataExists(cacheDir,files):
    for f in files:
        cacheFileName = genCacheFileName(f)
        cacheFile = os.path.join(cacheDir,cacheFileName)

        if not os.path.isfile(cacheFile):
            return False

    return True

def listCachedFiles(cacheDir,files):
    '''
    Returns a list of .npz files associated with a list of filenames

    Associated means we assume they're associated, we don't actually check they
    are the ones that we care about.
    '''
    cachedFiles = []
    for f in files:
        cacheFileName = genCacheFileName(f)
        cacheFile = os.path.join(cacheDir,cacheFileName)

        if not os.path.isfile(cacheFile):
            raise ValueError("Couldn't find cached file {:}".format(cacheFileName))

        cachedFiles.append(cacheFile)

    return cachedFiles

def getCachedTime(cacheFile):
    npz = np.load(cacheFile)

    return npz['time']

def calcFT(data,x,y,windowFunc):
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    Nx = x.shape[0]
    Ny = y.shape[0]-1

    #if windowFunc is not None:
    #    xWindow = windowFunc(Nx)
    #    yWindow = windowFunc(Ny)
    #    window = np.outer(xWindow,yWindow)
    #    print('xWindow',len(xWindow),'yWindow',len(yWindow),len(data))
    #else:
    #    print('esle windoqw')
    #    xWindow = yWindow = 1.0

    xWindow = windowFunc(len(data[:,1]))
    yWindow = windowFunc(len(data[1,:]))
    window = np.outer(xWindow,yWindow)
    print('xWindow',len(xWindow),'yWindow',len(yWindow),data.shape)
    # Attempt to normalise so that amplitudes are correct.
    # Probably doesn't work completely but as good as it's going to get.
    ampCoeff = 2./(Nx*np.mean(xWindow)*Ny*np.mean(yWindow))
    FT = ampCoeff*np.abs(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(window*data,threads=_nProc)))

    xwns = 2*math.pi*np.fft.fftshift(np.fft.fftfreq(Nx,dx))
    ywns = 2*math.pi*np.fft.fftshift(np.fft.fftfreq(Ny,dy))

    return FT,xwns,ywns

def calcFTy(data,y,windowFunc):
    dy = y[1]-y[0]
    Ny = y.shape[0]+1

    if windowFunc is not None:
        window = windowFunc(Ny)
    else:
        window = 1.0

    # Keep amplitude of modes with k parallel to y the same, modes with k
    # parallel to x will have twice the amplitude...
    # Attempt to normalise so that amplitudes are correct.
    # Probably doesn't work completely but as good as it's going to get.
    ampCoeff = 2./(Ny*np.mean(window))

    FTy = ampCoeff*np.abs(pyfftw.interfaces.numpy_fft.fft(window*data,axis=1,threads=_nProc))
    #FTy[:,0] = 0.5*FTy[:,0]
    FTy = np.fft.fftshift(FTy,axes=1)

    ywns = 2*math.pi*np.fft.fftshift(np.fft.fftfreq(Ny,dy))

    return FTy,ywns

def calcWltx(data,x,wltFreqMult,ySkip,kSkip,kMax=None):
    Nx = x.shape[0]
    dx = x[1]-x[0]

    wlFac = 4.0*math.pi/(wltFreqMult + math.sqrt(2.+wltFreqMult**2))
    if Nx % 2 == 0:
        Nk = Nx/2+1
    else:
        Nk = (Nx+1)/2

    wns = 2.0*math.pi*np.fft.fftfreq(Nx,dx)[1:Nk:kSkip]
    if kMax is not None: wns = wns[np.where(wns < kMax)]

    Nk = len(wns)
    wls = 2.0*math.pi/wns
    scales = wls/wlFac

    wav = np.zeros((Nk,Nx))
    Ny = data.shape[1]
    NyWlt = int(math.ceil(float(Ny)/ySkip))

    print("Calculating wavelet transform")
    print(" - Transform is average over transforms of each {:}th y-strip ({:} strips transformed)".format(ySkip,NyWlt))
    print(" - Transform is performed for each {:}th Fourier frequency ({:} wavelet frequencies)".format(kSkip,int(math.ceil(float(Nk)/kSkip))))
    t1 = time.time()
    for i in range(Ny)[::ySkip]:
        t2 = time.time()
        # wav += np.abs(scipyWavelet.cwt(data[:,i],scipyWavelet.morlet2,scales/dx,w=wltFreqMult))**2
        #print("    Wavelet transform {:} of {:} took {:.3}s".format(i/ySkip + 1,NyWlt,time.time()-t2))
    wav = wav/NyWlt
    eTime = time.time()-t1
    print(" - Finished transforming {:} strips, took {:.3}s ({:.3}s each)".format(NyWlt,eTime,eTime/NyWlt))

    return wav.transpose(),x,wns

def calcEnv(data,dirn=0.0):
    '''
    Calculates a wave envelope assuming propagation at a given angle

    The wave envelope is the modulus of the analytic signal, see:
    https://en.wikipedia.org/wiki/Analytic_signal
    '''
    dataEnv = pyfftw.interfaces.numpy_fft.fftn(data,threads=8) #np.fft.fftn(data)

    dataEnv[data.shape[0]/2+1:,:] = 0.0
    #dataEnv[:,data.shape[1]/2+1:] = 0.0

    # Remove k_x = 0, k_y < 0 values
    #dataEnv[0,data.shape[1]/2+1:] = 0.0
    #dataEnv[data.shape[0]/2+1:,0] = 0.0
    dataEnv = 2.0*np.abs(pyfftw.interfaces.numpy_fft.ifftn(dataEnv,threads=8))#2.0*np.abs(np.fft.ifftn(dataEnv))

    return dataEnv

def plotGridData(fig,ax,data,dataName,x,y,t,symmetric,cmap,xLabel,yLabel,zLabel,
                 xLims=None,yLims=None,zLims=None,maxFunc=np.max,maxArgs=None,
                     minFunc=np.min,minArgs=None,log=False,aspect = 'auto',
                     noTitle=False,grid=False,noCBar=False,
                     cbOrientation='horizontal',xLabelPos='top',tickColor='k'):
    if not zLims:
        zLims = [None,None]
    else:
        zLims = list(zLims)

    if not zLims[0]:
        if symmetric:
            if zLims[1]:
                zLims[0] = -zLims[1]
            else:
                if maxArgs is not None:
                    zLims[0] = -maxFunc(np.abs(data),*maxArgs)
                else:
                    zLims[0] = -maxFunc(np.abs(data))
        elif log:
            zLims[0] = minFunc(data)
        else:
            zLims[0] = 0.0
    if not zLims[1]:
        if symmetric:
            zLims[1] = -zLims[0]
        else:
            if maxArgs is not None:
                zLims[1] = maxFunc(data,*maxArgs)
            else:
                zLims[1] = maxFunc(data)

    if log:
        cMapNorm = colors.LogNorm(vmin=zLims[0],vmax=zLims[1])
    else:
        cMapNorm = colors.Normalize(vmin=zLims[0],vmax=zLims[1])

    extent = misc.getExtent(x,y)

    #from scipy.ndimage import gaussian_filter as gaussian_filter
    #data = gaussian_filter(data,sigma=10.0)

    im = ax.imshow(np.transpose(data),extent=extent,#interpolation='none',
                   aspect=aspect,origin='lower',norm=cMapNorm,cmap=cmap)
    
    
    
    # if Log==False:
    #     cb.formatter.set_powerlimits((-4,4))
    
    if args.dataName.endswith('Poynting_Flux_x'):
         
        quantity_title = '$Sx/I_0$'
   
    elif args.dataName.endswith('Poynting_Flux_y'):

        quantity_title = '$Sy/I_0$'

    elif args.dataName.endswith('Poynting_Flux_z'):

        quantity_title = '$Sz/I_0$'

    elif args.dataName.endswith('Field_Ex'):
        quantity_title = '$e |E_x|^2/( cm_e \omega_0)$'

    elif args.dataName.endswith('Field_Ey'):

        quantity_title = '$e |E_y|^2/( cm_e \omega_0)$'
        
    elif args.dataName.endswith('Field_Ez'):

        quantity_title = '$e |E_z|^2/( cm_e \omega_0)$'
    
    elif args.dataName.endswith('Field_Bx'):
        quantity_title = '$e |B_x|^2/( m_e \omega_0)$'

    elif args.dataName.endswith('Field_By'):

        quantity_title = '$e |B_y|^2/( m_e \omega_0)$'
        
    elif args.dataName.endswith('Field_Bz'):

        quantity_title = '$e |B_z|^2/( m_e \omega_0)$'


    elif args.dataName.endswith('Density_Electron'):

        quantity_title = '$n_e/n_c$'

    elif args.dataName.endswith('Density_a1_ion') or args.dataName.endswith('carbon') or args.dataName.endswith('proton'):

        quantity_title = '$n/n_c$'


    ax.tick_params(axis='both', which='major', labelsize=80)
    if not noCBar:
        cb = fig.colorbar(im,ax=ax,orientation='vertical')     #     div = make_axes_locatable(ax)
        cb.ax.set_ylabel(quantity_title, rotation=90,labelpad=10,**axis_font) 
    
        cb.ax.yaxis.set_offset_position('left') 
        cb.update_ticks() 
        cb.ax.tick_params(labelsize=20) 
            
    ax.set_xlabel(xLabel,**axis_font) 
    ax.set_ylabel(yLabel,**axis_font)  

    #if not noCBar: cb.set_label(zLabel)

    if xLims:
        ax.set_xlim(xLims)
    if yLims:
        ax.set_ylim(yLims)

    ax.grid(grid)

    if not noCBar:
        ax.xaxis.set_label_position(xLabelPos)
        ax.xaxis.set_ticks_position(xLabelPos)
    ax.tick_params(reset=True,axis='both',color='w',direction='in',labelsize  = 20)

    if not args.noTitle:
        # fig.suptitle(args.dataName.replace('_',' ') + ', {:.3f}ps'.format(t/1e-12))
        fig.suptitle('t ={:.3f}ps'.format(t/1e-12),**axis_font)
        
    return ax

def animHelperFunc(needzLims,readDataFunc=getData,procFunc=None,rawDataName=None,rawFiles=None,
                       zLims=None,cacheDir=None,cacheFiles=None,writeCache=False,
                       minFunc=None,maxFunc=None,parallel=False):
    '''
    Helper function for animGridData

    Runs through data snapshots and either processes the data, processes and
    caches the data, or reads processed data from the cache.
    '''
    if not any(needzLims) and ((cacheFiles is None) or not writeCache):
        raise ValueError(textwrap.fill("no zLims required and don't need to generate cache. So why was this function called?",80))

    # Define function for getting processed data depending on what it is we're
    # trying to achieve
    def processProcessedData(fName):
        if cacheFiles is None or writeCache:
            t,rawData = readDataFunc(fName,rawDataName)
            processedData = procFunc(rawData)
        else:
            t = getCachedTime(fName)
            processedData = readCachedData(fName)

        if writeCache:
            cacheFile = os.path.join(cacheDir,genCacheFileName(fName))
            saveCacheData(t,cacheFile,*processedData)

        zLims = [None,None]
        if needzLims[0]: zLims[0] = minFunc(processedData[0])
        if needzLims[1]: zLims[1] = maxFunc(processedData[0])
        return t,zLims

    if cacheFiles is None or writeCache:
        files = rawFiles
    else:
        files = cacheFiles

    # Now run through files and update zLims
    # TODO: Replace with parallel version
    if parallel:
        pool = mp.Pool()
        result = pool.map(processProcessedData,files)
        pool.close()
        pool.join()

        ts,newzLims = zip(*result)
        if needzLims[0]: zLims[0] = np.min(np.array(newzLims)[:,0])
        if needzLims[1]: zLims[1] = max([0.0,np.max(np.array(newzLims)[:,1])])
    else:
        ts = []
        for f in files:
            t,newzLims = processProcessedData(f)
            print('newzLims: {:}'.format(newzLims))
            ts.append(t)

            if needzLims[0]:
                if zLims[0] is None:
                    zLims[0] = newzLims[0]
                else:
                    zLims[0] = min(newzLims[0], zLims[0])
            if needzLims[1]:
                if zLims[1] is None:
                    zLims[1] = newzLims[1]
                else:
                    zLims[1] = max(newzLims[1], zLims[1], 0.0)

    return ts,zLims

def animGridData(fig,ax,files,dataName,x,y,procDataFunc,symmetric,cmap,
                     xLabel,yLabel,zLabel,xLims=None,yLims=None,zLims=None,
                     maxFunc=np.max,minFunc=np.min,log=False,aspect='auto',
                     noTitle=False,grid=False,noCBar=False,
                     cbOrientation='horizontal',xLabelPos='top',tickColor='k',
                     overlayFunc=None,cache=False,cacheDir=None,
                     readDataFunc=getData,useExistingCache=False,parallel=False):
    '''
    Function for animating gridded image data using imshow

    Description
    ===========

    This does the dirty work of actually animating so it is fairly general. To
    speed things up the data may be first processed and saved to a cache on
    disk.

    Parameters
    ==========

    cache: Generate a cache or use existing one if useExistingCache is set
    '''
    # Setup color map limit variable
    if not zLims:
        zLims = [None,None]
    else:
        zLims = list(zLims)

    # Figure out whether we need to do an initial pass to find the colorbar
    # limits.
    needzLims = [False,False]
    if zLims[0] is None or zLims[1] is None:
        if symmetric:
            if zLims[1] is not None:
                zLims[0] = -zLims[1]
            elif zLims[0] is not None:
                zLims[1] = -zLims[0]
            else:
                # Awkward 'cos we're going to need to remember to set the lower
                # limit afterwards too
                # Also rename old function otherwise will be infinitely recursive
                _maxFunc = maxFunc
                maxFunc = lambda x: _maxFunc(np.abs(x))
                needzLims[1] = True
        elif log:
            if zLims[0] is None:
                needzLims[0] = True
            if zLims[1] is None:
                needzLims[1] = True
        else:
            if zLims[0] is None:
                zLims[0] = 0.0
            if zLims[1] is None:
                needzLims[1] = True

    # If we're using the cache then if it exists already run through and find
    # the zLims (if not supplied already). Otherwise generate the cache and
    # figure out those limits as we go.
    if cache:
        # Set up cache
        if cacheDir is None:
            dataDir = os.path.dirname(files[0])
            cacheDir = os.path.join(dataDir,'pFS_tmp')

        print("Caching enabled, using directory {:}".format(cacheDir))

        # Figure out if we can use an existing cache
        if useExistingCache and cachedDataExists(cacheDir,files):
            print("Using processed data cache to generate plot")

            # We can! Get the list of files for later use
            cachedFiles = listCachedFiles(cacheDir,files)

            # If we need to calculate colour bar limits do that now
            if any(needzLims):
                print("Performing initial pass through data to find colour bar limits")
                ts,zLims = animHelperFunc(needzLims,zLims=zLims,
                                          readDataFunc=readDataFunc,
                                          cacheDir=cacheDir,
                                          cacheFiles=cachedFiles,
                                          minFunc=minFunc,maxFunc=maxFunc,
                                          parallel=parallel)
        else:
            # Nope, we'll need to generate a new cache

            if useExistingCache:
                print(textwrap.fill("Instructed to use previously cached data but it doesn't exist.",80))
            print(textwrap.fill("Generating new cache, this may take some time...",80))

            # Make sure we're not overwriting data
            if os.path.exists(cacheDir):
                raise ValueError(textwrap.fill("Need to generate a fresh set of cached data but requested directory already exists.",80))
            else:
                os.makedirs(cacheDir)

            # Go ahead and generate new cache
            ts,zLims = animHelperFunc(needzLims,procFunc=procDataFunc,
                                          readDataFunc=readDataFunc,
                                      rawDataName=dataName,rawFiles=files,
                                      zLims=zLims,cacheDir=cacheDir,
                                      writeCache=True,minFunc=minFunc,
                                      maxFunc=maxFunc,parallel=parallel)

            # Now that we've generated the cache, get the list of files
            cachedFiles = listCachedFiles(cacheDir,files)
    elif any(needzLims):
        # We're not using the cache but we do need to figure out the colour bar
        # limits. This is unfortunate as we're going to have to process the data
        # twice but for whatever reason you chose to go down this route. Don't
        # tell me I didn't warn you. Some people just never learn do they.
        print("Performing initial pass through data to find colour bar limits")
        ts,zLims = animHelperFunc(needzLims,procFunc=procDataFunc,
                                          readDataFunc=readDataFunc,
                                  rawDataName=dataName,rawFiles=files,
                                  zLims=zLims,minFunc=minFunc,maxFunc=maxFunc,
                                  parallel=parallel)

    if any(needzLims):
        if symmetric:
            if zLims[1] is None:
                raise ValueError(textwrap.fill("Symmetric colour bar upper limit should have been found earlier on in this function",80))
            if zLims[0] is None:
                zLims[0] = -zLims[1]
        else:
            if zLims[0] is None:
                raise ValueError(textwrap.fill("Colour bar upper limit should have been found earlier on in this function",80))
            if zLims[0] is None:
                raise ValueError(textwrap.fill("Colour bar lower limit should have been found earlier on in this function",80))

        print("Completed initial pass, colour bar limits: {:}\n".format(zLims))

    # Plot first frame of animation. We need to do this to have the image object
    # returned by imshow for later use by the animation function. Also set up
    # other aspects of the plot (labels etc.).
    print("Generating animation")
    #t = ts[0]
    if cache:
        data,x,y = readCachedData(cachedFiles[0])
    else:
        _,data = readDataFunc(files[0],dataName)
        data,x,y = procDataFunc(data)

    extent = misc.getExtent(x,y)

    if log:
        cMapNorm = colors.LogNorm(vmin=zLims[0],vmax=zLims[1])
    else:
        cMapNorm = colors.Normalize(vmin=zLims[0],vmax=zLims[1])

    im = ax.imshow(np.transpose(data),interpolation='none',extent=extent,
                   aspect=aspect,origin='lower',norm=cMapNorm,cmap=cmap)

    if not noCBar:
        cb = fig.colorbar(im,orientation=cbOrientation)

    ax.set_xlabel(xLabel,**axis_font)
    ax.set_ylabel(yLabel,**axis_font)
    if not noCBar: cb.set_label(zLabel)

    if overlayFunc is not None:
        overlayFunc(ax,x,y)

    timeText = ax.set_title('')

    if xLims:
        ax.set_xlim(xLims)
    if yLims:
        ax.set_ylim(yLims)

    ax.grid(grid)

    if not noCBar:
        ax.xaxis.set_label_position(xLabelPos)
        ax.xaxis.set_ticks_position(xLabelPos)
    ax.tick_params(reset=True,axis='both',color=tickColor)

    # Define animation function. This does the actual plotting of each frame.
    def animate(i):
        #print(i)
        if cache:
            t = getCachedTime(cachedFiles[i])
            processedData = np.transpose(readCachedData(cachedFiles[i])[0])
        else:
            t,rawData = readDataFunc(files[i],dataName)
            processedData = np.transpose(procDataFunc(rawData)[0])

        im.set_array(processedData)
        if not noTitle:
            timeText.set_text(dataName.replace('_',' ') + ', {:.3f}ps'.format(t/1e-12))
        else:
            timeText.set_text('{:.3f}ps'.format(t/1e-12))
        return im,timeText

    # Call FuncAnimation to go animate
    anim = animation.FuncAnimation(fig,animate,frames=len(files),blit=True)

    return anim

def plotTPDWNs(ax,Te,ne,wnNorm,linewidth=0.8,linecolor='w'):
    # Plot TPD maximum growth wavenumbers
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()
    
    print(Te,ne)

    wns1 = srsUtils.tpd.wns(ne,Te)['k1']/wnNorm
    wns2 = srsUtils.tpd.wns(ne,Te)['k2']/wnNorm

    ax.plot( wns1[:,0], wns1[:,1],linecolor,linewidth=linewidth)
    ax.plot( wns1[:,0],-wns1[:,1],linecolor,linewidth=linewidth)
    #ax.plot(-wns1[:,0], wns1[:,1],linecolor,linewidth=linewidth)
    #ax.plot(-wns1[:,0],-wns1[:,1],linecolor,linewidth=linewidth)
    ax.plot( wns2[:,0], wns2[:,1],linecolor,linewidth=linewidth)
    ax.plot( wns2[:,0],-wns2[:,1],linecolor,linewidth=linewidth)
    #ax.plot(-wns2[:,0], wns2[:,1],linecolor,linewidth=linewidth)
    #ax.plot(-wns2[:,0],-wns2[:,1],linecolor,linewidth=linewidth)

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotTPDyWNsVsx(ax,x,Te,ne,wnNorm,linewidth=0.8):
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    # Same for both TPD daughter waves
    wnPerp = srsUtils.tpd.wns(ne,Te)['k1'][:,1]

    ax.plot(x, wnPerp/wnNorm,'w-',linewidth=linewidth)
    ax.plot(x,-wnPerp/wnNorm,'w-',linewidth=linewidth)

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotTPDxWNsVsx(ax,x,Te,ne,wnNorm):
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    # Calculate k_x for forward and backward propagating waves
    op  = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
    vth = np.sqrt(const.k*Te/const.m_e)
    bth = vth/const.c
    opRel = op*math.sqrt(1.-2.5*bth**2)
    ld  = vth/opRel

    wnParaf = srsUtils.tpd.wnsNodim(bth,srsUtils.omegaNIF/opRel)['k1'][:,0]/ld
    wnParab = srsUtils.tpd.wnsNodim(bth,srsUtils.omegaNIF/opRel)['k2'][:,0]/ld

    # We're overlaying a snapshot in time, so ignore propagation direction
    ax.plot(x, wnParaf/wnNorm,'w-')
    ax.plot(x,-wnParaf/wnNorm,'w-')
    ax.plot(x, wnParab/wnNorm,'w-')
    ax.plot(x,-wnParab/wnNorm,'w-')

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotSRSWNs(ax,Te,ne,wnNorm,EPW=True,EM=True,linewidth=0.8,linecolor='w'):
    # Plot SRS wavenumbers
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    Ntheta = 200

    bth = np.sqrt(const.k*Te/const.m_e)/const.c
    OpMin = np.sqrt(np.min(ne)/srsUtils.nCritNIF)
    OpMax = np.sqrt(np.max(ne)/srsUtils.nCritNIF)

    if EPW:
        thetas = np.linspace(0.0,2.0*math.pi,Ntheta)
        KMin = srsUtils.srs.wnsNodimMatchTheta(OpMin,bth,thetas,relativistic=True)
        kMin = np.array(KMin)*srsUtils.wnVacNIF/wnNorm
        ax.plot(kMin[0,:], kMin[1,:],linecolor,linewidth=linewidth)

        if OpMax < 0.5:
            KMax = srsUtils.srs.wnsNodimMatchTheta(OpMax,bth,thetas,relativistic=True)
            kMax = np.array(KMax)*srsUtils.wnVacNIF/wnNorm
            ax.plot(kMax[0,:], kMax[1,:],linecolor,linewidth=linewidth)

    if EM:
        thetas = np.linspace(0.5*math.pi,1.5*math.pi,Ntheta)
        K0Min = np.sqrt(1. - OpMin**2*(1.-2.5*bth**2))
        K0Max = np.sqrt(1. - OpMax**2*(1.-2.5*bth**2))

        KMin = srsUtils.srs.wnsNodimMatchTheta(OpMin,bth,thetas,relativistic=True)
        KsMin = -np.copy(KMin)
        KsMin[0,:] = K0Min + KsMin[0,:]
        ksMin = np.array(KsMin)*srsUtils.wnVacNIF/wnNorm
        ax.plot(ksMin[0,:], ksMin[1,:],linecolor,linewidth=linewidth)

        if OpMax < 0.5:
            KMax = srsUtils.srs.wnsNodimMatchTheta(OpMax,bth,thetas,relativistic=True)
            KsMax = -np.copy(KMax)
            KsMax[0,:] = K0Max + KsMax[0,:]
            ksMax = np.array(KsMax)*srsUtils.wnVacNIF/wnNorm
            ax.plot(ksMax[0,:], ksMax[1,:],linecolor,linewidth=linewidth)

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotSSRSyWNsVsx(ax,x,Te,ne,wnNorm,linewidth=0.8):
    # Plot SRS wavenumbers
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    Ntheta = 200

    bth = np.sqrt(const.k*Te/const.m_e)/const.c
    Op = np.sqrt(ne/srsUtils.nCritNIF)

    theta = 0.5*math.pi
    _,Ky = srsUtils.srs.wnsNodimMatchTheta(Op,bth,theta,relativistic=True)
    ax.plot(x, Ky*srsUtils.omegaNIF/const.c/wnNorm,'w--',linewidth=linewidth)
    ax.plot(x,-Ky*srsUtils.omegaNIF/const.c/wnNorm,'w--',linewidth=linewidth)

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotSRSWNsVsx(ax,x,Te,ne,wnNorm):
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    # Calculate k_x for forward and backscatter EPWs
    op  = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
    bth = math.sqrt(const.k*Te/const.m_e)/const.c
    OpRel = op*math.sqrt(1. - 2.5*bth**2)/srsUtils.omegaNIF
    K0 = np.sqrt(1. - OpRel**2)

    wnParab = srsUtils.srs.wnsNodimMatch(OpRel,bth,K0)['kb']
    wnParaf = srsUtils.srs.wnsNodimMatch(OpRel,bth,K0)['kf']

    wnParab = wnParab*srsUtils.omegaNIF/const.c
    wnParaf = wnParaf*srsUtils.omegaNIF/const.c

    ax.plot(x, wnParaf/wnNorm,'w-')
    ax.plot(x,-wnParaf/wnNorm,'w-')
    ax.plot(x, wnParab/wnNorm,'w-')
    ax.plot(x,-wnParab/wnNorm,'w-')

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotLandauCutoffWNs(ax,Te,ne,wnNorm,linecolor='w',linewidth=0.8):
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    # Plot Landau damping cutoff
    ldMin = math.sqrt(const.k*Te*const.epsilon_0/(ne[-1]*const.e**2))
    ldMax = math.sqrt(const.k*Te*const.epsilon_0/(ne[0]*const.e**2))

    minCutoff = 0.3/ldMax/wnNorm
    maxCutoff = 0.3/ldMin/wnNorm

    circle1 = plt.Circle((0.0,0.0),minCutoff,color=linecolor,linestyle='--',fill=False,linewidth=linewidth)
    #circle2 = plt.Circle((0.0,0.0),maxCutoff,color=linecolor,linestyle='--',fill=False,linewidth=linewidth)
    ax.add_artist(circle1)
    #ax.add_artist(circle2)

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotLandauCutoffyWNsVsx(ax,x,Te,ne,wnNorm):
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    ld = np.sqrt(const.k*Te*const.epsilon_0/(ne*const.e**2))
    cutoff = 0.3/ld/wnNorm

    ax.plot(x, cutoff,'g--')
    ax.plot(x,-cutoff,'g--')

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotTPDLandauCutoffx(ax,x,Te,ne,wnNorm,color='w',linewidth=None):
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    bth = math.sqrt(const.k*Te/const.m_e)/const.c
    neCutoff = srsUtils.tpd.landauCutoffDens(bth,relativistic=True,cutoff=0.3)

    diffs = np.abs(ne/srsUtils.nCritNIF-neCutoff)
    loc = x[np.where(diffs == np.min(diffs))[0]]

    ax.axvline(loc,color=color,linestyle='--',linewidth=linewidth)

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotSRSLandauCutoffx(ax,x,Te,ne,wnNorm,color='w',linewidth=None):
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    bth = math.sqrt(const.k*Te/const.m_e)/const.c
    neCutoff = srsUtils.srs.landauCutoffDens(bth,math.pi,relativistic=True,cutoff=0.3)

    diffs = np.abs(ne/srsUtils.nCritNIF-neCutoff)
    loc = x[np.where(diffs == np.min(diffs))[0]]

    ax.axvline(loc,color=color,linestyle='--',linewidth=linewidth)

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotLandauCutoffxWNsVsx(ax,x,Te,ne,wnNorm):
    tempXLims = ax.get_xlim()
    tempYLims = ax.get_ylim()

    ld = np.sqrt(const.k*Te*const.epsilon_0/(ne*const.e**2))
    cutoff = 0.3/ld/wnNorm

    ax.plot(x, cutoff,'w--')
    ax.plot(x,-cutoff,'w--')

    ax.set_xlim(tempXLims)
    ax.set_ylim(tempYLims)

def plotAngGrid(ax):
    xLims = ax.get_xlim()
    yLims = ax.get_ylim()

    xTicks = np.array(ax.get_xticks())
    rs = xTicks[np.where(xTicks > 0.0)]

    N = 12
    angles = 2.*math.pi*np.linspace(0,(N-1.)/N,N)
    RMax = 2.*max(max(xLims),max(yLims))
    RMin = 0.05*RMax

#	pts = []
    for theta in angles:
        xs = math.cos(theta)*np.array([RMin,RMax])
        ys = math.sin(theta)*np.array([RMin,RMax])

        ax.plot(xs,ys,color='w',linestyle=':',linewidth=0.75)
#		pts.append((xs,ys))

    for r in rs:
        c = plt.Circle((0.0,0.0),r,color='w',linestyle=':',linewidth=0.75,fill=False)
        ax.add_artist(c)


    ax.set_xlim(xLims)
    ax.set_ylim(yLims)

def processField(field,x,y,fieldName):
    # Cell centered grid
    # TODO: fix this for magnetic field components on cell edges (see diagram
    # in Arber PPCF PIC paper)
    xCC = 0.5*(x[1:] + x[:-1])/1e-6
    yCC = 0.5*(y[1:] + y[:-1])/1e-6

    if fieldName[-2] == 'B':
        norm = const.m_e*srsUtils.omegaNIF/const.e
    else:
        norm = const.m_e*const.c*srsUtils.omegaNIF/const.e

    field = field/norm
    return field,xCC,yCC

def plotField(fig,ax,field,x,y,t,fieldName,**plotArgs):
    xLabel = '$x$ $/\mu$m'
    yLabel = '$y$ $/\mu$m'

    fDir = fieldName[-1]
    if fieldName[-2] == 'B':
        zLabel = r'$B_{dirn}/\frac{{m_e\omega_0}}{{e}}$'
    else:
        zLabel = r'$E_{dirn}/\frac{{m_ec\omega_0}}{{e}}$'
    zLabel = zLabel.format(dirn=fieldName[-1])

    field,x,y = processField(data,x,y,fieldName)

    plotGridData(fig,ax,field,fieldName,x,y,t,True,'RdBu_r',xLabel,yLabel,zLabel,grid=True,**plotArgs)

def animField(fig,ax,sdfFiles,x,y,fieldName,**plotArgs):
    xLabel = '$x$ $/\mu$m'
    yLabel = '$y$ $/\mu$m'

    fDir = fieldName[-1]
    if fieldName[-2] == 'B':
        zLabel = r'$B_{dirn}/\frac{{m_e\omega_0}}{{e}}$'
    else:
        zLabel = r'$E_{dirn}/\frac{{m_ec\omega_0}}{{e}}$'
    zLabel = zLabel.format(dirn=fieldName[-1])

    procDataFunc = lambda f: processField(f,x,y,fieldName)

    anim = animGridData(fig,ax,sdfFiles,fieldName,x,y,procDataFunc,True,'RdBu_r',
                        xLabel,yLabel,zLabel,grid=True,**plotArgs)

    return anim

def processFieldEnv(data,x,y,fieldName):
    field,_,_ = processField(data,x,y,fieldName)
    fieldEnv = calcEnv(field)

    xCC = 0.5*(x[1:] + x[:-1])/1e-6
    yCC = 0.5*(y[1:] + y[:-1])/1e-6

    return fieldEnv,xCC,yCC

def plotFieldEnv(fig,ax,field,x,y,t,fieldName,**plotArgs):
    xLabel = '$x$ $/\mu$m'
    yLabel = '$y$ $/\mu$m'

    fDir = fieldName[-1]
    if fieldName[-2] == 'B':
        zLabel = r'$B_{dirn}/\frac{{m_e\omega_0}}{{e}}$'
    else:
        zLabel = r'$E_{dirn}/\frac{{m_ec\omega_0}}{{e}}$'
    zLabel = zLabel.format(dirn=fieldName[-1])

    fieldEnv,x,y = processFieldEnv(data,x,y,fieldName)

    plotGridData(fig,ax,fieldEnv,fieldName,x,y,t,False,'viridis',xLabel,yLabel,zLabel,
                 grid=True,**plotArgs)

def processFieldFT(data,x,y,fieldName,windowFunc):
    field,_,_ = processField(data,x,y,fieldName)

    xCC = 0.5*(x[1:] + x[:-1])
    yCC = 0.5*(y[1:] + y[:-1])

    fieldFT,xwns,ywns = calcFT(field,xCC,yCC,windowFunc)

    wnNorm = srsUtils.omegaNIF/const.c
    xwns = xwns/wnNorm
    ywns = ywns/wnNorm

    return fieldFT,xwns,ywns

def plotFieldFT(fig,ax,field,x,y,t,fieldName,Te=None,ne=None,windowFunc=np.hanning,**plotArgs):
    #xLabel = r'$k_x$ /$\frac{\omega_0}{c}$'
    #yLabel = r'$k_y$ /$\frac{\omega_0}{c}$'
    xLabel = r'$ck_x/\omega_0$'
    yLabel = r'$ck_y/\omega_0$'

    fDir = fieldName[-1]
    if fieldName[-2] == 'B':
        zLabel = r'$B_{dirn} /\frac{{m_e\omega_0}}{{e}}$'
    else:
        zLabel = r'$E_{dirn} /\frac{{m_ec\omega_0}}{{e}}$'
    zLabel = zLabel.format(dirn=fieldName[-1])

    fieldFT,xwns,ywns = processFieldFT(field,x,y,fieldName,windowFunc)

    plotGridData(fig,ax,fieldFT**2,fieldName,xwns,ywns,t,False,'inferno',xLabel,yLabel,zLabel,
                 grid=False,cbOrientation='vertical',xLabelPos='bottom',
                 tickColor='w',**plotArgs)

    if Te is not None and ne is not None:
        wnNorm = srsUtils.omegaNIF/const.c
        plotTPDWNs(ax,Te,ne,wnNorm)
        plotSRSWNs(ax,Te,ne,wnNorm)
        #plotAngGrid(ax)
        plotLandauCutoffWNs(ax,Te,ne,wnNorm)

def animFieldFT(fig,ax,sdfFiles,x,y,fieldName,Te=None,ne=None,windowFunc=np.hanning,**plotArgs):
    xLabel = r'$k_x$ /$\frac{\omega_0}{c}$'
    yLabel = r'$k_y$ /$\frac{\omega_0}{c}$'

    fDir = fieldName[-1]
    fType = None
    if fieldName[-2] == 'B':
        fType = 'EM'
        zLabel = r'$B_{dirn}/\frac{{m_e\omega_0}}{{e}}$'
    else:
        fType = 'ES'
        zLabel = r'$E_{dirn}/\frac{{m_ec\omega_0}}{{e}}$'
    zLabel = zLabel.format(dirn=fieldName[-1])

    procDataFunc = lambda data: processFieldFT(data,x,y,fieldName,windowFunc)

    # Plot Landau damping cutoff + TPD maximum growth wavenumbers
    if Te is not None and ne is not None:
        def overlayFunc(ax,xPlot,yPlot):
            wnNorm = srsUtils.omegaNIF/const.c
            plotTPDWNs(ax,Te,ne,wnNorm)
            plotSRSWNs(ax,Te,ne,wnNorm,EPW=(fType=='ES'),EM=(fType=='EM'))
            plotLandauCutoffWNs(ax,Te,ne,wnNorm)

    anim = animGridData(fig,ax,sdfFiles,fieldName,x,y,procDataFunc,False,
                        'inferno',xLabel,yLabel,zLabel,grid=False,
                        cbOrientation='vertical',xLabelPos='bottom',
                                            tickColor='w',overlayFunc=overlayFunc,**plotArgs)

    return anim

def processFieldFTy(data,x,y,fieldName,windowFunc):
    field,xPlot,_ = processField(data,x,y,fieldName)
    yCC = 0.5*(y[1:] + y[:-1])
    fieldFTy,ywns = calcFTy(field,yCC,windowFunc)

    wnNorm = srsUtils.omegaNIF/const.c
    ywnsPlot = ywns/wnNorm

    return fieldFTy,xPlot,ywnsPlot

def plotFieldFTy(fig,ax,field,x,y,t,fieldName,Te=None,ne=None,windowFunc=np.hanning,**plotArgs):
    xLabel = r'$x$ /$\mu$m'
    #yLabel = r'$k_y$ /$\frac{\omega_0}{c}$'
    yLabel = r'$ck_y/\omega_0$'

    fDir = fieldName[-1]
    if fieldName[-2] == 'B':
        zLabel = r'$B_{dirn} /\frac{{m_e\omega_0}}{{e}}$'
    else:
        zLabel = r'$E_{dirn} /\frac{{m_ec\omega_0}}{{e}}$'
    zLabel = zLabel.format(dirn=fieldName[-1])

    fieldFTy,x,ywns = processFieldFTy(field,x,y,fieldName,windowFunc)

    plotGridData(fig,ax,fieldFTy**2,fieldName,x,ywns,t,False,'inferno',xLabel,yLabel,zLabel,
                 aspect='auto',grid=False,cbOrientation='vertical',
                 xLabelPos='bottom',tickColor='w',**plotArgs)

    if Te is not None and ne is not None:
        ax2 = ax.twiny()
        wnNorm = srsUtils.omegaNIF/const.c
        plotTPDyWNsVsx(ax,x,Te,ne,wnNorm,linewidth=1.0)
        plotSSRSyWNsVsx(ax,x,Te,ne,wnNorm,linewidth=1.0)
        #plotLandauCutoffyWNsVsx(ax,x,Te,ne,wnNorm)
        plotTPDLandauCutoffx(ax,x,Te,ne,wnNorm,linewidth=1.0)
        plotSRSLandauCutoffx(ax,x,Te,ne,wnNorm,linewidth=1.0,color='g')

        # TODO: make minVal and interval as command line arguments
        #srsUtils.misc.addNeScale(ax,ax2,x,ne,minVal=0.10,interval=0.05,minor=True)

def animFieldFTy(fig,ax,sdfFiles,x,y,fieldName,Te=None,ne=None,windowFunc=np.hanning,**plotArgs):
    xLabel = r'$x$ /$\mu$m'
    yLabel = r'$k_y$ /$\frac{\omega_0}{c}$'

    fDir = fieldName[-1]
    if fieldName[-2] == 'B':
        zLabel = r'$B_{dirn} /\frac{{m_e\omega_0}}{{e}}$'
    else:
        zLabel = r'$E_{dirn} /\frac{{m_ec\omega_0}}{{e}}$'
    zLabel = zLabel.format(dirn=fieldName[-1])

    procDataFunc = lambda data: processFieldFTy(data,x,y,fieldName,windowFunc)

    # Plot Landau damping cutoff + TPD maximum growth wavenumbers
    if Te is not None and ne is not None:
        xCC = 0.5*(x[1:] + x[:-1])
        def overlayFunc(ax,xPlot,yPlot):
            wnNorm = srsUtils.omegaNIF/const.c
            plotTPDyWNsVsx(ax,xPlot,Te,ne,wnNorm)
            plotLandauCutoffyWNsVsx(ax,xPlot,Te,ne,wnNorm)

    anim = animGridData(fig,ax,sdfFiles,fieldName,x,y,procDataFunc,False,
                        'inferno',xLabel,yLabel,zLabel,aspect='auto',grid=False,
                        cbOrientation='vertical',xLabelPos='bottom',
                                            tickColor='w',overlayFunc=overlayFunc,**plotArgs)

    return anim

def processFieldWltx(data,x,y,fieldName,wltFreqMult=None,ySkip=None,kSkip=None,
                     kMax=None):
    field,xPlot,_ = processField(data,x,y,fieldName)
    xCC = 0.5*(x[1:] + x[:-1])

    fieldWltx,_,xwns = calcWltx(field,xCC,wltFreqMult=wltFreqMult,ySkip=ySkip,
                                kSkip=kSkip,kMax=kMax)

    wnNorm = srsUtils.omegaNIF/const.c
    xwnsPlot = xwns/wnNorm

    return fieldWltx,xPlot,xwnsPlot

def plotFieldWltx(fig,ax,field,x,y,t,fieldName,Te=None,ne=None,
                      wltFreqMult=None,ySkip=None,kSkip=None,kMax=None,**plotArgs):
    xLabel = r'$x$ /$\mu$m'
    yLabel = r'$k_x$ /$\frac{\omega_0}{c}$'

    fDir = fieldName[-1]
    if fieldName[-2] == 'B':
        zLabel = r'$B_{dirn} /\frac{{m_e\omega_0}}{{e}}$'
    else:
        zLabel = r'$E_{dirn} /\frac{{m_ec\omega_0}}{{e}}$'
    zLabel = zLabel.format(dirn=fieldName[-1])

    fieldWltx,x,xwns = processFieldWltx(field,x,y,fieldName,
                                        wltFreqMult=wltFreqMult,ySkip=ySkip,
                                        kSkip=kSkip,kMax=kMax)

    plotGridData(fig,ax,fieldWltx,fieldName,x,xwns,t,False,'inferno',xLabel,yLabel,zLabel,
                 aspect='auto',grid=False,cbOrientation='vertical',
                 xLabelPos='bottom',tickColor='w',**plotArgs)

    if Te is not None and ne is not None:
        wnNorm = srsUtils.omegaNIF/const.c
        plotTPDxWNsVsx(ax,x,Te,ne,wnNorm)
        plotSRSWNsVsx(ax,x,Te,ne,wnNorm)
        plotLandauCutoffxWNsVsx(ax,x,Te,ne,wnNorm)

def animFieldWltx(fig,ax,sdfFiles,x,y,fieldName,Te=None,ne=None,
                      wltFreqMult=None,ySkip=None,kSkip=None,kMax=None,**plotArgs):
    xLabel = r'$x$ /$\mu$m'
    yLabel = r'$k_x$ /$\frac{\omega_0}{c}$'

    fDir = fieldName[-1]
    if fieldName[-2] == 'B':
        zLabel = r'$B_{dirn} /\frac{{m_e\omega_0}}{{e}}$'
    else:
        zLabel = r'$E_{dirn} /\frac{{m_ec\omega_0}}{{e}}$'
    zLabel = zLabel.format(dirn=fieldName[-1])

    procDataFunc = lambda data: processFieldWltx(data,x,y,fieldName,
                                                 wltFreqMult,ySkip,kSkip,kMax)

    # Plot Landau damping cutoff + TPD maximum growth wavenumbers
    if Te is not None and ne is not None:
        xCC = 0.5*(x[1:] + x[:-1])
        def overlayFunc(ax,xPlot,yPlot):
            wnNorm = srsUtils.omegaNIF/const.c
            plotTPDxWNsVsx(ax,xPlot,Te,ne,wnNorm)
            plotSRSWNsVsx(ax,xPlot,Te,ne,wnNorm)
            plotLandauCutoffxWNsVsx(ax,xPlot,Te,ne,wnNorm)

    anim = animGridData(fig,ax,sdfFiles,fieldName,x,y,procDataFunc,False,
                        'inferno',xLabel,yLabel,zLabel,aspect='auto',grid=False,
                        cbOrientation='vertical',xLabelPos='bottom',
                                            tickColor='w',overlayFunc=overlayFunc,**plotArgs)

    return anim

def expDensity_alex(x,n0,Ln):
    return n0*np.exp(x/Ln)

def initial_density_profile(n0,x,Lnc):
    return n0*np.exp(x*1e6/Lnc)


def processDensity(density,x,y,densityName,lims0,lims1,subtract=False,frac=1.0,ne=None):
    # Cell centered grid (on which density is based)
    xCC = 0.5*(x[1:] + x[:-1])
    yCC = 0.5*(y[1:] + y[:-1])
    Ny = yCC.shape[0]
    # print('subtract',subtract)
    # if subtract:
    #     print(frac)
    #     expDensity = frac*np.outer(ne,np.ones(Ny))
    #     # if lims0==None and lims1 == None:
            
    #     expDensity = ne[0:len(density)]
    #     # else:
    #     #     expDensity = ne[lims0:lims1]
            
    #     print('len(density)',len(density),'len(expDensity)',len(expDensity))

    #     print('μ before: {:}'.format(np.mean(density)))
    #     density = (density/expDensity - 1.0)
    #     print('μ after: {:}'.format(np.mean(density)))
    # else:
    #     density = density/srsUtils.nCritNIF
    #     print('still else')

    return density,xCC/1e-6,yCC/1e-6

def plotDensity(fig,ax,data,x,y,t,densityName,subtract=False,frac=None,ne=None,**plotArgs):
    xLabel = r'$x$ $/\mu$m'
    yLabel = r'$y$ $/\mu$m'
    zLabel = r'$\delta n/n_0$'

    print('plot density')
    # density,xCC,yCC = processDensity(data,x,y,densityName,index_xlims0,index_xlims1,subtract,frac,ne)

    xCC = x
    yCC = y
    
    density = data
    
    if subtract:
        cmap = 'RdBu_r'
    else:
        cmap = 'viridis'

    symmetric = subtract
    
    print(data.shape)

    plotGridData(fig,ax,density,densityName,xCC,yCC,t,symmetric,cmap,xLabel,yLabel,zLabel,**plotArgs)

def animDensity(fig,ax,sdfFiles,x,y,densityName,subtract=False,frac=None,ne=None,**plotArgs):
    xLabel = r'$x$ $/\mu$m'
    yLabel = r'$y$ $/\mu$m'
    zLabel = r'$\delta n/n_0$'

    procDataFunc = lambda d: processDensity(d,x,y,densityName,subtract,frac,ne,index_xlims0,index_xlims1)

    if subtract:
        cmap = 'RdBu_r'
    else:
        cmap = 'viridis'

    anim = animGridData(fig,ax,sdfFiles,densityName,x,y,procDataFunc,True,cmap,
                        xLabel,yLabel,zLabel,grid=True,**plotArgs)

    return anim

def processDensityFT(density,x,y,densityName,windowFunc,subtract=False,frac=1.0,ne=None):
    density,_,_ = processDensity(density,x,y,densityName,index_xlims0,index_xlims1,subtract=subtract,
                                 frac=frac,ne=ne)
    xCC = 0.5*(x[1:] + x[:-1])
    yCC = 0.5*(y[1:] + y[:-1])
    print('enter process density')
    densityFT,xwns,ywns = calcFT(density,xCC,yCC,windowFunc)

    wnNorm = srsUtils.omegaNIF/const.c
    xwns = xwns/wnNorm
    ywns = ywns/wnNorm

    return densityFT,xwns,ywns

def plotDensityFT(fig,ax,density,x,y,t,densityName,subtract=False,frac=None,ne=None,Te=None,windowFunc=np.hanning,**plotArgs):
    xLabel = r'$ck_x/\omega_0$'
    yLabel = r'$ck_y/\omega_0$'
    zLabel = r'$\delta n/n_0$'
    
#    print('lenghts',lex(x),len(y),len(density))

    densityFT,xwns,ywns = processDensityFT(density,x,y,densityName,windowFunc,subtract=subtract,frac=frac,ne=ne)

    plotGridData(fig,ax,densityFT,densityName,xwns,ywns,t,False,'inferno',xLabel,yLabel,zLabel,
                 grid=False,cbOrientation='vertical',xLabelPos='bottom',
                 tickColor='w',**plotArgs)

    if Te is not None and ne is not None:
        wnNorm = srsUtils.omegaNIF/const.c
        if densityName.endswith('electrons'):
            fac = 1.0
        else:
            fac = 0.5
        #plotTPDWNs(ax,Te,ne,fac*wnNorm)
        #plotLandauCutoffWNs(ax,Te,ne,fac*wnNorm)

def animDensityFT(fig,ax,sdfFiles,x,y,densityName,subtract=False,frac=None,ne=None,Te=None,windowFunc=np.hanning,**plotArgs):
    xLabel = r'$k_x$ /$\frac{\omega_0}{c}$'
    yLabel = r'$k_y$ /$\frac{\omega_0}{c}$'
    zLabel = r'$\delta n/n_0$'

    procDataFunc = lambda data: processDensityFT(data,x,y,densityName,windowFunc,subtract=subtract,frac=frac,ne=ne)

    # Plot Landau damping cutoff + TPD maximum growth wavenumbers
    if Te is not None and ne is not None:
        xCC = 0.5*(x[1:] + x[:-1])
        def overlayFunc(ax,xPlot,yPlot):
            wnNorm = srsUtils.omegaNIF/const.c
            if densityName.endswith('electrons'):
                fac = 1.0
            else:
                fac = 0.5
            plotTPDWNs(ax,Te,ne,fac*wnNorm)
            plotLandauCutoffWNs(ax,Te,ne,fac*wnNorm)

    anim = animGridData(fig,ax,sdfFiles,densityName,x,y,procDataFunc,False,
                        'inferno',xLabel,yLabel,zLabel,grid=False,
                        cbOrientation='vertical',xLabelPos='bottom',
                                            tickColor='w',overlayFunc=overlayFunc,**plotArgs)

    return anim

def processDensityFTy(density,x,y,densityName,windowFunc,subtract=True,frac=None,ne=None):
    density,x,_ = processDensity(density,x,y,densityName,index_xlims0,index_xlims1,subtract=subtract,
                                 frac=frac,ne=ne)
        
    yCC = 0.5*(y[1:] + y[:-1])
    densityFTy,ywns = calcFTy(density,yCC,windowFunc)

    wnNorm = srsUtils.omegaNIF/const.c
    ywns = ywns/wnNorm

    return densityFTy,x,ywns

def plotDensityFTy(fig,ax,density,x,y,t,densityName,subtract=True,frac=None,ne=None,Te=None,windowFunc=np.hanning,**plotArgs):
    xLabel = r'$x$ /$\mu$m'
    yLabel = r'$k_y$ /$\frac{\omega_0}{c}$'
    zLabel = r'$\delta n/n_0$'

    densityFTy,x,ywns = processDensityFTy(density,x,y,densityName,windowFunc,subtract=subtract,frac=frac,ne=ne)

    plotGridData(fig,ax,densityFTy,densityName,x,ywns,t,False,'inferno',xLabel,yLabel,zLabel,
                 aspect='auto',grid=False,cbOrientation='vertical',
                 xLabelPos='bottom',**plotArgs)

    if Te is not None and ne is not None:
        #xCC = 0.5*(x[1:] + x[:-1])
        wnNorm = srsUtils.omegaNIF/const.c
        if densityName.endswith('electrons'):
            fac = 1.0
        else:
            fac = 0.5
        plotTPDyWNsVsx(ax,x,Te,ne,fac*wnNorm)
        plotLandauCutoffyWNsVsx(ax,x,Te,ne,fac*wnNorm)

    ax.tick_params(reset=True,direction='in',color='w')

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def animDensityFTy(fig,ax,sdfFiles,x,y,densityName,subtract=True,frac=None,ne=None,Te=None,windowFunc=np.hanning,**plotArgs):
    xLabel = r'$x$ /$\mu$m'
    yLabel = r'$k_y$ /$\frac{\omega_0}{c}$'
    zLabel = r'$\delta n/n_0$'

    procDataFunc = lambda data: processDensityFTy(data,x,y,densityName,windowFunc,
                                                  subtract=subtract,frac=frac,
                                                  ne=ne)

    # Plot Landau damping cutoff + TPD maximum growth wavenumbers
    if Te is not None and ne is not None:
        xCC = 0.5*(x[1:] + x[:-1])
        def overlayFunc(ax,xPlot,yPlot):
            wnNorm = srsUtils.omegaNIF/const.c
            if densityName.endswith('electrons'):
                fac = 1.0
            else:
                fac = 0.5
            plotTPDyWNsVsx(ax,xPlot,Te,ne,wnNorm)
            plotLandauCutoffyWNsVsx(ax,xPlot,Te,ne,wnNorm)

    anim = animGridData(fig,ax,sdfFiles,densityName,x,y,procDataFunc,False,
                        'inferno',xLabel,yLabel,zLabel,aspect='auto',grid=False,
                        cbOrientation='vertical',xLabelPos='bottom',
                                            tickColor='w',overlayFunc=overlayFunc,**plotArgs)

    return anim

def plotDensityMeanDiffY(fig,ax,density,x,y,t,densityName,subtract=False,
                         frac=None,ne=None,xLims=None,yLims=None,**plotArgs):
    xLabel = r'$x$ $/\mu$m'
    yLabel = r'$y$ $/\mu$m'
    zLabel = r'$\delta n/n_0$'

    density,xCC,yCC = processDensity(density,x,y,densityName,index_xlims0,index_xlims1,subtract,frac,ne)

    meanY = np.mean(density,axis=1)
    ax.plot(xCC,meanY)
    ax.grid()

    maxAmp = np.max(np.abs(ax.get_ylim()))

    ax.set_xlim(x[0]/1e-6,x[-1]/1e-6)
#	if subtract:
#		ax.set_ylim(-maxAmp,maxAmp)
#	else:
#		ax.set_ylim(0.0,maxAmp)

    if xLims:
        ax.set_xlim(xLims)
    if yLims:
        ax.set_ylim(yLims)
    #ax.set_yscale('log')

    ax.set_xlabel(r'$x$ /$\mu$m',**axis_font)
    if subtract:
        ax.set_ylabel(r'$\langle\frac{\delta n}{n_0}\rangle_y$',rotation=0,**axis_font)
    else:
        ax.set_ylabel(r'$\langle n/n_{\mathrm{cr}}\rangle_y$',rotation=0,**axis_font)

    return ax

def getMaxAmp(sdfFiles,var):
    maxVals = []
    for f in sdfFiles:
        maxVal = np.max(sdf.read(f).__dict__[var].data)
        maxVals.append(maxVal)
        print("done")

    return max(maxVals)

def getMaxAmpFT(sdfFiles,var):
    maxVals = []
    for f in sdfFiles:
        maxVal = np.max(sdf.read(f).__dict__[var].data)
        maxVals.append(maxVal)
        print("done")

    return max(maxVals)



if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('fName')
    parser.add_argument('dataName')
    parser.add_argument('dataDir')

    parser.add_argument('--prefix',default='regular')
    parser.add_argument('--dataDir',default='./')

    # Arguments to decide what kind of plot we're after
    parser.add_argument('--plotType',
                        choices=['normal','env','FT','FTy','wltx','meanDiffY'],
                                            default='normal')
    parser.add_argument('-a','--animate',action='store_true')
    parser.add_argument('--subtractDensity',action='store_true')
    parser.add_argument('-l','--log',action='store_true')

    # Wavelet transform options
    parser.add_argument('--wltx_kMax',type=float)

    # Options controlling detailed choice of data
    parser.add_argument('-n','--fileNum',type=int,default=0)
    parser.add_argument('--nCrop',type=int,default=6)
    parser.add_argument('--nCropx',type=int,nargs=2)
    parser.add_argument('--nCropy',type=int,nargs=2)
    parser.add_argument('--cropx',type=float,nargs=2)
    parser.add_argument('--cropy',type=float,nargs=2)
    parser.add_argument('--minN',type=int,default=0)
    parser.add_argument('--maxN',type=int,default=-1)
    parser.add_argument('--spatialX',type=float,nargs=2,default =[0,1950])
    parser.add_argument('--spatialY',type=float,nargs=2,default =[-9,9])

    # # Options controlling plot limits
    parser.add_argument('--minX',type=float)
    parser.add_argument('--maxX',type=float)
    parser.add_argument('--minY',type=float)
    parser.add_argument('--maxY',type=float)
    parser.add_argument('--minF',type=float)
    parser.add_argument('--maxF',type=float)

    # # Options for specifying physical parameters needed for overlaying of e.g.
    # # dispersion relations.
    parser.add_argument('--densityRange',type=float,nargs=2) # 0.194580059689 0.260175683371
    parser.add_argument('--temperature',type=float)
    parser.add_argument('--densityFrac',type=float,default=1.0) # H = 0.183922829582, C = 0.136012861736
    parser.add_argument('--densityProfile',choices=['exp','calc'])

    # # Animation options
    parser.add_argument('--cache',action='store_true')
    parser.add_argument('--useExistingCache',action='store_true')
    parser.add_argument('--cacheDir')

    # # Options for controlling miscellaneous aspects of the plot
    parser.add_argument('--figSize',type=float,nargs=2)
    parser.add_argument('--fontSize',type=float)
    parser.add_argument('--noTitle',action='store_true')
    parser.add_argument('--noCBar',action='store_true')
    parser.add_argument('-o','--output',required=True)
    
    parser.add_argument('--IntervalInitial',type=int, default = 1)
    parser.add_argument('--IntervalFinal',type=int, default = 40)
    # parser.add_argument('--saving_also_snapshot',type = bool,default = False)
    

    parser.add_argument('--which_run',type = str)
    parser.add_argument('--saving_to_mine',type = bool,default = False)
    

    # ### Adding external lineouts for density
    # parser.add_argument('--densityMarkLineouts',type = bool,default = False)
    parser.add_argument('--densLim',type=float,nargs=2, default = None)
    parser.add_argument('--Lnc4',type=float, default = None)
    parser.add_argument('--initialDensitySim',type=float,default = .1)
    
    parser.add_argument('--averaged_y_density_snapshots',type = bool,default = False)
    parser.add_argument('--colorplots',type = bool,default = False)
    
    

    args = parser.parse_args()
    
    if args.spatialX:
        xSpatialLimits = args.spatialX
        print(xSpatialLimits)
        
    if args.spatialY:
        ySpatialLimits = args.spatialY
        print(ySpatialLimits)
#    parser.add_argument('--spatialY',type=float,nargs=2)


    # Find data
    outDir = os.path.join(args.fName,_outputDir)

    if args.animate:
        # Read data
        files = sdfUtils.listFiles(args.fName,args.prefix)[args.minN:args.maxN]
        grid = sdf.read(files[0]).Grid_Grid.data

        print('Animating data from {:} snapshots'.format(len(files)))
        

# =============================================================================
#             loop
# =============================================================================

# Find data


    files = sdfUtils.listFiles(args.dataDir,args.prefix)[args.IntervalInitial:args.IntervalFinal]
    print(files)

   
    time_integrated_data = 0
    
    lastFile = args.IntervalFinal - args.IntervalInitial

    for i in range(0,lastFile-1):
              
        
        

           
        time_data =  sdf.read(files[i])
        time_snapshot = np.array(time_data.Header['time'] )/1e-12
        if i == 0:
            t0 = time_snapshot = np.array(time_data.Header['time'] )/1e-12
        elif i == lastFile-2:
            tf = time_snapshot = np.array(time_data.Header['time'] )/1e-12
            print(tf,tf)

        print(time_snapshot,'time_snapshot')
        
        sdfFile = files[i]
        print(files[i]) 

        grid = sdf.read(sdfFile).Grid_Grid.data
        # grid = sdf.read(data).Grid_Grid.data
        if not args.animate:
            t,data = getData(sdfFile,args.dataName)

        
 

        xOrig,yOrig = grid
        
        
        xOrig = xOrig[6:len(xOrig)-7]
        yOrig = yOrig[6:len(yOrig)-7]
        
        
        
    

    
# =============================================================================
#       NOTE: NOT IMPLEMENTED SPATIAL Y
# =============================================================================
        # Process various arguments controlling the final output
        xLims=(args.minX,args.maxX)
        yLims=(args.minY,args.maxY)
        zLims=(args.minF,args.maxF)
    
        if args.fontSize:
            import matplotlib as mpl
            mpl.rcParams.update({'font.size':args.fontSize})
    
    
        if args.wltx_kMax is not None:
            wltx_kMax = args.wltx_kMax*srsUtils.wnVacNIF
        else:
            wltx_kMax = None
    
         
        # Process arguments informing us about the expected simulation plasma
        if args.densityRange is not None:
            args.densityRange = np.array(args.densityRange)*srsUtils.nCritNIF
            args.densityProfile = 'exp'
    
        if args.densityProfile == 'exp':
            
            if i ==0:
       
                
                
                Ln = 1e6*(xOrig[-1]-xOrig[0])/math.log(args.densityRange[1]/args.densityRange[0])
    

            
                print('PLEASE, CHECK IF YOUR args.initialDensitySim IS CONSITENT WITH YOUR LOWEST DENSITY IN SIM')
                print('Density scale lenght at qc',Ln)
                ne = initial_density_profile(args.initialDensitySim,xOrig,Ln)
        
        
   
        # elif args.densityProfile == 'calc':
        #     if args.animate:
        #         initSnapFName = os.path.join(args.fName,'regular_0000.sdf')
        #     else:
        #         initSnapFName = os.path.join(os.path.dirname(args.fName),'regular_0000.sdf')
        #     ne = sdf.read(initSnapFName).Derived_Number_Density_Electron.data
        #     ne = cropArray(ne,_nCropx,_nCropy)
        #     ne = np.mean(ne,axis=1)
        #     ne = gaussian_filter1d(ne,sigma=10)
    
        if args.temperature:
            args.temperature = args.temperature*srsUtils.TkeV
    

        # These are keyword arguments that get passed to the function that
        # ultimately plots everything (either plotGridData or animGridData)
        plotArgs = {'xLims':xLims,'yLims':yLims,'zLims':zLims,'log':args.log,
                    'noTitle':args.noTitle,'noCBar':args.noCBar}
        
        
        if args.spatialX:
            
            
            xlims0,index_xlims0 = find_nearest(xOrig*1e6,xSpatialLimits[0])
            xlims1,index_xlims1 = find_nearest(xOrig*1e6,xSpatialLimits[1]) 
            x = xOrig[index_xlims0:index_xlims1]
            data_2d = data[index_xlims0:index_xlims1,:]
            n_time = np.mean(data[index_xlims0:index_xlims1,:],axis=1)
            if i ==0:
                ne = ne[index_xlims0:index_xlims1]


            
            print('new x lims', x.min()*1e6,x.max()*1e6)
            


        else:
            xlims0 = xOrig.min()*1e6
            xlims1 = xOrig.max()*1e6
            data_2d = data[0:len(xOrig),:]
            n_time = np.mean(data,axis=1)


            x = xOrig
            
    
        if args.spatialY:
            
            ylims0,index_ylims0 = find_nearest(yOrig*1e6,ySpatialLimits[0])
            ylims1,index_ylims1 = find_nearest(yOrig*1e6,ySpatialLimits[1]) 
            y = yOrig[index_ylims0:index_ylims1]
            data_2d = data_2d[:,index_ylims0:index_ylims1]

        else:
            ylims0 = yOrig.min()*1e6
            ylims1 = xOrig.max()*1e6
            data_2d = data[:,0:len(yOrig)]

            y = yOrig
            
        
# =============================================================================
#         y-averaged density vs x plots
# =============================================================================
        
        if args.averaged_y_density_snapshots == True:
        
            
            

            if args.dataName.endswith('Electron'):              
                delta_density = n_time/srsUtils.nCritNIF/ne-1.0
            if args.dataName.endswith('carbon'):    
                delta_density = n_time/(ne*srsUtils.nCritNIF/12)-1.715
            if args.dataName.endswith('carbon'):    
                delta_density = n_time/(ne*srsUtils.nCritNIF/1)-1


            
            plt.title('t ={:.3f}ps'.format(t/1e-12),**axis_font)
            plt.plot(x*1e6,delta_density)
            plt.xlabel('x ($\mu$m)',**axis_font)
            plt.tick_params(labelsize=15,length=10, width=1,direction='inout',labelcolor='black')
            # plt.legend(loc = 'best',prop={'size':12})
            plt.grid(linestyle='--')
            
            if args.dataName.endswith('Electron'):              
            
                plt.ylabel('$\Delta ne$  ',**axis_font) 
    
                plt.savefig('./pics/ne_y_aver/ne'+str(i)+'.jpg')
                
            if args.dataName.endswith('carbon'):              
            
                plt.ylabel('$\Delta nc$  ',**axis_font) 
    
                plt.savefig('./pics/nc_y_aver/nc'+str(i)+'.jpg')
                
            if args.dataName.endswith('proton'):              
            
                plt.ylabel('$\Delta nh$  ',**axis_font) 
    
                plt.savefig('./pics/nh_y_aver/nh'+str(i)+'.jpg')
                
                
                
                
            plt.close()
            
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if args.figSize:
            fig.set_size_inches(*args.figSize)
        elif args.plotType == 'normal':
            fig.set_size_inches(10,3)
        elif args.plotType == 'FT':
            fig.set_size_inches(7.5,5)
        elif args.plotType == 'FTy':
            fig.set_size_inches(10,3)
        else:
            fig.set_size_inches(10,3)


# =============================================================================
#         x-y density snapshots
# =============================================================================

        ne_2d_reduced = np.outer(ne[index_xlims0:index_xlims1],np.ones(len(y)))

        if args.colorplots == True:
            if args.subtractDensity:

                if args.dataName.endswith('Electron'):              
                    data_2d = (data_2d/srsUtils.nCritNIF)/ne_2d_reduced-1.0
                if args.dataName.endswith('carbon'):    
                    # data_2d = data_2d/(ne_2d_reduced*srsUtils.nCritNIF/12)-1.715
                    data_2d = (data_2d/srsUtils.nCritNIF)/ne_2d_reduced-1.0
                if args.dataName.endswith('carbon'):    
                    data_2d = (data_2d/srsUtils.nCritNIF)/ne_2d_reduced-1.0

                
                
            if args.plotType == 'normal':
                im=plotDensity(fig,ax,data_2d,x,y,t,args.dataName,grid=True,**plotArgs)

            elif args.plotType == 'FT':
                im=plotDensityFT(fig,ax,data_2d,x,y,t,args.dataName,Te=args.temperature,windowFunc=np.hanning,**plotArgs)
            elif args.plotType == 'FTy':
                print(i)
                if i> 1:
                    print('qua')
                    im=plotDensityFTy(fig,ax,data_2d,x,y,t,args.dataName,Te=args.temperature,windowFunc=np.hanning,**plotArgs)
                else:
                    im=plotDensityFTy(fig,ax,data_2d,x,y,t,args.dataName,Te=args.temperature,windowFunc=np.hanning,**plotArgs)
            elif args.plotType == 'meanDiffY':
                im=plotDensityMeanDiffY(fig,ax,data_2d,x,y,t,args.dataName,grid=True,**plotArgs)
    
                
       
            fig.tight_layout(pad=0,w_pad=0,h_pad=0)
            print(args.plotType)
        #fig.savefig(args.output+'_'+str(int(time_snapshot*1e3))+'_fs_x_range_'+str(args.spatialX[0])+'_'+str(args.spatialX[1])+'_um.jpg',dpi=600,pad='tight')
#                fig.savefig(+args.output+'_.jpg',dpi=600,pad='tight')
            if args.saving_to_mine == True:
                plt.xlabel('')
                plt.ylabel('')
                plt.xticks([])
                plt.yticks([])
                 
                fig.savefig('/work/e689/e689/ruocco89/pioneer_project/2D/runs/'+args.which_run+'/pics/'+args.output+args.plotType+'/'+args.output+'_'+str(int(time_snapshot*1e3))+'_fs_x_'+str(args.spatialX[0])+'_'+str(args.spatialX[1])+'_um.jpg',dpi=1200,pad='tight')
                plt.close()
            else:
                fig.savefig('./pics/'+args.output+args.plotType+'/'+args.output+'_'+str(int(time_snapshot*1e3))+'_fs_x_'+str(args.spatialX[0])+'_'+str(args.spatialX[1])+'_um.jpg',dpi=600,pad='tight')
                plt.close()

    
