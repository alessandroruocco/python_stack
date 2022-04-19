#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
from scipy.ndimage.filters import gaussian_filter1d
import sdf
import time
import os

import srsUtils
import sdfUtils

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('figure', autolayout=True)

_maxMemoryGiB = 16.0

def poyntFluxVsTime(ax,files,nStrips,xLims=None,log=False,subset='strip_ss'):
	data = [ sdf.read(f) for f in files ]

	NtAcc = data[0].Grid_A_strip_ss.data[2].shape[0]
	x = data[0].Grid_A_strip_ss.data[0]
	t = np.array([])
	for d in data:
		t = np.concatenate((t,np.array(d.Grid_A_strip_ss.data[-1])))
	print(t.shape)
	print(t[-1]/1e-12)

	smoothLen = 2*2*np.pi*const.c/omega0
	smoothTime = 2*2.0*math.pi/omega0
	dx = x[1]-x[0]
	dt = t[1]-t[0]
	xSigma = smoothLen/dx
	tSigma = smoothTime/dt

	xSkip = int(math.floor(xSigma/4))
	tSkip = int(math.floor(tSigma/2))

	xSmoothedArray = np.zeros((int(math.ceil(float(x.shape[0])/xSkip)),t.shape[0]))
	print(xSmoothedArray.shape)

	for i,d in enumerate(data):
		#for j in
		print(i)
		#Ex = d.__dict__['Electric_Field_Ex_Acc_'+subset].data[:,0,:]
		Ey = d.__dict__['Electric_Field_Ey_Acc_'+subset].data[:,0,:]
		#Ez = d.__dict__['Electric_Field_Ez_Acc_'+subset].data[:,0,:]
		#Bx = d.__dict__['Magnetic_Field_Bx_Acc_'+subset].data[:,0,:]
		#By = d.__dict__['Magnetic_Field_By_Acc_'+subset].data[:,0,:]
		Bz = d.__dict__['Magnetic_Field_Bz_Acc_'+subset].data[:,0,:]

		#poynt = (Ey*Bz - Ez*By)/const.mu_0 # x component
		#poynt = (Ez*Bx - Ex*Bz)/const.mu_0 # y component
		poynt = Ey*Bz/const.mu_0

		poynt = gaussian_filter1d(poynt,sigma=smoothLen/dx,axis=0)[::int(math.floor(xSigma/4))]

		xSmoothedArray[:,i*NtAcc:(i+1)*NtAcc] = poynt

	tSmoothedArray = gaussian_filter1d(xSmoothedArray,sigma=smoothTime/dt,axis=1)[:,::int(math.floor(tSigma/2))]

	tSmoothedArray = (tSmoothedArray/1e19).transpose()

def poyntFluxVsSpaceTime(files,omega0,filtOut='scatter',xDims=None,yDims=None,
                         xLims=None,yLims=None,maxF=None,dims=2,
                         subset='strip_ss',direction='x'):
	'''
	Calculate Poynting flux as a function of space and time along a strip of cells

	Description
	===========

	Coming soon...

	Arguments
	=========

	files   : datafiles containing accumulated strip data
	omega0	: Initial laser angular frequency
	filtOut : String indicating what should be filtered out, either 'laser'
	          or 'scatter'
	xDims   : Number of samples along spatial dimension to take & plot. If None
	          then use same number as original dataset, which may result in a
	          very large image! This will also take a long time to process...
	xLims   : Limits in spatial direction to process, in μm
	yLims   : Limits in temporal direction to process, in ps
	maxF    : maximum value for colour scale, in W/cm^2
	dims    : Number of dimensions of simulation (this affects the accumulator
	          array shape)
	subset  : field subset name as specified in input deck
	direction: String parameter, either 'x' or 'y'. Specifies component of the
	          Poynting vector to use. Normally this would be 'x'.
	'''

	exKey = 'Electric_Field_Ex_Acc_'+subset
	eyKey = 'Electric_Field_Ey_Acc_'+subset
	ezKey = 'Electric_Field_Ez_Acc_'+subset
	bxKey = 'Magnetic_Field_Bx_Acc_'+subset
	byKey = 'Magnetic_Field_By_Acc_'+subset
	bzKey = 'Magnetic_Field_Bz_Acc_'+subset

	key1 = bzKey
	if direction == 'x':
		key2 = eyKey
	else:
		key2 = exKey

	# Main filter properties, should probably be arguments
	bandwidth=0.3
	filtLen = 1001

	data = [ sdf.read(f) for f in files ]

	NtAcc = data[0].__dict__['Grid_A_'+subset].data[-1].shape[0]
	x = data[0].__dict__['Grid_A_'+subset].data[0]
	x = 0.5*(x[1:] + x[:-1])

	Nx = len(x)
	if xDims is not None:
		xSkip = max([ 1, Nx//xDims ])
	else:
		xSkip = 1

	# Choose range of x to filter based on xLims
	xMinInd = 0
	xMaxInd = len(x)
	if xLims is not None:
		ltInds = np.where(x < xLims[0]*1e-6)[0]
		if len(ltInds) != 0:
			xMinInd = ltInds[-1]

		gtInds = np.where(x > xLims[1]*1e-6)[0]
		if len(gtInds) != 0:
			xMaxInd = gtInds[0]+1

	# Select range from x grid array
	x = x[xMinInd:xMaxInd:xSkip]

	# Slices to select spatial region of interest in accumulated data
	if dims == 1:
		slc = np.s_[xMinInd:xMaxInd:xSkip]
		slcNoSkip = np.s_[xMinInd:xMaxInd]
	elif dims == 2:
		slc = np.s_[xMinInd:xMaxInd:xSkip,0]
		slcNoSkip = np.s_[xMinInd:xMaxInd,0]

	# Construct array of times at which snapshots were taken
	t = np.array([])
	accMinInd = 0
	accMaxInd = len(data)
	for i,d in enumerate(data):
		newT = np.array(d.__dict__['Grid_A_'+subset].data[-1])
		dt = newT[1]-newT[0]

		# Check for missing data, fix by padding with zeros and repairing time grid
		if newT.shape[0] != NtAcc:
			print("Missing data in file {:}".format(files[i]))
			numMissing = NtAcc-newT.shape[0]
			print("numMissing {:}".format(numMissing))
			if numMissing < 0:
				# Dirty hack to fix a dirty bug
				newT = newT[:numMissing]
				print(data[i].__dict__[key1].data.shape)
				print(data[i].__dict__[key2].data.shape)
				new1 = data[i].__dict__[key1].data[:,:numMissing]
				new2 = data[i].__dict__[key2].data[:,:numMissing]
				data[i].__dict__[key1].data = new1
				data[i].__dict__[key2].data = new2
				print(newT.shape)
				print(data[i].__dict__[key2].data.shape)
				print(data[i].__dict__[key1].data.shape)
			else:
				newT = np.concatenate([newT[0] - dt*np.arange(1,numMissing+1)[::-1],newT])
				new1 = np.pad(data[i].__dict__[key1].data[slcNoSkip],((0,0),(numMissing,0)),mode='constant')
				new2 = np.pad(data[i].__dict__[key2].data[slcNoSkip],((0,0),(numMissing,0)),mode='constant')
				if dims == 2:
					new1 = new1[:,np.newaxis,:]
					new2 = new2[:,np.newaxis,:]
				data[i].__dict__[key1].data = new1
				data[i].__dict__[key2].data = new2
				print(data[i-1].__dict__[key2].data.shape)
				print(data[i].__dict__[key2].data.shape)

		if yLims is not None:
			# If last time sample is before start of times desired, move on
			if newT[-1]  < (yLims[0]*1e-12 - (filtLen//2 + 1)*dt):
				continue
			# If last time sample is after start of desired times *and* first time
			# sample is before, then this is the first block we keep
			elif newT[0] < (yLims[0]*1e-12 - (filtLen//2 + 1)*dt):
				accMinInd = i

		t = np.concatenate((t,newT))

		if yLims is not None:
			# If last time sample is after end of desired times then we're done and
			# can end the loop
			if newT[-1] > (yLims[1]*1e-12 + (filtLen//2 + 1)*dt):
				accMaxInd = i+1
				break
	dt = t[1]-t[0]
	#dx = x[1]-x[0]

	# Do some sanity checking
	oNyq = math.pi/dt
	if oNyq < 2*omega0:
		print("WARNING: sampling interval too long for effective filtering!")

	# Reduce list of accumulated blocks based on desired time range
	data = data[accMinInd:accMaxInd]

	Nx = x.shape[0]
	Nt = t.shape[0]

	print("x length     : {:}".format(Nx))
	print("t length     : {:}".format(Nt))
	print("NtAcc        : {:}".format(NtAcc))
	print("Filter length: {:}\n".format(filtLen))

	# Filter by convolving filter kernel with padded blocks of data in the
	# following format (* == a sample):
	#
	#             BLOCK 1
	#   |* *|* * * * * * * * *|* *|
	#   ---------------------------
	#     ^  :       ^         :^
	#     |  :       |         : \
	#  nEdge :     nMain       : nEdge
	#        :                 :
	#        :                 :    BLOCK 2
	#        :            |* *|* * * * * * * * *|* *|
	#        :            ---------------------------
	#        :                 :                 :
	#        :                 :                etc.                :
	#        :                 :                FINAL BLOCK -> |* *|* * * *|* *|
	#        :                 :                               -----------------
	#        :                 :                                          :
	#        :                 :    OUTPUT ARRAY                          :
	#       |* * * * * * * * * * * * * * * * * ...                ... * * * |
	#       -----------------------------------                       -------
	# Where:
	#  - nMain = Number of samples that will end up in the filtered output
	#  - nEdge = floor(Nf/2) (Nf == Length of filter kernel), i.e. length of
	#            padding

	# Make nMain be a number of samples in units of accumulator blocks. This
	# may not be optimal as time to do an FFT goes as N*log(N) so there may be
	# a smarter choice.
	nMain = int(math.ceil(float(filtLen)/NtAcc)) * NtAcc
	nEdge = filtLen//2
	nTotal = nMain + 2*nEdge

	# Number of convolutions we need to do, i.e. number of padded blocks of
	# data we can construct. Note that the last block will typically have fewer
	# than nMain samples excluding padding.
	numConvolutions = int(math.ceil(float(Nt - nEdge*2)/nMain))

	# Temporary arrays for holding field data pre-convolution
	temp1 = np.zeros((Nx,nTotal))
	temp2 = np.zeros((Nx,nTotal))

	# Output array
	poynt = np.zeros((Nx,Nt-2*nEdge))

	# Initialise lSample to nEdge so that first fSample == 0
	lSample = 2*nEdge - 1
	for i in range(numConvolutions):
		print("Processing data block {:} of {:}".format(i+1,numConvolutions))

		# First & last samples required for current padded block
		fSample = lSample - 2*nEdge + 1
		lSample = min([fSample + nTotal - 1,Nt - 1])

		# First & last accumulated files required for current padded block
		fBlock = fSample // NtAcc
		lBlock = lSample // NtAcc

		print("  Need samples {:}-{:}".format(fSample+1,lSample+1))
		print("  So need accumulator blocks {:}-{:}".format(fBlock+1,lBlock+1))

		# Location within temp array to place next accumulator chunk
		tempPos = 0

		# Ensure arrays are empty before we fill them (just in case)
		temp1.fill(0.0)
		temp2.fill(0.0)

		# Loop through accumulator blocks and fill up current padded block (i.e. the
		# temp1 and temp2 arrays)
		for j in range(fBlock,lBlock+1):
			# Figure out locations of first and last samples to grab within
			# current accumulator block
			if j == fBlock:
				fBlockSample = fSample % NtAcc
			else:
				fBlockSample = 0

			if j == lBlock:
				lBlockSample = lSample % NtAcc
			else:
				lBlockSample = NtAcc-1

			# Length of current chunk
			chunkLen = lBlockSample - fBlockSample + 1

			# Insert chunk into temporary arrays
			print("  Reading from accumulator file {:}".format(j+1))
			t1 = time.time()
			temp1[:,tempPos:tempPos + chunkLen] = data[j].__dict__[key1].data[slc][:,fBlockSample:lBlockSample+1]
			temp2[:,tempPos:tempPos + chunkLen] = data[j].__dict__[key2].data[slc][:,fBlockSample:lBlockSample+1]
			print("  Finished, took {:.2f}s".format(time.time()-t1))

			tempPos += chunkLen

		# Now do the actual filtering
		if filtOut is not None:
			omegaNyq = math.pi/dt

			do = 0.5*bandwidth*omega0
			o0 = omega0
			cutoffs = [(o0-do)/omegaNyq,(o0+do)/omegaNyq]

			if filtOut == 'laser':
				# Filter out the laser
				coeffs = srsUtils.filter.winSincFilter(filtLen,cutoffs,btype='bandstop')
			elif filtOut == 'scatter':
				# Filter out everything other than the laser
				coeffs = srsUtils.filter.winSincFilter(filtLen,cutoffs,btype='bandpass')

			print("  Filtering")
			t1 = time.time()
			tempFilt1 = srsUtils.filter.convolveAxis(temp1,coeffs,1)
			tempFilt2 = srsUtils.filter.convolveAxis(temp2,coeffs,1)
			t2 = time.time()
			print(temp1.shape)
			print(tempFilt1.shape)
			print("  Finished, 2 x Convolution took {:.2f}s, data shape: {:}".format(t2-t1,tempFilt1.shape))
			#print("Convolved {:} length-{:} arrays, took {:.3f}s, {:}s each, {:}s per sample".format(data.shape[1],data.shape[0],t2-t1,(t2-t1)/data.shape[0],(t2-t1)/data.shape[0]/data.shape[1]))

		# Calculate the Poynting flux, can often ignore one EM polarisation
		#poyntTemp = (Ey*Bz - Ez*By)/const.mu_0 # x component
		#poyntTemp = (Ez*Bx - Ex*Bz)/const.mu_0 # y component
		if direction == 'x':
			poyntTemp =  tempFilt1*tempFilt2/const.mu_0
		else:
			poyntTemp = -tempFilt1*tempFilt2/const.mu_0


		#poynt = gaussian_filter1d(poynt,sigma=smoothLen/dx,axis=0)[::int(math.floor(xSigma/4))]

		# Insert filtered data into output array
		fOutIndex = fSample
		lOutIndex = lSample - 2*nEdge
		lTempIndex = lOutIndex-fOutIndex # Handles last filtered data block

		print("  Inserting filtered data into output array slice {:}:{:}".format(fOutIndex,lOutIndex))
		poynt[:,fOutIndex:lOutIndex] = poyntTemp[:,:lTempIndex]

	t = t[filtLen//2:-(filtLen//2)]

	# Apply Gaussian filter in time to smooth out oscillations at the laser frequency
	smoothTime = 4.0*math.pi/omega0
	tSigma = smoothTime/dt
	tSkip = max([ 1, int(math.floor(tSigma/2)) ])
	poyntSmooth = gaussian_filter1d(poynt,sigma=smoothTime/dt,axis=1)[:,::tSkip]

	t = t[::tSkip]

	return poyntSmooth,x,t

def plotPoyntFluxVsSpaceTime(fig,ax,poynt,x,t,I0,xLims=None,yLims=None,
                             yDims=None,maxF=None,contours=True,log=False,
                             xNe=None,ne=None,Te=None,minNeTick=None,
                             neTickInterval=None,CBar=True,CBarLabel=True,
                             markQC=True,markTPDCutoff=True,markSRSCutoff=True,
                             landauCutoff=0.30):
	'''
	Plot Poynting flux as a function of space and time along a strip of cells

	Description
	===========

	Coming soon...

	Arguments
	=========

	fig, ax : Matplotlib figure and axis objects used for plotting result
	I0      : Initial laser intensity in W/m^2
	yDims   : Number of samples along temporal dimension to plot.
	xLims   : Limits in spatial direction to plot, in μm
	yLims   : Limits in temporal direction to plot, in ps
	maxF    : maximum value for colour scale, in W/cm^2
	contours: Boolean value indicating whether to plot contours
	'''
	poynt = (poynt/1e19).transpose()
	if yDims is not None:
		tSkip = max([ 1, poynt.shape[0]//yDims ])
		poynt = poynt[::tSkip]
		print(poynt.shape)

	extent = srsUtils.misc.getExtent(x/1e-6,t/1e-12)

	if not maxF:
		maxF = I0/1e15 #np.percentile(np.abs(tSmoothedArray),99.5)
	if log:
		norm = colors.SymLogNorm(linthresh=maxF/1e3,vmin=-maxF,vmax=maxF)
	else:
		norm = colors.Normalize(vmin=-maxF,vmax=maxF)
	im = ax.imshow(poynt,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='RdBu_r',norm=norm)

	levels = np.array([0.25,0.5,0.75,1.0])*I0/1e15
	if contours:
		ax.contour(poynt,levels,extent=extent,origin='lower',linewidths=1.0,colors='grey',linestyles=[':','-.','--','-'])

	if CBar:
		cb = fig.colorbar(im,orientation='vertical',ax=ax)
		cb.ax.yaxis.set_offset_position('left')

	ax.tick_params(reset=True,axis='both',color='k',direction='in')

	#ne = cropArray(ne,_nCropx,_nCropy)

	# Annotate location of important densities
	if ne is not None:
		ax2 = ax.twiny()
		srsUtils.misc.addNeScale(ax,ax2,xNe/1e-6,ne,minVal=minNeTick,interval=neTickInterval,minor=True,tickColor='k')

		if markQC:
			if np.any(np.where(ne/srsUtils.nCritNIF >= 0.25)) \
			  and np.any(np.where(ne/srsUtils.nCritNIF <= 0.25)):
				xQC = xNe[np.min(np.where(ne/srsUtils.nCritNIF >= 0.25))]
				ax.axvline(xQC/1e-6,color='k',linestyle='--')
			if np.any(np.where(ne/srsUtils.nCritNIF >= 0.25**2)) \
			  and np.any(np.where(ne/srsUtils.nCritNIF <= 0.25**2)):
				xSC = xNe[np.min(np.where(ne/srsUtils.nCritNIF >= 0.25**2))]
				ax.axvline(xSC/1e-6,color='k',linestyle='-.')

		if Te is not None:
			bth = math.sqrt(const.k*Te/const.m_e)/const.c
			if markTPDCutoff:
				tpdCutoffNe = srsUtils.tpd.landauCutoffDens(bth,cutoff=landauCutoff)
				diff = np.abs(ne/srsUtils.nCritNIF-tpdCutoffNe)
				xCutoffTPD = xNe[np.where(diff == np.min(diff))]
				ax.axvline(xCutoffTPD/1e-6,linestyle='--',color='k',linewidth=0.8)
			if markSRSCutoff:
				srsCutoffNe = srsUtils.srs.landauCutoffDens(bth,math.pi,cutoff=landauCutoff)
				diff = np.abs(ne/srsUtils.nCritNIF-srsCutoffNe)
				xCutoffSRS = xNe[np.where(diff == np.min(diff))]
				ax.axvline(xCutoffSRS/1e-6,linestyle='--',color='g',linewidth=0.8)

	if xLims is not None:
		ax.set_xlim(xLims)

	if yLims is not None:
		ax.set_ylim(yLims)

	ax.set_xlabel(r'x $/\mu$m')
	ax.set_ylabel('time /ps')
	#cb.set_label('$I/10^{15}$ Wcm$^{-2}$')
	if CBarLabel:
		cb.ax.set_title('$I/10^{15}$ Wcm$^{-2}$')


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('dataDir')
	parser.add_argument('I',type=float)
	parser.add_argument('--omega0', type=float, default=srsUtils.omegaNIF)
	parser.add_argument('--space',action='store_true')
	parser.add_argument('--filter',choices=('laser','scatter'),default='laser')
	parser.add_argument('--direction',choices=('x','y'),default='x')

	parser.add_argument('--prefix',default='strip_')
	parser.add_argument('--subset')
	parser.add_argument('--oneDim',action='store_true')

	parser.add_argument('--useCached',action='store_true')
	parser.add_argument('--cacheFile')

	parser.add_argument('--log',action='store_true')
	parser.add_argument('--xLims',type=float,nargs=2)
	parser.add_argument('--yLims',type=float,nargs=2)
	parser.add_argument('--maxF',type=float)
	parser.add_argument('--xDims',type=int)
	parser.add_argument('--yDims',type=int)

	parser.add_argument('--densProfFile',default='regular_0000.sdf')
	parser.add_argument('--densitySpecies',default='electrons')
	parser.add_argument('--Te',type=float)

	parser.add_argument('--noMarkQC',action='store_true')
	parser.add_argument('--markTPDCutoff',action='store_true')
	parser.add_argument('--markSRSCutoff',action='store_true')
	parser.add_argument('--landauCutoff',type=float,default=0.30)
	parser.add_argument('--noCBar',action='store_true')
	parser.add_argument('--noCBarLabel',action='store_true')
	parser.add_argument('--minNeTick',type=float)
	parser.add_argument('--neTickInterval',type=float)
	parser.add_argument('--contour',action='store_true')

	parser.add_argument('--fontSize',type=float)
	parser.add_argument('-o','--output')
	parser.add_argument('--figSize',type=float,nargs=2)

	args = parser.parse_args()

	# Main calculation, read + filter data
	if args.space:
		if args.useCached:
			npCache = np.load(args.cacheFile)
			poynt = npCache['poynt']
			x     = npCache['x']
			t     = npCache['t']
		else:
			files = sdfUtils.listFiles(args.dataDir,args.prefix)
			#files = args.dataDir
			if len(files) == 0:
				raise EnvironmentError("No SDF files found with prefix \"{:}\" in directory {:}".format(args.prefix,args.dataDir))

			print("Found {:} sdf files with prefix \"{:}\" in {:}".format(len(files),args.prefix,args.dataDir))

			if args.oneDim:
				if args.subset is None:
					subset='field'
				else:
					subset = args.subset
				dims=1
			else:
				if args.subset is None:
					subset='strip_ss'
				else:
					subset = args.subset
				dims=2

			poynt,x,t = poyntFluxVsSpaceTime(files,omega0=args.omega0,
			    filtOut=args.filter,xLims=args.xLims,yLims=args.yLims,
			    xDims=args.xDims,dims=dims,subset=subset,direction=args.direction)
			if args.cacheFile is not None:
				np.savez_compressed(args.cacheFile,poynt=poynt,x=x,t=t)
	else:
		# TODO: Get this working again
		poyntFluxVsTime(ax,files,args.I,xLims=args.xLims,log=args.log,subset=subset)

	if args.densProfFile is not None:
		# Read density profile
		sdfProf = sdf.read(os.path.join(args.dataDir,args.densProfFile))
		xNe = sdfProf.Grid_Grid.data[0]
		xNe = 0.5*(xNe[1:] + xNe[:-1])

		if args.densitySpecies == '':
			ne = sdfProf.__dict__['Derived_Number_Density'].data
		else:
			ne = sdfProf.__dict__['Derived_Number_Density_'+args.densitySpecies].data

		if not args.oneDim:
			ne = np.mean(ne,axis=1)
		ne = gaussian_filter1d(ne,sigma=10)
	else:
		xNe = None
		ne  = None

	if args.Te is not None:
		Te = args.Te*srsUtils.TkeV
	else:
		Te = None

	# Plotting stuff goes below here
	if args.fontSize:
		import matplotlib as mpl
		mpl.rcParams.update({'font.size':args.fontSize})

	fig = plt.figure()
	ax = fig.add_subplot(111)

	plotPoyntFluxVsSpaceTime(fig,ax,poynt,x,t,args.I,xLims=args.xLims,
		yLims=args.yLims,maxF=args.maxF,yDims=args.yDims,contours=args.contour,
		log=args.log,xNe=xNe,ne=ne,Te=Te,minNeTick=args.minNeTick,
		neTickInterval=args.neTickInterval,CBar=not args.noCBar,
		CBarLabel=not args.noCBarLabel,markQC=not args.noMarkQC,
		markTPDCutoff=args.markTPDCutoff,markSRSCutoff=args.markSRSCutoff,
		landauCutoff=args.landauCutoff)

	if args.output:
		if args.figSize:
			fig.set_size_inches(args.figSize)
		fig.savefig(args.output)
	else:
		plt.show()
