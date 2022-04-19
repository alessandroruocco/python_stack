#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.colors as colors
import scipy.constants as const
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d
import time
import os
import re
import sys
import sdf

import srsUtils
from srsUtils import misc
import sdfUtils

style.use('classic')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('figure', autolayout=True)


# Output directory
_outputDir = 'processed'

# ID of burst to read (won't be necessary with subsets)
_burstNum  = 4

# Maximum amount of data to read into memory
_maxDataSize = 32*1024**3

# Hacky fix for bug affecting job34
_truncate=True

def readBoundary(dataDir,first,num,boundary,reqData,prefix,accumulated=True,is1D=False,reducedOut=False):
	'''
	Reads a series of boundary data samples

	Parameters
	----------

	dataDir : directory SDF files are in
	first : First sample number
	num : number of samples to read
	boundary : string indicating which boundary data to grab, e.g. 'x_min'
	'''
	files = sdfUtils.listFiles(dataDir,prefix)
	if accumulated:
		NtAcc = sdfUtils.numAcc(files[0])
		fFile = first//NtAcc
		lFile = (first+num)//NtAcc
		if (first+num)%NtAcc:
			lFile += 1
	else:
		fFile = first
		lFile = first+num

	files = files[fFile:lFile]
	metaData = sdfUtils.getManySDFMetadata(files)
	steps = np.array([ d['step'] for d in metaData ])
	if len(files) > 1:
		diffs = np.diff(steps)
		#if not np.all(diffs == stats.mode(diffs)[0][0]):
		#	raise ValueError("Data files are not equally spaced in time")

	print("Reading {:} boundary data".format(boundary))
	sTime = time.time()
	if accumulated:
		if is1D:
			if boundary == 'x_min':
				slc = [slice(0,1)]
			else:
				slc = [slice(-2,-3,-1)]
			if reducedOut:
				if boundary == 'x_min':
					data = sdfUtils.readMultipleAcc(files,reqData,'left_boundary',slc)
				else:
					data = sdfUtils.readMultipleAcc(files,reqData,'right_boundary',slc)
			else:
				data = sdfUtils.readMultipleAcc(files,reqData,'field',slc)

		else:
			data = sdfUtils.readMultipleAcc(files,reqData,'{:}_ss'.format(boundary))
		data['time'] = data['time'][first % NtAcc:(first % NtAcc) + num]
		for d in reqData:
			data[d] = data[d][first % NtAcc:(first % NtAcc) + num]
	else:
		data = sdfUtils.readMultiple(files,reqData,blockSize=100,
		                             fancyProgress=True,altGrid='Grid_{:}_ss'.format(boundary))
	print("Finished reading data, took {:.3f}s".format(time.time()-sTime))

	data['sample'] = np.arange(first,first+num)
	return data

def showBandRejectFilter(freq,bandwidth,filtLen):
	cutoffs = [freq-0.5*bandwidth,freq+0.5*bandwidth]
	coeffs = srsUtils.filter.winSincFilter(filtLen,cutoffs,btype='bandstop')

	srsUtils.filter.plotFIRFilterInfo1D(coeffs)

def filterLaser(field,omega0,dt,bandwidth,filtLen):
	'''
	Filters the laser frequency from an array of data

	The first dimension is assumed to be time with the second space

	Parameters
	----------

	field : Array with dimensions (Nt,Nx,Ny,...) to be filtered
	ts   : Array of time values, length Nt
	bandwidth : Filter bandwidth, in units of ω_0
	filtLen : Filter length in samples
	'''
	omegaNyq = math.pi/dt

	do = 0.5*bandwidth*omega0
	o0 = omega0
	print(o0)
	cutoffs = [(o0-do)/omegaNyq,(o0+do)/omegaNyq]
	print(cutoffs)
	coeffs = srsUtils.filter.winSincFilter(filtLen,cutoffs,btype='bandstop')

	t1 = time.time()
	if np.isnan(field).any(): print("Warning: data contains NaN")
	filtData = srsUtils.filter.convolveAxis(field,coeffs,0)
	laserData = field[filtLen//2:-filtLen//2+1] - filtData
	t2 = time.time()
	print("Convolution took {:}s, data shape: {:}".format(t2-t1,field.shape))
	#print("Convolved {:} length-{:} arrays, took {:.3f}s, {:}s each, {:}s per sample".format(data.shape[1],data.shape[0],t2-t1,(t2-t1)/data.shape[0],(t2-t1)/data.shape[0]/data.shape[1]))

	return filtData,laserData

def filterAtBoundary(dataDir,omega0,boundary,prefix,firstSample,numSamples,
                     bandwidth,filtLen,is1D,reducedOut=False,polarisation='both'):
	'''
	Filters all fields over a series of samples at a given boundary

	Parameters
	----------

	dataDir : directory in which SDF files are located
	boundary : boundary to filter (e.g. 'x_min')
	firstSample : Number of the first sample
	numSamples : Number of samples to filter
	bandwidth : Filter bandwidth in units of ω_0
	filtLen : Filter length
	'''

	if is1D:
		# For the 1D case assume propagation along x
		if polarisation == 'both':
			reqData = ['Electric_Field_Ey','Electric_Field_Ez',
			           'Magnetic_Field_By','Magnetic_Field_Bz']
		elif polarisation == 'y':
			reqData = ['Electric_Field_Ey','Magnetic_Field_Bz']
		elif polarisation == 'z':
			reqData = ['Electric_Field_Ez','Magnetic_Field_By']
		else:
			raise ValueError('Unrecognised polarisation \"{:}\"'.format(polarisation))
	else:
		# For 2D case assume propagation along x or y
		if polarisation == 'both':
			reqData = ['Electric_Field_Ex','Electric_Field_Ey',
			           'Electric_Field_Ez','Magnetic_Field_Bx',
			           'Magnetic_Field_By','Magnetic_Field_Bz']
		elif polarisation == 'y':
			reqData = ['Electric_Field_Ex','Electric_Field_Ey',
			           'Magnetic_Field_Bz']
		elif polarisation == 'z':
			reqData = ['Electric_Field_Ez','Magnetic_Field_Bx',
			           'Magnetic_Field_By']
		else:
			raise ValueError('Unrecognised polarisation \"{:}\"'.format(polarisation))

	data = readBoundary(dataDir,firstSample,numSamples,boundary,reqData,prefix,is1D=is1D,reducedOut=reducedOut)
	#print(data.keys())
	#print("Here")
	for d in data.keys():
		if d in ['sample', 'space', 'time']: continue
		#print(d)
		name = [ n for n in reqData if n in d ][0]
		#print(name)
		vals = data.pop(d)
		data[name] = vals
	#print(data.keys())

	dt = data['time'][1]-data['time'][0]

	filtData = {}
	laserData = {}
	filtData['time'] = data['time'][filtLen//2:data['time'].shape[0]-filtLen//2]
	laserData['time'] = filtData['time']

	fFiltSample = firstSample + filtLen//2
	lFiltSample = fFiltSample + len(filtData['time'])
	filtData['sample'] = np.arange(fFiltSample,lFiltSample)
	filtData['space'] = data['space']
	laserData['sample'] = np.arange(fFiltSample,lFiltSample)
	laserData['space'] = data['space']
	for d in reqData:
		print(d)
		print(data[d].shape)
		filtResults = filterLaser(data[d],omega0,dt,bandwidth,filtLen)
		filtData[d] = filtResults[0]
		laserData[d] = filtResults[1]

	filtData['filtLen'] = filtLen
	filtData['filtWidth'] = omega0*bandwidth
	laserData['filtLen'] = filtLen
	laserData['filtWidth'] = omega0*bandwidth

	return filtData,laserData

def filterSimulationBoundaries(dataDir,omega0,bandwidth,filtLen,prefix,is1D,overwrite,reducedOut=False,polarisation='both'):
	'''
	Performs filtering of all simulation boundary data and saves it to disk
	'''
	filtLen = int(filtLen)

	#files = sdfUtils.listFiles(dataDir,'burst{:}_'.format(_burstNum))
	files = sdfUtils.listFiles(dataDir,prefix=prefix)
	if not len(files): raise IOError("Couldn't find any boundary SDF files")

	# Size of boundary data files
	fileSize = os.path.getsize(files[0])

	# Number of accumulated snapshots in file
	NtAcc = sdfUtils.numAcc(files[0])

	# Data per accumulated snapshot
	sampleSize = fileSize//NtAcc

	# Number of snapshots that can be processed with available memory
	chunkLen = _maxDataSize//sampleSize

	if chunkLen < filtLen: raise MemoryError("Not enough memory to filter data")
	#chunkLen = 1500

	numChunks = int(math.ceil((len(files)*NtAcc - 2*(filtLen//2))/float(chunkLen - 2*(filtLen//2))))
	if not numChunks: raise IOError("Insufficient data for filter of specified length")

	print("Max memory {:.2f}GiB".format(_maxDataSize/1024.**3))
	print("Boundary data files each {:.2f}GiB, containing {:} snapshots".format(fileSize/1024.**3,NtAcc))
	print("Processing {:} chunks of {:} snapshots".format(numChunks,chunkLen))

	for i in range(numChunks):
		first = i*chunkLen - i*2*(filtLen//2)
		if is1D:
			boundaryList = ['x_min', 'x_max']
		else:
			boundaryList = ['x_min','x_max','y_min','y_max']
		for boundary in boundaryList:
			print("Filtering {:} boundary, samples {:}-{:}".format(boundary,first,first+chunkLen))
			filtData,laserData = filterAtBoundary(dataDir,omega0,boundary,
			    prefix,first,chunkLen,bandwidth,filtLen,is1D,reducedOut,
			    polarisation=polarisation)
			print("Filtering complete\n")

			print("Saving to disk")
			saveFilteredBoundaryData(dataDir,boundary,filtData,laserData,overwrite)
			print("Finished saving to disk\n")

def saveFilteredBoundaryData(dataDir,boundary,filtData,laserData,overwrite=False):
	outDir  = os.path.join(dataDir,_outputDir)
	if not os.path.exists(outDir):
		os.makedirs(outDir)

	fSample = filtData['sample'][0]
	lSample = filtData['sample'][-1]
	outFile = os.path.join(os.path.join(dataDir,_outputDir),'filtered_{:}_{:}-{:}.npz'.format(boundary,fSample,lSample))
	if not overwrite and os.path.exists(outFile):
		raise IOError("Boundary filtered data file already exists")

	np.savez(outFile,**filtData)

	outFile = os.path.join(os.path.join(dataDir,_outputDir),'laser_{:}_{:}-{:}.npz'.format(boundary,fSample,lSample))
	if not overwrite and os.path.exists(outFile):
		raise IOError("Boundary laser data file already exists")

	np.savez(outFile,**laserData)

def getFilteredFileRange(fName):
	'''
	Extract range of time samples from file name

	Note that this range is inclusive
	'''
	return [int(n) for n in fName.split('_')[-1].split('.')[0].split('-')]

def listFilteredFiles(dataDir,boundary,noLaser=True):
	if noLaser:
		npzFiles = [ os.path.join(dataDir,f) for f in os.listdir(dataDir) if re.match('filtered_{:}_[0-9]*-[0-9]*\.npz'.format(boundary),f) ]
	else:
		npzFiles = [ os.path.join(dataDir,f) for f in os.listdir(dataDir) if re.match('laser_{:}_[0-9]*-[0-9]*\.npz'.format(boundary),f) ]
	npzFiles.sort(key=lambda name: getFilteredFileRange(name)[0])

	return npzFiles

def loadFilteredBoundaryData(dataDir,boundary,components='all',noLaser=True,
                             fs=None,ns=None,ls=None):
	'''
	Loads previously filtered boundary data from .npz files

	Parameters
	----------

	dataDir : Directory in which the simulation was run (not the subdirectory
	          containing processed data)
	boundary : Boundary for which the data is required
	fs : First sample to read
	ns : Number of samples to read. Either this or ls must be specified
	ls : End of sample range (i.e. the last sample is ls-1). Either this or ns
	     must be specified
	'''
	# Find filtered data files
	outDir  = os.path.join(dataDir,_outputDir)
	files = listFilteredFiles(outDir,boundary,noLaser)
	ranges = [ getFilteredFileRange(f) for f in files ]

	# If start and end data indices have not been supplied, make these span as
	# large a range as possible
	if not fs:
		fs = ranges[0][0]

	if not ns and not ls:
		ls = ranges[-1][-1]
		ns = ls-fs

	# If one of ls and ns have been supplied, calculate the other
	if not ns: ns = ls-fs
	else: ls = fs+ns

	# Define desired range (inclusive)
	sRange = (fs,ls-1)

	#print("Loading samples {:} - > {:}".format(sRange[0],sRange[1]))

	# Cut list of files down to those matching the time range requested
	files = [ f for f in files if misc.overlap(sRange,getFilteredFileRange(f),inclusive=True) ]

	# Ensure that we've got data spanning the range of times requested
	filesRange = (getFilteredFileRange(files[0])[0],getFilteredFileRange(files[-1])[1])
	overlap = misc.overlap(sRange,filesRange,inclusive=True)
	if overlap < ns:
		raise IOError("Couldn't find all the requested filtered data")

	# Read files in series and concatenate arrays that have a time-dependency
	files = [ f for f in files ]
	if components == 'all':
		#extVars = ['Electric_Field_Ex_Acc_{:}_ss'.format(boundary),
		#           'Electric_Field_Ey_Acc_{:}_ss'.format(boundary),
		#           'Electric_Field_Ez_Acc_{:}_ss'.format(boundary),
		#           'Magnetic_Field_Bx_Acc_{:}_ss'.format(boundary),
		#           'Magnetic_Field_By_Acc_{:}_ss'.format(boundary),
		#           'Magnetic_Field_Bz_Acc_{:}_ss'.format(boundary),
		#		   'sample', 'time']
		extVars = ['Electric_Field_Ex',
		           'Electric_Field_Ey',
		           'Electric_Field_Ez',
		           'Magnetic_Field_Bx',
		           'Magnetic_Field_By',
		           'Magnetic_Field_Bz',
		           'sample', 'time']
	elif isinstance(components,basestring):
		extVars = ['{:}_Acc_{:}_ss'.format(components,boundary),
		           'sample', 'time']
	else:
		extVars = ['{:}_Acc_{:}_ss'.format(c,boundary) for c in components]
		extVars += ['sample','time']

	# Offsets of first and last samples from end of their respective files
	fOffset = sRange[0]-filesRange[0]
	lOffset = sRange[1]-filesRange[1]

	# Read first filtered data file and populate output dictionary with first arrays
	filtData = {}
	with np.load(files[0],allow_pickle=True) as tempData:
		if tempData['sample'][-1] == filesRange[1]: l = len(tempData['sample'])+lOffset
		else: l = len(tempData['sample'])
		for k in tempData.keys():
			if k in extVars:
				filtData[k] = tempData[k][fOffset:l]
			else:
				filtData[k] = tempData[k]

	# Read the rest of the filtered data and concatenate arrays with a time dependence
	for f in files[1:]:
		tempData = np.load(f)
		if tempData['sample'][-1] == filesRange[1]: l = len(tempData['sample'])+lOffset
		else: l = len(tempData['sample'])
		for v in extVars:
			#print(v)
			#print(filtData.keys())
			# Hacky fix to get around EPOCH output bug
			# On some jobs (job34 only so far) the output shape changed part way through
			# the simulation. This ignores that and resizes arrays + pads with zeroes...
			if filtData[v].shape[1:] != tempData[v].shape[1:]:
				sF = np.array(filtData[v].shape[1:])
				sT = np.array(tempData[v].shape[1:])
				newShape = np.min(zip(sF,sT),axis=1)

				# This only works for 2D datasets
				filtData[v] = filtData[v][:,:newShape[0],:newShape[1]]
				tD = tempData[v][:,:newShape[0],:newShape[1]]
				filtData[v] = np.concatenate((filtData[v],tD[:l]))
			else:
				filtData[v] = np.concatenate((filtData[v],tempData[v][:l]))

	return filtData

def fluxAtBoundary(data,boundary,total=False,polarisation='both'):
	'''
	Calculates the flux through a boundary in W/m^2

	Uses the individual field components F(x,t) to calculate the Poynting flux
	perpendicular to the boundary.
	'''
	Ex = Ey = Ez = Bx = By = Bz = 0.0
	if boundary[0] == 'x':
		if (polarisation == 'y') or (polarisation == 'both'):
			Ey = data['Electric_Field_Ey']
			Bz = data['Magnetic_Field_Bz']
		if (polarisation == 'z') or (polarisation == 'both'):
			Ez = data['Electric_Field_Ez']
			By = data['Magnetic_Field_By']
	else:
		if (polarisation == 'y') or (polarisation == 'both'):
			Ex = data['Electric_Field_Ey']
			Bz = data['Magnetic_Field_Bz']
		if (polarisation == 'z') or (polarisation == 'both'):
			Ez = data['Electric_Field_Ez']
			Bx = data['Magnetic_Field_Bx']

	if polarisation == 'y':
		c1 = 1.0
		c2 = 0.0
	elif polarisation == 'z':
		c1 = 0.0
		c2 = 1.0
	elif polarisation == 'both':
		c1 = 1.0
		c2 = 1.0
	else:
		raise ValueError('Unrecognized value for polarisation: \"{:}\"'.format(polarisation))

	if boundary[0] == 'x':
		outFlux = (c1*Ey*Bz - c2*Ez*By)/const.mu_0
	elif boundary[0] == 'y':
		outFlux = (c2*Ez*Bx - c1*Ex*Bz)/const.mu_0
		#print(boundary)
		#outFlux = Ez#Bx/const.mu_0

		#plt.plot(np.abs(Ez[1000,:,0]))
		#plt.grid()
		#ax = plt.gca()
		#ax.set_yscale('log')
		#plt.show()
	else: raise ValueError("Unrecognised boundary dimension")

	if boundary[2:] == 'min':
		outFlux *= -1.0

	# Sum over all axes other than first (time) axis
	if total:
		axes = tuple(range(1,len(outFlux.shape)))
		outFlux = np.sum(outFlux,axis=axes)

	return outFlux

def powerAtBoundary(outFlux,grid,boundary):
	'''
	Converts Poynting flux through a boundary into a power
	'''
	if boundary[0] == 'x':
		outPow = outFlux*(grid[1][1]-grid[1][0])
	elif boundary[0] == 'y':
		outPow = outFlux*(grid[0][1]-grid[0][0])

	return outPow

def totalFluxes(dataDir,ne,is1D,polarisation):
	'''
	Calculates the total flux through each boundary

	Integrates in space, then filters the fluxes through each boundary again to
	remove oscillations with frequency ω>ω_p.
	'''
	if is1D:
		boundaries = ['x_min','x_max']
	else:
		boundaries = ['x_min','x_max','y_min','y_max']
		#boundaries = ['x_min','x_max','y_min']

	# Load first two data samples to get dt for filter coefficients
	dataFilt = loadFilteredBoundaryData(dataDir,'x_min',ns=2)
	dt = dataFilt['time'][1]-dataFilt['time'][0]
	filtLen = 501

	#o0 = srsUtils.omegaNIF
	op = math.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
	omegaNyq = math.pi/dt
	cutoff = 0.5*op/omegaNyq
	coeffs = srsUtils.filter.winSincFilter(filtLen,cutoff,btype='lowpass')

	#srsUtils.filter.plotFIRFilterInfo1D(coeffs)

	boundRefl = {}
	boundLas = {}
	t = np.array([])
	gotTime = False
	for b in boundaries:
		print("Integrating flux spatially over {:} boundary".format(b))

		# Find size of each data file
		filtFiles = listFilteredFiles(os.path.join(dataDir,_outputDir),b)
		fileSize = os.path.getsize(filtFiles[0])

		# Number of samples in each file
		fileRanges = [ getFilteredFileRange(f) for f in filtFiles ]
		fileNumSamples = [ R[1] - R[0] + 1 for R in fileRanges ] # + 1 as range is inclusive
		totalSamples = sum(fileNumSamples)
		NtPerFile = fileNumSamples[0]

		# Data per sample
		sampleSize = fileSize//NtPerFile

		# Number of snapshots that can be processed with available memory
		chunkLen = _maxDataSize//sampleSize

		if chunkLen < filtLen: raise MemoryError("Not enough memory to filter data")
		#chunkLen = 1500

		numChunks = int(math.ceil((totalSamples - 2*(filtLen//2))/float(chunkLen - 2*(filtLen//2))))
		if not numChunks: raise IOError("Insufficient data for filter of specified length")

		print("Max memory {:.2f}GiB".format(_maxDataSize/1024.**3))
		print("Filtered boundary files each {:.2f}GiB, containing max. {:} snapshots".format(fileSize/1024.**3,NtPerFile))
		print("Total number of samples {:}, processing in {:} chunks of max. {:} samples".format(totalSamples,numChunks,chunkLen))
		print("Filtered data samples {:} -> {:}".format(fileRanges[0][0],fileRanges[-1][1]))

		reflFilt = np.array([])
		for i in range(numChunks):
			firstSample = fileRanges[0][0] + i*chunkLen
			lastSample  = min([firstSample+chunkLen,fileRanges[-1][1]])
			dataFilt = loadFilteredBoundaryData(dataDir,b,noLaser=True,
			                                    fs=firstSample,ls=lastSample)
			if not gotTime: t = np.concatenate([t,dataFilt['time']]) # Only need to do this once
			reflFilt = np.concatenate([reflFilt,fluxAtBoundary(dataFilt,b,total=True,polarisation=polarisation)])
		if not is1D: reflFilt = powerAtBoundary(reflFilt,dataFilt['space'],b)
		reflFilt2 = srsUtils.filter.convolveAxis(reflFilt,coeffs,0)

		gotTime = True

		boundRefl[b] = reflFilt2

		lasFilt = np.array([])
		for i in range(numChunks):
			firstSample = fileRanges[0][0] + i*chunkLen
			lastSample  = min([firstSample+chunkLen,fileRanges[-1][1]])
			dataFilt = loadFilteredBoundaryData(dataDir,b,noLaser=False,
			                                    fs=firstSample,ls=lastSample)
			lasFilt = np.concatenate([lasFilt,fluxAtBoundary(dataFilt,b,total=True,polarisation=polarisation)])
		if not is1D: lasFilt = powerAtBoundary(lasFilt,dataFilt['space'],b)
		lasFilt2 = srsUtils.filter.convolveAxis(lasFilt,coeffs,0)

		boundLas[b] = lasFilt2

	boundRefl['time'] = t[filtLen//2:-filtLen//2+1]
	boundLas['time']  = t[filtLen//2:-filtLen//2+1]

	outDir = os.path.join(dataDir,_outputDir)
	np.savez_compressed(os.path.join(outDir,'totalRefl.npz'),**boundRefl)
	np.savez_compressed(os.path.join(outDir,'totalLas.npz' ),**boundLas )

def specklePower(k0,F,I):
	w = 2.0*math.pi*F/k0
	P = math.sqrt(0.5*math.pi)*I*w/math.sqrt(2.0)
	#                                  ^
	# FIXME:   This is a fudge factor, figure out why it's needed
	return P

def planeWavePower(k0,I,snapshotFile):
	'''
	Calculates the expected incoming laser power for a plane-wave simulation
	'''
	data = sdf.read(snapshotFile)
	grid = data.Grid_Grid.data[1]
	Ly = grid[-1]-grid[0]

	ne = data.Derived_Number_Density_electrons.data

	# Calculate mean laser boundary density and error
	neErr = np.mean(ne[0])/np.sqrt(ne.shape[1])
	ne = np.mean(ne[0])

	print(I*Ly)
	return I*Ly

def empiricalLaserPower(time,xminLPow,laserDelay=0.175e-12,lpiOnset=0.325e-12):
	'''
	Empirically calculate the laser power

	Uses the initial power through the left boundary at ω_0 to estimate the
	laser's power. Assumes no SBS is occurring during this period.

	Returns the laser power and an estimated error
	'''
	roi = np.where(np.logical_and(time > laserDelay,time < laserDelay + lpiOnset))
	t = time[roi]
	lPow = -xminLPow[roi]

	meanPow = np.mean(lPow)
	stdPow  = np.std(lPow)

	#fig,ax = plt.subplots(1,1)
	#ax.plot(t,lPow)
	#ax.grid()
	#ax.set_yscale('log')
	##ax.set_ylim(0.0,ax.get_ylim()[1])
	#plt.show()

	return meanPow,stdPow

def plotReflectivity(ax,reflFile,lasFile,P0,tLims=None,toPlot=None,
                     noLegend=False,smoothTime=None):
	remove = None
	# Removes spikes from job9
	#remove = [(17000,17500),(38500,39000),(47000,47400)]
	#remove = [(10000,10001)]
	#remove = [(69500,71000)] # for job48

	rAliases = {'x_min':'bSRS', 'x_max':'fSRS', 'y_min':'y_min', 'y_max':'y_max'}

	rFlux = np.load(reflFile) # Load reflectivity data
	tFlux = np.load(lasFile) # Load transmission data

	if remove is not None:
		for i,j in remove:
			rFluxTemp = {}
			for k,v in rFlux.items():
				if k != 'time':
					v[i:j] = 0.5*(v[i] + v[j])
				rFluxTemp[k] = v
			rFlux = rFluxTemp

			tFluxTemp = {}
			for k,v in tFlux.items():
				if k != 'time':
					v[i:j] = 0.5*(v[i] + v[j])
				tFluxTemp[k] = v
			tFlux = tFluxTemp

	if smoothTime is not None:
		dt = rFlux['time'][-1] - rFlux['time'][-2]
		rFlux = { (k):(gaussian_filter1d(v,sigma=smoothTime/dt) if k != 'time' else v)
				  for k,v in rFlux.items() }
		tFlux = { (k):(gaussian_filter1d(v,sigma=smoothTime/dt) if k != 'time' else v)
				  for k,v in tFlux.items() }

	if tLims is not None:
		if tLims[0] is not None and tLims[1] is None:
			tLims = (tLims[0],rFlux['time'][-1]/1e-12)
		elif tLims[0] is None and tLims[0] is not None:
			tLims = (rFlux['time'][0]/1e-12,tLims[1])
		elif tLims[0] is None and tLims[1] is None:
			tLims = (rFlux['time'][0]/1e-12,rFlux['time'][-1]/1e-12)
	else:
		tLims = (rFlux['time'][0]/1e-12,rFlux['time'][-1]/1e-12)


	total = 0.0
	time = rFlux['time']
	roi = np.where(np.logical_and(time > tLims[0]*1e-12,time < tLims[1]*1e-12))

	# Plot time history of SRS scatter from each boundary
	print("Time-averaged scatter (relative to laser power):")
	for b in sorted(rFlux.keys()):
		if b == 'time': continue

		r = rFlux[b]

		# Calculate time average scatter fraction
		rRoi = r[roi]
		print(" - SRS {:}: {:.2f}%, σ: {:.2f}%".format(b,np.mean(100*rRoi/P0),np.std(100*rRoi/P0)))

		total += r

		if toPlot is not None:
			if b not in toPlot: continue

		#ax.plot(time/1e-12,r/P0,label='$'+b.replace('_','_{\mathrm{')+'}}$')
		ax.plot(time/1e-12,r/P0,label=rAliases[b])

	totalRoi = total[roi]
	print(" - SRS (all boundaries): {:.2f}%, σ: {:.2f}%".format(np.mean(100*totalRoi/P0),np.std(100*totalRoi/P0)))

	# Plot SBS time history
	sbs = tFlux['x_min'] + P0
	sbsRoi = sbs[roi]
	print(" - SBS x_min: {:.2f}%, σ: {:.2f}%".format(np.mean(100*sbsRoi/P0),np.std(100*sbsRoi/P0)))

	if toPlot is None or 'SBS' in toPlot:
		ax.plot(time/1e-12,sbs/P0,color='olive',label='SBS')

	if toPlot is None or 'total' in toPlot:
		ax.fill_between(time/1e-12,total/P0,linewidth=0.0,facecolor='k',alpha=0.25)
		ax.plot(time/1e-12,total/P0,'k-',label='Total')

	#totTrans = np.sum([ tFlux[b] for b in tFlux if b != 'time' and b != 'x_min' ],axis=0)
	totTrans = tFlux['x_max']
	print("Time-averaged transmission: {:.2f}%, σ {:.2f}%".format(np.mean(100*totTrans[roi]/P0),np.std(100*totTrans[roi]/P0)))

	if toPlot is None or 'transmission' in toPlot:
		ax.plot(time/1e-12,totTrans/P0,color='purple',label='Transmission')

	ax.set_xlim(tLims[0],tLims[1])
	#ax.set_yscale('log')
	#ax.set_ylim(0.0,1.0)

	ax.set_xlabel('time /ps')
	ax.set_ylabel('$P/P_0$')
	if not noLegend:
		ax.legend(fontsize=args.fontSize-1,ncol=1,loc='best')

	ax.grid()

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('dataDir')

	parser.add_argument('--prefix',default='boundary_')
	parser.add_argument('--polarisation',choices=['y','z','both'],default='both')
	parser.add_argument('--1D',action='store_true')

	parser.add_argument('ne',type=float)
	parser.add_argument('--omega0',type=float,default=srsUtils.omegaNIF)
	parser.add_argument('-b','--bandwidth',type=float,default=0.3)
	parser.add_argument('-l','--filtLen',type=int,default=1001)
	parser.add_argument('-p','--plotFilter',action='store_true')
	parser.add_argument('--overwrite',action='store_true')
	parser.add_argument('-nf','--noFilter',action='store_true')
	parser.add_argument('--onlyTotal',action='store_true')
	parser.add_argument('-r','--plotReflectivity',action='store_true')
	parser.add_argument('-I',type=float)
	parser.add_argument('-F',type=float)
	parser.add_argument('--calcLaserPower',action='store_true')
	parser.add_argument('--lpiOnset',type=float,default=0.325e-12)
	parser.add_argument('--laserRiseTime',type=float,default=0.175e-12)
	parser.add_argument('--smoothTime',type=float)
	parser.add_argument('--toPlot',
	                    choices=['x_min','x_max','y_min','y_max','SBS',
	                             'transmission','total'],
						nargs='+')

	parser.add_argument('--fontSize',type=float)
	parser.add_argument('-o','--output')
	parser.add_argument('-fs','--figSize',type=float,nargs=2,default=(5,4))
	parser.add_argument('--maxt',type=float)
	parser.add_argument('--mint',type=float)
	parser.add_argument('--maxR',type=float,default=1.0)
	parser.add_argument('--minR',type=float,default=0.0)
	parser.add_argument('--noLegend',action='store_true')

	args = parser.parse_args()

	if args.fontSize:
		import matplotlib as mpl
		mpl.rcParams.update({'font.size':args.fontSize})
	# If ω_Nyq = 4*ω_0 and ω_filt = ω_0, then ω_filt/ω_Nyq = 0.25
	#showBandRejectFilter(0.25,bandwidth=args.bandwidth/4.,filtLen=args.filtlen)

	if vars(args)['1D']:
		if args.prefix == 'boundary_':
			args.prefix = 'strip_'

	ne = args.ne
	omega0 = args.omega0

	if args.plotFilter:
		showBandRejectFilter(0.25,bandwidth=args.bandwidth/4.,filtLen=args.filtLen)

	if not args.noFilter:
		if not args.onlyTotal:
			# Check if there is any filtered data in output directory
			if any([ (files.startswith('filtered') or files.startswith('laser'))
			         and files.endswith('.npz')
			         for files in os.listdir(os.path.join(os.path.join(args.dataDir,_outputDir))) ]):
				raise IOError("Boundary filtered data file already exists")

			filterSimulationBoundaries(args.dataDir,args.omega0,args.bandwidth,
			    args.filtLen,args.prefix,vars(args)['1D'],
			    polarisation=args.polarisation,overwrite=args.overwrite)

		totalFluxes(args.dataDir,ne,vars(args)['1D'],args.polarisation)
	if args.plotReflectivity:
		procDir = os.path.join(args.dataDir,_outputDir)
		reflFile = os.path.join(procDir,'totalRefl.npz')
		lasFile  = os.path.join(procDir,'totalLas.npz')

		if args.calcLaserPower:
			tFlux = np.load(lasFile)
			P0 = empiricalLaserPower(tFlux['time'],tFlux['x_min'],
			                         laserDelay=args.laserRiseTime,
			                         lpiOnset=args.lpiOnset)[0]
			print("Calculated laser power: {:}TW (units??)".format(P0/1e12))
		else:
			if args.I is None:
				raise ValueError("Laser intensity required to plot figure")

			I = args.I*1e4

			if vars(args)['1D']:
				P0 = I
			elif args.F:
				P0 = specklePower(srsUtils.wnVacNIF,args.F,I)
			else:
				P0 = planeWavePower(srsUtils.wnVacNIF,I,
				                    os.path.join(args.dataDir,'regular_0000.sdf'))

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plotReflectivity(ax,reflFile,lasFile,P0,tLims=(args.mint,args.maxt),
		                 toPlot=args.toPlot,noLegend=args.noLegend,
		                 smoothTime=args.smoothTime)
		ax.set_ylim(args.minR,args.maxR)
		fig.set_size_inches(args.figSize)

		if args.output is not None:
			fig.savefig(args.output)
		else:
			plt.show()
