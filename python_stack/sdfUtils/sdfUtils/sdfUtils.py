#!/usr/bin/env python
# coding=UTF-8

# ***************************** Module Imports ****************************** #
import math
import scipy.constants as const
import multiprocessing as mp
import struct
import sys
import os
import datetime
import time
import functools
import sdf
import numpy as np
import re
try:
	import numba
	numbaPresent = True
	#numbaPresent = False
except ImportError:
	numbaPresent = False
	print("WARNING: Couldn't import numba")

# ***************************** Define Functions **************************** #

def terminal_size():
    import fcntl, termios, struct
    h, w, hp, wp = struct.unpack('HHHH',
        fcntl.ioctl(0, termios.TIOCGWINSZ,
        struct.pack('HHHH', 0, 0, 0, 0)))
    return w, h

def startprogress(title,fancy=False):
	"""Creates a progress bar 40 chars long on the console
	and moves cursor back to beginning with BS character"""
	global progressBar
	progressBar = {'title'     :title,
	               'progress'  :0,
	               'startTime' :time.time(),
	               'fancy'     :fancy}

	try:
		termWidth = terminal_size()[0]
	except IOError:
		termWidth = 80

	progressSize = termWidth - (len(progressBar['title'])+1) - 4 - (7+1) - (7+1)

	sys.stdout.write(progressBar['title'] + ": [>" + " " * (progressSize-1) + "] " + '{:6.2f}% '.format(0.00) + str(datetime.timedelta(seconds=int(time.time()-progressBar['startTime'])))+"")
	sys.stdout.flush()

def progress(x):
	"""Sets progress bar to a certain percentage x.
	Progress is given as whole percentage, i.e. 50% done
	is given by x = 50"""

	try:
		termWidth = terminal_size()[0]
	except IOError:
		termWidth = 80

	progressSize = termWidth - (len(progressBar['title'])+1) - 4 - (7+1) - (7+1)

	xInt = int(x * progressSize // 100)
	xFloat = x
	if(progressBar['fancy']):
		sys.stdout.write(chr(8) * termWidth)
	else:
		sys.stdout.write('\n')
	sys.stdout.write(progressBar['title'] + ": [" + "=" * xInt + ">" + " " * (progressSize-1 - xInt) + "] " + '{:6.2f}% '.format(xFloat) + str(datetime.timedelta(seconds=int(time.time()-progressBar['startTime'])))+"")
	sys.stdout.flush()
	progressBar['progress'] = x


def endprogress(x=None):
	"""End of progress bar;
	Write full bar, then move to next line"""
	try:
		termWidth = terminal_size()[0]
	except IOError:
		termWidth = 80

	progressSize = termWidth - (len(progressBar['title'])+1) - 4 - (7+1) - (7+1)

	if(x == None):
		x = progressBar['progress']
	xInt = int(x * progressSize // 100)
	xFloat = x
	if(progressBar['fancy']):
		sys.stdout.write(chr(8) * termWidth)
	else:
		sys.stdout.write('\n')
	sys.stdout.write(progressBar['title'] + ": [" + "=" * min(xInt-1,progressSize) + ">" + " " * (progressSize-1 - xInt) + "] " + '{:6.2f}% '.format(xFloat) + str(datetime.timedelta(seconds=int(time.time()-progressBar['startTime'])))+"\n")
	sys.stdout.flush()

def get_open_fds():
	'''
	return the number of open file descriptors for current process

	.. warning: will only work on UNIX-like os-es.
	'''
	import subprocess
	import os

	pid = os.getpid()
	procs = subprocess.check_output(
		[ "lsof", '-w', '-Fntf', "-p", str( pid ) ] )
	print(procs)
	nprocs = filter(
			lambda s: s and s[ 0 ] == 'n', #and s[1: ].isdigit(),
			procs.split( '\n' ) )
		#)
	return nprocs[101:]

def readMultipleChild(processID,fileList,requiredData,altGrid=None):
	dataList = []
	try:
		dataList = list(map(lambda i:sdf.read(fileList[i],mmap=True),range(len(fileList))))
	except IOError:
		print("Hmmm, IOError...")
		exit()

	data = dict.fromkeys(requiredData)
	if altGrid:
		spaceGrid = dataList[0].__dict__[altGrid].data
	else:
		spaceGrid = dataList[0].Grid_Grid.data
	data['space'] = tuple([ 0.5*(s[:-1] + s[1:]) for s in spaceGrid ])
	data['time'] = np.array([ i.Header['time'] for i in dataList ])
	for d in requiredData:
		dataType = type(dataList[0].__dict__[d])
		firstData = dataList[0].__dict__[d].data
		#if(dataType == sdf.BlockPlainMesh):
		#	data[d] = np.empty((len(data['time']),len(firstData))+firstdata[0].shape)
		if(dataType == sdf.BlockPlainVariable
		or dataType == sdf.BlockPointVariable):
			data[d] = np.empty((len(data['time']),)+firstData.shape)
		elif(dataType == sdf.BlockPointMesh):
			data[d] = np.empty((len(data['time']),len(firstData),firstData[0].shape[0]))
		else:
			raise EnvironmentError("Don't know what to do with this type of data!")
		#print(data[d].shape)
		for i,sdfFile in enumerate(dataList):
			if(dataType in [sdf.BlockPlainVariable,sdf.BlockPlainMesh,sdf.BlockPointVariable]):
				np.copyto(data[d][i],sdfFile.__dict__[d].data)
			elif(dataType == sdf.BlockPointMesh):
				for j in range(len(firstData)):
					np.copyto(data[d][i][j],sdfFile.__dict__[d].data[j])
			else:
				raise EnvironmentError("Don't know what to do with this type of data!")

	np.savez('./_temp_' + str(processID) + '.npz',**data)

def readMultipleShort(fileList,requiredData,altGrid=None):
	'''
	Function for reading less than 1000 sdf files and putting the data in to one array
	'''
	# Get spatial grid
	firstFile = sdf.read(fileList[0])
	if altGrid: # for distribution function, generally
		spaceGrid = firstFile.__dict__[altGrid].data
	else:
		spaceGrid = firstFile.Grid_Grid.data
		spaceGrid = tuple([ 0.5*(s[:-1] + s[1:]) for s in spaceGrid ])

	lenTimes = len(fileList)

	data = dict.fromkeys(requiredData)
	for d in data:
		dataType = type(firstFile.__dict__[d])
		if(dataType == sdf.BlockPlainVariable
		or dataType == sdf.BlockPointVariable):
			lenVar = firstFile.__dict__[d].data.shape
		elif(dataType == sdf.BlockPointMesh):
			lenVar = (len(firstFile.__dict__[d].data),) + firstFile.__dict__[d].data[0].shape
		else:
			raise EnvironmentError("Don't know what to do with this type of data!")

		data[d] = np.zeros((lenTimes,) + lenVar)

	data['time'] = np.zeros(lenTimes)
	data['space'] = spaceGrid

	for i in range(lenTimes):
		tempTime = sdf.read(fileList[i]).Header['time']
		data['time'][i] = tempTime
		del tempTime

		for d in requiredData:
			tempField = sdf.read(fileList[i]).__dict__[d].data
			data[d][i,:] = tempField
			del tempField

	return data

def readMultiple(fileList,requiredData,blockSize=1000,fancyProgress=False,

		         altGrid=None):
	'''
	Function for reading multiple SDF files and munging together the data

	Probably needs rewriting as looks pretty stupid at the moment
	'''
	# Create process pool
	pool = mp.Pool(maxtasksperchild=blockSize)

	# Filter down to files that have data
	filesWithData = pool.map(functools.partial(hasData,requiredData=requiredData),fileList)
	if(any(filesWithData) == False):
		raise IOError("No sdf files found with all of the required data")
	elif(all(filesWithData) == False):
		print("WARNING: "+str(sum(filesWithData))+" files of "+str(len(fileList))+" contain the required data.")
		fileList = [ i for i,j in zip(fileList,filesWithData) if j ]

	lenTimes = len(fileList)

	# Get spatial grid. Field quantities defined on midpoints so calculate these
	firstFile = sdf.read(fileList[0])
	if altGrid: # Switch in case grid is defined somewhere other than Grid_Grid
		spaceGrid = firstFile.__dict__[altGrid].data
	else:
		spaceGrid = firstFile.Grid_Grid.data
	spaceGrid = tuple([ 0.5*(s[:-1] + s[1:]) for s in spaceGrid ])

	numBlocks = len(fileList)//blockSize + 1
	dataBlockList = []

	data = dict.fromkeys(requiredData)
	for d in data:
		dataType = type(firstFile.__dict__[d])
		#if(dataType == sdf.BlockPlainMesh):
		#	lenVar = (len(firstFile.__dict__[d].data),)+firstFile.__dict__[d].data.shape
		if(dataType == sdf.BlockPlainVariable
		or dataType == sdf.BlockPointVariable):
			lenVar = firstFile.__dict__[d].data.shape
		elif(dataType == sdf.BlockPointMesh):
			lenVar = (len(firstFile.__dict__[d].data),) + firstFile.__dict__[d].data[0].shape
		else:
			raise EnvironmentError("Don't know what to do with this type of data!")

		data[d] = np.zeros((lenTimes,)+lenVar)

	data['time'] = np.zeros(lenTimes)
	data['space'] = spaceGrid

	fileSizeMB = sum([ os.path.getsize(f) for f in fileList ])/1024.0**2

	startprogress("Reading files",fancyProgress)

	for i in range(numBlocks):
		if(i == numBlocks-1):
			if(len(fileList) % blockSize != 0):
				fileBlock = fileList[i*blockSize:]
			else: break
		else:
			fileBlock = fileList[i*blockSize:(i+1)*blockSize]

		#print("Block",i+1,"length:",len(fileBlock))
		p = mp.Process(target=readMultipleChild,args=(i,fileBlock,requiredData,altGrid))
		p.start()
		p.join()
		time.sleep(1)

		blockData = np.load('./_temp_' + str(i) + '.npz')
		os.remove('./_temp_' + str(i) + '.npz')
		currBlockSize = len(blockData['time'])

		data['time'][i*blockSize:i*blockSize+currBlockSize] = blockData['time']
		for d in requiredData:
			data[d][i*blockSize:i*blockSize+currBlockSize] = blockData[d]

		if(i+1 != numBlocks):
			progress(float(i+1)/numBlocks*100.0)
		else:
			endprogress(100.0)

	pool.close()
	print(str(fileSizeMB)+" MB @ "+str(fileSizeMB/(time.time()-progressBar['startTime'])) + " MB/s")
	return data

# For backward compatibility with ridiculous old name
getDataFromSDFFilesMaster = readMultiple

def readMultipleAcc(files,requiredData,subsetName,spaceSlices=None,onMissing='pad'):
	'''
	Read in data from multiple accumulated data files

	onMissing: Determines action taken if missing data detected. Options are:
	           'pad': fill missing data region with zeros
			   'error': throw error
	'''
	# Filter down to files that have requested data
	filesWithData = map(functools.partial(hasDataAcc,requiredData=requiredData,subset=subsetName),files)
	if not any(filesWithData):
		raise IOError("No sdf files found with all of the required data")
	elif not all(filesWithData):
		print("WARNING: "+str(sum(filesWithData))+" files of "+str(len(files))+" contain the required data.")
		files = [ i for i,j in zip(files,filesWithData) if j ]

	# Read first file to get spatial grid.
	# Normal spatial grid written alongside accumulator one, so use this
	# if accumulator grid is not present
	firstFile = sdf.read(files[0])
	if 'Grid_A_'+subsetName in firstFile.__dict__:
		spaceGrid = firstFile.__dict__['Grid_A_'+subsetName].data[:-1]
	else:
		spaceGrid = firstFile.__dict__['Grid_'+subsetName].data

	# Field quantities defined on midpoints
	spaceGrid = tuple([ 0.5*(s[:-1] + s[1:]) for s in spaceGrid ])

	if spaceSlices is not None:
		spaceGrid = tuple([ g[s] for g,s in zip(spaceGrid,spaceSlices) ])

	# Find the Grid_A_* variable which contains the time grid. This seems to
	# only be written for one of the subsets being output (presumably the
	# first). Doesn't matter if it corresponds to the current subset or not
	# as all we need are the time values from now on.
	accGridKey = 'Grid_A_'+subsetName
	if accGridKey not in firstFile.__dict__:
		accGridKey = [ k for k in firstFile.__dict__.keys()
		               if k.startswith('Grid_A_') and not k.endswith('_mid') ][0]

	# Number of accumulated snapshots in each file
	NtAcc = numAcc(files[0])

	# Create dictionary to hold data and fill it with empty arrays to be filled
	data = dict.fromkeys(requiredData)
	data['space'] = spaceGrid
	data['time'] = np.zeros((NtAcc*len(files),))
	for r in requiredData:
		data[r] = np.zeros([ data['time'].shape[0] ] + [ s.shape[0] for s in spaceGrid ])

	if spaceSlices is not None:
		slc = spaceSlices + [slice(None)]
	else:
		slc = [slice(None)]

	# Iterate over files and read the data
	for i,f in enumerate(files):
		sdfDict = sdf.read(f).__dict__
		missing = None
		newT  = sdfDict[accGridKey].data[-1]
		if sdfDict[accGridKey].data[-1].shape[0] < NtAcc:
			if onMissing == 'error':
				raise ValueError("Missing data in file {:}".format(f))
			elif onMissing == 'pad':
				# Figure out where missing data is located
				lastT = data['time'][i*NtAcc-1]
				dt    = lastT - data['time'][i*NtAcc-2]
				numMissing = NtAcc-newT.shape[0]

				if (newT[0] - lastT)/dt > 1.5:
					# Missing data is at the beginning
					missing = 'begin'
					newT = np.concatenate([newT[0] - dt*np.arange(1,numMissing+1)[::-1],newT])
				else:
					# Missing data is assumed to be at the end
					missing = 'end'
					newT = np.concatenate([newT,newT[-1] + dt*np.arange(1,numMissing+1)])
				print("Warning: file {:} is missing {:} snapshots. Attempting to pad with zeros.".format(f,numMissing))

			else:
				raise ValueError("parameter onMissing must be either \'pad\' or \'error'")
		elif sdfDict[accGridKey].data[-1].shape[0] > NtAcc:
			numMissing = NtAcc-newT.shape[0]
			newT = newT[:numMissing]
			missing = 'negative'

		data['time'][i*NtAcc:(i+1)*NtAcc] = newT
		for r in requiredData:
			#print(r)
			#print(data[r].shape)
			newData = np.rollaxis(sdfDict[r+'_Acc_'+subsetName].data[slc],-1)

			# Pad with zeros to reach NtAcc accumulated time snapshots if missing
			if missing == 'negative':
				newData = newData[:numMissing]
			else:
				if missing == 'begin':
					newData = np.pad(newData,[(numMissing,0)] + [(0,0)]*(len(newData.shape)-1),mode='constant')
				elif missing == 'end':
					newData = np.pad(newData,[(0,numMissing)] + [(0,0)]*(len(newData.shape)-1),mode='constant')

			try:
				data[r][i*NtAcc:(i+1)*NtAcc] = newData
			except ValueError:
				# This is meant to hackily deal with the case where the spatial
				# dimension of the accumulator block changes mid-simulation by
				# one cell. Ideally fix by correcting the bug in EPOCH that
				# is the cause...
				data[r][i*NtAcc:(i+1)*NtAcc][:,:,:-1] = newData
			#print(data[r].shape)

	return data

def numAcc(sdfFile):
	'''
	Returns the number of accumulated snapshots in a file
	'''
	sdfDict = sdf.read(sdfFile).__dict__

	# Find accumulator grid with time variable
	accGrid = [ g for g in sdfDict.keys() if g.startswith('Grid_A_')
			                              and not g.endswith('_mid') ][0]

	# Time variable is the final array in the grid tuple
	NtAcc = sdfDict[accGrid].data[-1].shape[0]

	return NtAcc

def hasData(sdfFile,requiredData):
	dataFile = sdf.read(sdfFile)
	if(all(i in dataFile.__dict__ for i in requiredData)):
		return True
	else:
		return False

def hasDataAcc(sdfFile,requiredData,subset=''):
	dataFile = sdf.read(sdfFile)
	if(all(any([ k.startswith(d+'_Acc_'+subset) for k in dataFile.__dict__.keys() ])
	                                       for d in requiredData)):
		return True
	else:
		return False

def findSDFFiles(directory):
	files = [ os.path.join(directory,f) for f in os.listdir(directory) if re.match(r'.*\.sdf',f) ]
	files.sort(key=lambda name: int(os.path.splitext(os.path.basename(name))[0]))

	return files

def getSDFMetadata(sdfFile):
	data = {}
	t1 = time.time()
	with open(sdfFile,'rb') as sdfOpen:
		#print('open took {:}'.format(time.time()-t1))

		t1 = time.time()
		header = sdfOpen.read(106) # <- This takes ages
		#header = sdfOpen.read(1000)
		#print('header read took {:}'.format(time.time()-t1))

		headerSpec = (('sdf',0,'4s',4),
				      ('endianness',   4,  'i',  4),
					  ('sdf_version',  8,  'i',  4),
					  ('sdf_revision', 12, 'i',  4),
					  ('code_name',    16, '32s',32),
					  ('step',         76, 'i',  4),
					  ('time',         80, 'd',  8),
					  ('jobid1',       88, 'i',  4),
					  ('jobid2',       92, 'i',  4),
					  ('string_length',96, 'i',  4),
					  ('code_io_version',100,'i',4),
					  ('restart_flag', 104,'c',  1),
					  ('subdomain_file',105,'c', 1))


		for spec in headerSpec:
			#print(spec)
			name   = spec[0]
			offset = spec[1]
			dType  = spec[2]
			length = spec[3]
			#print(dType)
			#print(struct.unpack(dType,header[offset:offset+length]))
			data[name] = struct.unpack(dType,header[offset:offset+length])[0]

		data['code_name'] = data['code_name'].split(b'\0', 1)[0]
		data['restart_flag'] = bool(ord(data['restart_flag']))

		nextBlockLoc = struct.unpack('=Q',header[48:48+8])[0]
		numBlocks = struct.unpack('=L',header[68:68+4])[0]
		strLen = struct.unpack('=L',header[96:96+4])[0]

		#sdfOpen.seek(0,2)
		#eof = sdfOpen.tell()

		#print('First block header at {:}'.format(nextBlockLoc))
		blockNum = 0
		seekTimes = []
		readTimes = []
		data['blockNames'] = []
		while(blockNum < numBlocks):
		#while(nextBlockLoc != eof and blockNum < numBlocks):
			# Record current block header location
			loc = nextBlockLoc

			# Move to start of block header
			t1 = time.time()
			sdfOpen.seek(loc)
			seekTime = time.time()-t1
			seekTimes.append(seekTime)
			#print('Seek 1: {:}'.format(seekTime))

			t1 = time.time()
			blockHeader = sdfOpen.read(72+strLen)
			readTime = time.time() - t1
			readTimes.append(readTime)

			# Read location of next block header
			nextBlockLoc = struct.unpack('=Q',blockHeader[:8])[0]
			#print('Next block header at {:}'.format(nextBlockLoc))

			# Read block ID
			#sdfOpen.seek(loc+16)
			#blockID = sdfOpen.read(32).decode('utf-8')
			#print('blockID: '+blockID)

			# Read block name
			blockName = str(blockHeader[68:68+strLen].decode('ascii')).strip()
			for char in [' ','/','(',')']: blockName = blockName.replace(char,'_')
			blockName = blockName.replace('\x00','')
			#print('block name: '+blockName)
			data['blockNames'].append(blockName)

			blockNum += 1
			#time.sleep(1)
		#print('Total seek time: {:}'.format(sum(seekTimes)))
		#print('Total read time: {:}'.format(sum(readTimes)))

	return data

def getManySDFMetadata(sdfFiles):
	import threading
	from multiprocessing.pool import ThreadPool

	pool = ThreadPool()
	metaData = pool.map(getSDFMetadata,sdfFiles)
	pool.close()

	return metaData

def getSDFTime(sdfFile):
	t1 = time.time()
	with open(sdfFile,'rb') as sdfOpen:
		header = sdfOpen.read(106)
		t = struct.unpack('d',header[80:80+8])[0]

	#print('getTime time: {:}'.format(time.time()-t1))
	return t

def getSDFTimeAcc(sdfFile):
	sdfObj = sdf.read(sdfFile)
	gridKey = [ k for k in sdfObj.__dict__.keys()
	            if k.startswith('Grid_A_')
	            and not k.endswith('_mid') ][0]

	return sdfObj.__dict__[gridKey].data[-1]

def getSDFBlockNames(sdfFile):
	blockNames = []
	ts = time.time()
	with open(sdfFile,'rb') as sdfOpen:
		header = sdfOpen.read(106)
		nextBlockLoc = struct.unpack('=Q',header[48:48+8])[0]
		numBlocks = struct.unpack('=L',header[68:68+4])[0]
		strLen = struct.unpack('=L',header[96:96+4])[0]

		#sdfOpen.seek(0,2)
		#eof = sdfOpen.tell()

		#print('First block header at {:}'.format(nextBlockLoc))
		blockNum = 0
		seekTimes = []
		while(blockNum < numBlocks):
		#while(nextBlockLoc != eof and blockNum < numBlocks):
			# Record current block header location
			loc = nextBlockLoc

			# Move to start of block header
			t1 = time.time()
			sdfOpen.seek(loc)
			seekTime = time.time()-t1
			seekTimes.append(seekTime)
			#print('Seek 1: {:}'.format(seekTime))

			blockHeader = sdfOpen.read(72+strLen)

			# Read location of next block header
			nextBlockLoc = struct.unpack('=Q',blockHeader[:8])[0]
			#print('Next block header at {:}'.format(nextBlockLoc))

			# Read block ID
			#sdfOpen.seek(loc+16)
			#blockID = sdfOpen.read(32).decode('utf-8')
			#print('blockID: '+blockID)

			# Read block name
			blockName = str(blockHeader[68:68+strLen].decode('ascii')).strip()
			for char in [' ','/','(',')']: blockName = blockName.replace(char,'_')
			blockName = blockName.replace('\x00','')
			#print('block name: '+blockName)
			blockNames.append(blockName)

			blockNum += 1
			#time.sleep(1)
		#print('Total seek time: {:}'.format(sum(seekTimes)))
		tend = time.time()

	#print('closeTime: {:}'.format(time.time()-tend))

	funcTime = time.time()-ts
	#print('functime: {:}'.format(funcTime))
	return blockNames


def getPrefix(f):
	''' Extracts the SDF file prefix from its file name '''
	return os.path.basename(f).split('.')[0].rstrip('0123456789')

def getNum(f,prefix=''):
	''' Extracts the SDF file number from its name '''
	return int(os.path.basename(f).replace(prefix,'').split('.')[0])

def listFiles(dataDir,prefix=None):
	''' Finds all SDF files in the specified directory with a given prefix '''
	if prefix is not None:
		sdfFiles = [ os.path.join(dataDir,f) for f in os.listdir(dataDir) if re.match(prefix+r'[0-9]*\.sdf',f) ]
		sdfFiles.sort(key=lambda name: getNum(name,prefix))
	else:
		sdfFiles = [ os.path.join(dataDir,f) for f in os.listdir(dataDir) if re.match(r'.*[0-9]*\.sdf',f) ]
		sdfFiles.sort()#key=lambda name: int(os.path.basename(name).split('.')[0]))

	return sdfFiles

def particleWeightFunc(x,order=1):
	x = abs(x)
	if(order == 0):
		if(x < 1.0):
			return 1.0-x
		else:
			return 0.0
	elif(order == 1):
		if(x < 0.5):
			return 0.75-x**2
		elif(x < 1.5):
			return 0.5*(x-1.5)**2
		else:
			return 0.0
	elif(order == -1):
		if(x < 0.5):
			return 1.0
		else:
			return 0.0

if(numbaPresent): particleWeightFunc = numba.jit(particleWeightFunc,nopython=True)

# Calculate the density profile by mapping particle shape functions onto the grid
def calcDensity(grid,positions,weight,periodic=False,order=1):
	dx = grid[1]-grid[0]
	densityGrid = np.zeros(len(grid)-1)

	for p in positions:
		# Cell number particle resides in
		cellNum = int(p // dx)
		# Normalised position of cell center relative to particle center
		cellPos = 0.5 - (p%dx)/dx

		if(periodic or cellNum != 0):
			densityGrid[cellNum-1] += weight*particleWeightFunc(cellPos-1.0,order)

		densityGrid[cellNum] += weight*particleWeightFunc(cellPos,order)

		if(cellNum != len(densityGrid)-1):
			densityGrid[cellNum+1] += weight*particleWeightFunc(cellPos+1.0,order)
		elif(periodic):
			densityGrid[0] += weight*particleWeightFunc(cellPos+1.0,order)

	return densityGrid/dx

if(numbaPresent): calcDensity = numba.jit(calcDensity,nopython=True)

def calcElectricField(grid,positions,weight,periodic=False,order=1):
	dx = grid[1]-grid[0]
	dens = calcDensity(grid,positions,weight,periodic,order)
	chargeDens = -const.e/const.epsilon_0*(dens - np.mean(dens))

	ex = np.zeros(len(chargeDens))

	for i in range(1,len(chargeDens)-1):
		ex[i] = ex[i-1] + (chargeDens[i])*dx

	return ex

if(numbaPresent): calcElectricField = numba.jit(calcElectricField,nopython=True)

