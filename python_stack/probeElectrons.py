#!/usr/bin/env python
# coding=UTF-8

import datetime
import math
import numpy as np
import os
import pathos.multiprocessing as mp
import re
import scipy.constants as const
import time

import sdf
import sdfUtils

_boundaryNames = ['x_min','x_max','y_min','y_max','fakeProbe']
_outputDir = 'processed'

def particleKineticEnergy(px,py,pz):
	gamma = np.sqrt(1.+ (px**2 + py**2 + pz**2)/(const.m_e*const.c)**2)
	ke = const.m_e*const.c**2*(gamma - 1.0)

	return ke

def particleVelocity(px,py,pz):
	gamma = np.sqrt(1.+ (px**2 + py**2 + pz**2)/(const.m_e*const.c)**2)
	vx = px/(const.m_e*gamma)
	vy = py/(const.m_e*gamma)
	vz = pz/(const.m_e*gamma)

	return vx,vy,vz

def particleSpeed(px,py,pz):
	gamma = np.sqrt(1.+ (px**2 + py**2 + pz**2)/(const.m_e*const.c)**2)
	vx = px/(const.m_e*gamma)
	vy = py/(const.m_e*gamma)
	vz = pz/(const.m_e*gamma)

	return np.sqrt(vx**2 +vy**2 + vz**2)

def getMaxEnergy(fileNames):
	return None

if __name__ == '__main__':
	import argparse

	# Handle command line arguments:
	###########################################################################
	parser = argparse.ArgumentParser()

	# Arguments specifying location of data
	parser.add_argument('dataDir')
	parser.add_argument('-p','--prefix',default='regular_')
	parser.add_argument('-g','--gridFile')

	# Options controlling range of times used
	parser.add_argument('-fn','--firstFileNum',type=int)
	parser.add_argument('-ln','--lastFileNum',type=int)
	parser.add_argument('-ft','--firstTime',type=float)
	parser.add_argument('-lt','--lastTime',type=float)

	# Options controlling data processed
	parser.add_argument('--boundary',required=True)
	parser.add_argument('--boundaryType',choices=['min','max'])
	parser.add_argument('--combineY')
	# Which way to bin 'energy' - bin by velocity perpendicular to boundary,
	# velocity magnitude, KE perpendicular to boundary or total KE.
	parser.add_argument('--binQuantity',choices=['perpVel','speed','KE'],default='total')

	# Options to alter data binning. By default only bin w.r.t. energy/velocity.
	parser.add_argument('-ne','--numValBins',type=int,default=100)
	parser.add_argument('--minVal',type=float)
	parser.add_argument('--maxVal',type=float)

	parser.add_argument('-bt','--binTime',action='store_true')

	parser.add_argument('-bs','--binSpace',action='store_true')
	parser.add_argument('-ns','--numSpaceBins',type=int)
	parser.add_argument('--minPos',type=float)
	parser.add_argument('--maxPos',type=float)

	# Miscellaneous arguments
	parser.add_argument('--maxMemGiB',type=float,default=4.0)

	# Do the argument parsing
	args = parser.parse_args()

	binTime = (args.binTime == True)
	binSpace = (args.binSpace or args.numSpaceBins or args.minPos or args.maxPos)

	# Assign function for calculating 'energy' bin quantity
	if args.binQuantity == 'perpVel':
		if args.boundary.endswith('min') or args.boundaryType == 'min':
			fac = -1
		else:
			fac = 1
		if args.boundary[0] == 'x':
			valFunc = lambda px,py,pz: fac*particleVelocity(px,py,pz)[0]
		else:
			valFunc = lambda px,py,pz: fac*particleVelocity(px,py,pz)[1]
	elif args.binQuantity == 'speed':
		valFunc = particleSpeed
	else:
		valFunc = particleKineticEnergy

	# Get list of requested data files:
	###########################################################################

	# Get sorted list of SDF files in directory
	if args.boundary == 'fakeProbe':
		files = [ os.path.join(args.dataDir,f)
		          for f in os.listdir(args.dataDir)
		          if re.match(args.prefix+r'[0-9]*\.npz',f) ]
		files.sort(key=lambda name: sdfUtils.getNum(name,args.prefix))
	else:
		files = sdfUtils.listFiles(args.dataDir,args.prefix)
	#print(files)
	if not files:
		raise IOError("Couldn't find any SDF files with the specified prefix")
	print("Found {:} probe sdf files".format(len(files)))

	# Narrow down to range of file numbers requested
	if args.firstFileNum or args.lastFileNum:
		fileNums = [ sdfUtils.getNum(f,args.prefix) for f in files ]
		if args.firstFileNum:
			files = [ f for f,n in zip(files,fileNums) if n >= args.firstFileNum ]
			fileNums = [ sdfUtils.getNum(f,args.prefix) for f in files ]

		if args.lastFileNum:
			files = [ f for f,n in zip(files,fileNums) if n <= args.lastFileNum ]

		if not files:
			raise IOError("Couldn't find any SDF files with specified range of file indices")
		print("{:} SDF files in file index range".format(len(files)))

	# Narrow down to range of times requested
	if args.firstTime or args.lastTime:
		times = [ m['time'] for m in sdfUtils.getManySDFMetadata(files) ]
		if args.firstTime:
			files = [ f for f,t in zip(files,times) if t >= args.firstTime ]
			times = [ m['time'] for m in sdfUtils.getManySDFMetadata(files) ]

		if args.lastTime:
			files = [ f for f,t in zip(files,times) if t <= args.lastTime ]

		if not files:
			raise IOError("Couldn't find any SDF files with specified range of times")
		print("{:} SDF files in time range".format(len(files)))

	# Grab cell size, simulation size, time step etc.
	###########################################################################
	if args.gridFile:
		grid = sdf.read(os.path.join(args.dataDir,args.gridFile)).Grid_Grid.data
	else:
		grid = sdf.read(files[0]).Grid_Grid.data
	dx = grid[0][1]-grid[0][0]
	dy = grid[1][1]-grid[1][0]
	Lx = grid[0][-1]-grid[0][0]
	Ly = grid[1][-1]-grid[1][0]

	if args.boundary == 'fakeProbe':
		times = [ float(np.load(f)['t']) for f in files ]
	else:
		times = [m['time'] for m in sdfUtils.getManySDFMetadata(files)]
	dt = times[1]-times[0]
	Lt = dt*len(files)

	# Figure out binning ranges if we don't already know them
	###########################################################################

	# Energy/velocity range
	if not args.minVal or not args.maxVal:
		maxVal = None
		minVal = None
		# TODO: Likely to be horribly slow on HPCs, speed it up...
		for f in files:
			print(f)
			if args.boundary == 'fakeProbe':
				data = np.load(f)

				px = data['px']
				py = data['py']
				pz = data['pz']
			else:
				data = sdf.read(f)

				if args.boundary+'_Px' not in data.__dict__:
					print("Warning: no data in file {:}".format(f))
					continue

				px = data.__dict__[args.boundary+'_Px'].data
				py = data.__dict__[args.boundary+'_Py'].data
				pz = data.__dict__[args.boundary+'_Pz'].data

			if len(px) == 0:
				continue

			vals = valFunc(px,py,pz)
			if args.maxVal is None and (maxVal is None or max(vals) > maxVal):
				maxVal = max(vals)
			if args.minVal is None and (minVal == None or min(vals) < minVal):
				minVal = min(vals)

		if not args.maxVal:
			args.maxVal = maxVal
		if not args.minVal:
			args.minVal = minVal

	print("Energy/velocity range:")
	print("({:}, {:})".format(args.minVal,args.maxVal))

	# Temporal binning
	if binTime:
		minTime = times[0]-0.5*dt
		maxTime = times[-1]+0.5*dt

	print("Temporal range:")
	print("({:}, {:})".format(minTime,maxTime))

	# Spatial range (if requested)
	if binSpace:
		if args.boundary[0] == 'x' or args.boundary[0] == 'fakeProbe':
			grid = grid[1]
		else:
			grid = grid[0]

		if not args.minPos:
			args.minPos = grid[0]

		if not args.maxPos:
			args.maxPos = grid[-1]

		if not args.numSpaceBins:
			args.numSpaceBins = len(grid)

	print("Spatial range:")
	print("({:}, {:})".format(args.minPos,args.maxPos))

	# Split list of files up into chunks of size less than maximum allowed
	###########################################################################
	chunks = []
	chunk = []
	chunkTotal = 0.
	for f in files:
		fSize = os.path.getsize(f)/1024.**3
		if fSize > args.maxMemGiB:
			raise MemoryError("SDF file larger than maximum allowed memory")

		if chunkTotal + fSize < args.maxMemGiB:
			chunk.append(f)
			chunkTotal += fSize
		else:
			chunks.append(chunk)
			chunk = [f]
			chunkTotal = fSize
	chunks.append(chunk)

	print(chunks)

	# Run through files and count up numbers of particles in our bins
	###########################################################################
	histShape = [args.numValBins]
	if binSpace:
		histShape.append(args.numSpaceBins)
	if binTime:
		histShape.append(len(files))

	histRanges = [(args.minVal,args.maxVal)]
	if binSpace:
		histRanges.append((args.minPos,args.maxPos))
	if binTime:
		histRanges.append((minTime,maxTime))

	histogram = np.zeros(histShape)
	for i,chunk in enumerate(chunks):
		timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
		print(timestamp+': Reading chunk {:} of {:}'.format(i+1,len(chunks)))
		for f in chunk:
			# Read data
			if args.boundary == 'fakeProbe':
				data = np.load(f)
				ws = data['ws']
				px = data['px']
				py = data['py']
				pz = data['pz']

				vals = valFunc(px,py,pz)

				if binSpace:
					if args.boundary[0] == 'x':
						pos = data['ys']
					else:
						pos = data['xs']

				if binTime:
					time = np.ones(vals.shape)*float(data['t'])
			else:
				data = sdf.read(f)
				if args.boundary+'_weight' not in data.__dict__.keys():
					print("Warning: Couldn't find data in file {:}".format(f))
					continue
				ws = data.__dict__[args.boundary+'_weight'].data
				px = data.__dict__[args.boundary+'_Px'].data
				py = data.__dict__[args.boundary+'_Py'].data
				pz = data.__dict__[args.boundary+'_Pz'].data

				vals = valFunc(px,py,pz)

				if binSpace:
					if args.boundary[0] == 'x':
						pos = data.__dict__['Grid_Probe_'+args.boundary].data[1]
					else:
						pos = data.__dict__['Grid_Probe_'+args.boundary].data[0]

				if binTime:
					time = np.ones(vals.shape)*data.Header['time']

			sample = [vals]
			if binSpace: sample.append(pos)
			if binTime : sample.append(time)

			H,edges = np.histogramdd(sample,bins=histShape,range=histRanges,
			                         weights=ws,normed=False)
			histogram += H

	outDir  = os.path.join(args.dataDir,_outputDir)
	if not os.path.exists(outDir):
		os.makedirs(outDir)

	outFile = os.path.join(outDir,'probeData_{:}.npz'.format(args.boundary))
	if os.path.exists(outFile):
		raise IOError("Probe data file already exists")

	np.savez_compressed(outFile,hist=histogram,ranges=histRanges,dx=dx,dy=dy,Lx=Lx,Ly=Ly,Lt=Lt)
