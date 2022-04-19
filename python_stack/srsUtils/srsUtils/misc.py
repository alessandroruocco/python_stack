#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import scipy.stats
import functools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import skimage.measure
import scipy.constants as const

from srsUtils import srsUtils

def sizeof_fmt(num, suffix='B',separate=False):
	'''
	Converts a number of bytes into a readable expression

	From https://stackoverflow.com/a/1094933/2622765

	Parameters
	----------

	num : Number of bytes/bits
	suffix : Unit of measure, default 'B' for bytes. E.g. 'b' for bits
	separate : Return a tuple containing the number followed by the unit rather
	           than a single string
	'''
	foundUnit = False
	for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
		if abs(num) < 1024.0:
			sizeTup = ('{:3.1f}'.format(num),'{:}{:}'.format(unit, suffix))
			foundUnit = True
			break
		num /= 1024.0
	if not foundUnit: sizeTup = ('{:.1f}'.format(num),'{:}{:}'.format('Yi',suffix))

	if separate:
		return sizeTup
	else:
		return sizeTup[0]+sizeTup[1]

# Return index of the item in an array closest to a given value
def closestMatchLoc(array,val):
	diffArray = np.abs(array-val)
	closest = np.where(diffArray == np.min(diffArray))
	if(len(closest[0]) != 1):
		raise EnvironmentError("No unique closest match")

	return closest[0]

def getExtent(x,y):
	dx = scipy.stats.mode(np.diff(x))[0][0]
	dy = scipy.stats.mode(np.diff(y))[0][0]

	return [x[0]-0.5*dx,x[-1]+0.5*dx,
	        y[0]-0.5*dy,y[-1]+0.5*dy]

def overlap(a,b,inclusive=False):
	'''
	Returns the overlap between two ranges a and b

	https://stackoverflow.com/a/2953979/2622765

	Parameters
	----------

	a,b : array-like containing range of form (min,max)
	inclusive : Optional, specify whether the ranges are inclusive of the end value
	'''
	if inclusive:
		return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
	else:
		return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def floatToLatexScientific(num):
	return '{:.2e}'.format(num).replace('e',r'\times 10^{').replace('+','')+'}'

def printCols(data):
	col_width = max(len(str(word).decode("UTF-8")) for row in data for word in row) + 2  # padding
	for row in data:
		print("".join(str(word).decode("UTF-8").ljust(col_width) for word in row))

def div0(a, b, c):
	"""
	Handle division by zero in a/b by setting result to c

	Adapted from http://stackoverflow.com/a/35696047
	"""
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide( a, b )
		c[ ~ np.isfinite( c )] = result  # -inf inf NaN

	return c

def vectorize(f):
	'''
		Wrapper to np.vectorize to fix behaviour on scalar input

		From here: http://stackoverflow.com/q/39272465/2622765
	'''
	func = np.vectorize(f)
	@functools.wraps(f)
	def wrapper(*args,**kwargs):
		return func(*args,**kwargs)[()]

	return wrapper

def plotComplexFunc(axes,Z,x,y,xLabel=None,yLabel=None,funcName=None,vmax=None,
                    norm=None):
	'''
	Plots a function f(x+iy) as heat maps of Re(f), Im(f) and |f| respectively

	Parameters
	----------

	axes : Container of three matplotlib axis objects, operated on in-place
	Z    : 2D numpy array of complex values
	x,y  : 1D arrays defining x and y grid values
	xLabel, yLabel: x and y axis labels, if containing LaTeX maths expressions
	       these must be contained in '$'s
	funcName : Name of function, goes inside LaTeX maths expression
	vmax : maximum value for colour map (used for vmin: vmin = -vmax)
	norm : color scale normalisation (default is a SymLogNorm)
	'''
	Zs = [ np.real(Z),
	       np.imag(Z),
	       np.abs(Z)   ]

	if not xLabel: xLabel = '$x$'
	if not yLabel: yLabel = '$y$'
	if not funcName: funcName = 'f'
	titles = [ r'$\mathrm{{Re}}\left({f}\right)$'.format(f=funcName),
	           r'$\mathrm{{Im}}\left({f}\right)$'.format(f=funcName),
			   r'$\left|{f}\right|$'.format(f=funcName)               ]

	if not vmax: vmax = np.max(Zs[2])
	vmin = np.min(Zs[2])
	if not norm: norm=colors.SymLogNorm(linthresh=0.01,linscale=1,vmin=-vmax,vmax=vmax)

	X,Y = np.meshgrid(x,y)
	extent = getExtent(x,y)

	for i,ax in enumerate(axes):
		ax.set_title(titles[i])
		if i != 2:
			cs  = ax.contour(X,Y,Zs[0],levels=[0])
			cs1 = ax.contour(X,Y,Zs[1],levels=[0],linestyles='dashed')
		ax.imshow(Zs[i],origin='lower',extent=extent,norm=norm,cmap='RdBu_r',
				  interpolation='none')
		#ax[0].plot(result['x'][0],result['x'][1],'bx')
		ax.set_xlabel(xLabel)
		ax.set_ylabel(yLabel)

# TODO: Doesn't seem to work properly for some cases?
def genBinFromSympy(expr,args,outDir):
	'''
	Wrapper to sympy autowrap to enable complex calculations

	Description
	-----------

	Uses autowrap to generate fortran code for an expression, and then replaces
	all REAL variable declarations with COMPLEX ones and re-compiles using f2py

	Parameters
	----------

	These are simply passed to autowrap. See autowrap for reference.
	'''
	import os
	import imp
	from sympy.utilities import autowrap
	import subprocess

	os.makedirs(outDir)
	codeWrapper = autowrap.CodeWrapper
	counter = codeWrapper._module_counter
	try:
		autowrap.autowrap(expr,args=args,backend='f2py',tempdir=outDir)
	except:
		pass
	names = { 'h':os.path.join(outDir,'wrapped_code_{c}.h'.format(c=counter)),
			  'f90':os.path.join(outDir,'wrapped_code_{c}.f90'.format(c=counter)) }

	files = {}
	for ext,name in names.iteritems():
		with open(name,'r') as nameFile:
			lines = nameFile.readlines()
			files[ext] = lines

	for ext,lines in files.iteritems():
		lines = [ l.replace('REAL*8','COMPLEX*16') for l in lines ]
		files[ext] = lines

	for ext,lines in files.iteritems():
		os.remove(names[ext])
		with open(names[ext],'w') as nameFile:
			nameFile.writelines(lines)

	os.remove(os.path.join(outDir,'wrapper_module_{c}.so'.format(c=counter)))

	process = subprocess.Popen(['f2py','-c',os.path.basename(names['f90']),'-m','wrapper_module_{:}'.format(counter)],cwd=outDir)
	process.wait()
	if process.returncode != 0:
		raise EnvironmentError("f2py failed to generate *.so file")

	wrapper_module = imp.load_dynamic('wrapper_module_{:}'.format(counter),os.path.join(outDir,'wrapper_module_{:}.so'.format(counter)))

	return wrapper_module.autofunc

def maxLenContigSubseqIdxs(seq):
	'''
	Finds the longest contiguous subsequence in a boolean array

	Shamelessly stolen from:
	https://stackoverflow.com/a/21690865
	'''
	i = thisLen = maxLen = 0
	startIdx, endIdx = 0, 1
	for j in xrange(len(seq)):
		if seq[j]:
			thisLen += 1
		else:
			if thisLen > maxLen:
				maxLen = thisLen
				startIdx = i
				endIdx = j
			thisLen = 0
			i = j + 1
		#print('i: {:}, j: {:}, thisLen: {:}, maxLen: {:}'.format(i,j,thisLen,maxLen))

	if thisLen > maxLen:
		maxLen = thisLen
		startIdx = i
		endIdx = j+1
	return (maxLen, startIdx, endIdx)

def cropImage(data,x,y,xLims,yLims):
	# Choose range of x based on xLims
	xMinInd = 0
	xMaxInd = len(x)
	if xLims is not None:
		ltInds = np.where(x < xLims[0])[0]
		if len(ltInds) != 0:
			xMinInd = ltInds[-1]

		gtInds = np.where(x > xLims[1])[0]
		if len(gtInds) != 0:
			xMaxInd = gtInds[0]+1

	# Choose range of y based on yLims
	yMinInd = 0
	yMaxInd = len(y)
	if yLims is not None:
		ltInds = np.where(y < yLims[0])[0]
		if len(ltInds) != 0:
			yMinInd = ltInds[-1]

		gtInds = np.where(y > yLims[1])[0]
		if len(gtInds) != 0:
			yMaxInd = gtInds[0]+1

	return data[yMinInd:yMaxInd,xMinInd:xMaxInd],x[xMinInd:xMaxInd],y[yMinInd:yMaxInd]

def downsampleImage(data,newShape,method=np.mean):
	reduceArray = tuple([ max([1,o // n]) for o,n in zip(data.shape,newShape) ])
	dataDSed = skimage.measure.block_reduce(data,reduceArray,method)

	return dataDSed

def addNeScale(ax,ne_ax,x,ne,minVal=None,interval=None,tickColor='w',minor=False):
	'''
	Adds a density scale along the upper `x' axis of a figure

	Currently can't figure out a sensible number of tick labels to use, so
	minVal and interval can be used to manually choose these.

	Works by pretending to use the same length numbers as main x-axis, but
	makes the labels at those locations correspond to the density from density
	profile.

	Parameters
	==========

	ax: Main figure axis object
	ne_ax: axis object for new axis (created using ax.twiny())
	x: Grid corresponding to ne, assumed normalised to units used in figure
	ne: 1D density profile in SI units
	minVal: Minimum density tick label (normalised to nCr)
	interval: Interval to use for tick labels, also normalised
	'''
	xLim = ax.get_xlim()

	neNorm = ne/srsUtils.nCritNIF

	# If not given, calculate interval between major ticks
	# TODO: Make this take into account figure & font size
	if interval is None:
		neItvl = 10**(int(math.floor(math.log10(neNorm[-1]-neNorm[0])))-1)
	else:
		neItvl = interval

	# If not given calculate location of first major tick
	if minVal is None:
		tmpItvl    = 10**(int(math.floor(math.log10(neNorm[-1]-neNorm[0])))-1)
		minTickVal = int(math.ceil(neNorm[0]/tmpItvl))*tmpItvl
	else:
		minTickVal = minVal

	# Calculate location of last major tick
	numTicks = int(math.floor((neNorm[-1] - minTickVal)/neItvl)) + 1

	# Create list of major tick values
	xTickVals = minTickVal + np.arange(numTicks)*neItvl

	# Repeat calculations for minor ticks
	neItvlMinor = 0.01 #10**(int(math.floor(math.log10(neNorm[-1]-neNorm[0])))-1)
	minIntMinor = int(math.ceil(neNorm[0]/neItvlMinor))
	maxIntMinor = int(math.floor(neNorm[-1]/neItvlMinor)) + 1

	xTickValsMinor = np.arange(minIntMinor,maxIntMinor)*neItvlMinor

	# Set up major & minor tick parameters
	ne_ax.tick_params(reset=True,labelbottom=False,labeltop=True,bottom=False,
	                  top=True,zorder=10,color=tickColor,direction='in')
	if minor:
		ne_ax.tick_params(which='minor',bottom=False,direction='in',zorder=10,color=tickColor)

	# Generate density labels and calculate their locations on the x-axis
	xTickLabels = [ r'${:.2f}$'.format(v) for v in xTickVals ]
	xTickXLocs      = np.array([ x[np.where(np.abs(neNorm-v) == np.min(np.abs(neNorm-v)))] for v in xTickVals ])
	xTickXLocsMinor = np.array([ x[np.where(np.abs(neNorm-v) == np.min(np.abs(neNorm-v)))] for v in xTickValsMinor ])

	# Add ticks + tick labels to plot
	ne_ax.set_xticks(xTickXLocs)
	if minor: ne_ax.set_xticks(xTickXLocsMinor,minor=True)
	ne_ax.set_xticklabels(xTickLabels)

	# Label axis and ensure the x limits remain the same as the main axis
	ne_ax.set_xlabel(r'$n_e/n_{\mathrm{cr}}$')
	ne_ax.set_xlim(xLim)
