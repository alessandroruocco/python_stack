#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import pathos.multiprocessing as mp
import skimage.measure
from scipy.ndimage.filters import gaussian_filter1d
import os
import sdf

import srsUtils
import sdfUtils

mpl.style.use('classic')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('figure', autolayout=True)

def findDensity(x,ne,frac):
	xMid = 0.5*(x[:-1] + x[1:])

	if np.any(np.where(ne/srsUtils.nCritNIF >= frac)) \
	  and np.any(np.where(ne/srsUtils.nCritNIF <= frac)):
		xQC = xMid[np.min(np.where(ne/srsUtils.nCritNIF >= frac))]
	else:
		xQC = None

	return xQC

def meanEnergyVsTime(ax,files,field,xLims=None,log=False):
	data = [ sdf.read(f) for f in files ]
	time = np.array([ d.Header['time'] for d in data ])
	energy = np.array([ np.mean(d.__dict__[field].data**2) for d in data ])
	norm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
	#norm = 0.5*const.epsilon_0
	energy = energy/norm

	ax.plot(time/1e-12,energy)

	if xLims:
		ax.set_xlim(xLims)
	if log:
		ax.set_yscale('log')
	ax.set_xlabel('time /ps')
	ax.set_ylabel(r'$\frac{e^2\left\langle |E_x|^2 \right\rangle}{(m_e\omega_0c)^2}$')
	ax.grid()

def calcMeanEnergyVsSpaceTime(files,field,parallel=False):
	data = [ sdf.read(f) for f in files ]
	ts = np.array([ d.Header['time'] for d in data ])

	grid = data[0].Grid_Grid.data
	if len(grid) == 1:
		# We have a 1D dataset, no need to average over y
		energy = np.array([ d.__dict__[field].data**2 for d in data ])
		print('max val: {:}'.format(np.max(energy)))
	elif len(grid) == 2:
		# We have a 2D dataset, need to average over y
		y = data[0].Grid_Grid.data[1]

		if parallel:
			def aveFunc(f):
				data = sdf.read(f).__dict__[field].data
				energy = np.mean(data**2,axis=1)
				return energy

			pool = mp.Pool()
			result = pool.map(aveFunc,files)
			pool.close()
			pool.join()
			energy = np.array(result)
		else:
			energy = np.array([ np.mean(d.__dict__[field].data**2,axis=1) for d in data ])
	x = data[0].Grid_Grid.data[0]

	return energy,x,ts

def plotMeanEnergyVsSpaceTime(fig,ax,energy,x,t,field,smoothLen=None,ne=None,
                              xLims=None,yLims=None,maxF=None,minF=None,
                              minFPercentile=0.5,maxFPercentile=99.9,
                              log=False,minNeTick=None,neTickInterval=None,
                              CBar=True,CBarLabel=True,Te=None,markQC=True,
                              markTPDCutoff=True,markSRSCutoff=True,
                              landauCutoff=0.3):
	extent = srsUtils.misc.getExtent(x/1e-6,t/1e-12)

	if field.startswith('Magnetic_') and field.endswith('Bz'):
		smoothLen = 5*srsUtils.wlVacNIF
		dx = x[1]-x[0]
		energy = gaussian_filter1d(energy,sigma=smoothLen/dx)

		densityRange = np.array([0.194580059689,0.260175683371])*srsUtils.nCritNIF
		xMid = 0.5*(x[1:] + x[:-1])
		Ln = xMid/math.log(densityRange[1]/densityRange[0])
		ne = densityRange[0]*np.exp(xMid/Ln)
		op = np.sqrt(ne/(const.m_e*const.epsilon_0))*const.e
		kL = np.sqrt(srsUtils.omegaNIF**2 - op**2)/const.c
		vPhL = srsUtils.omegaNIF/kL

		eNorm = (const.m_e*srsUtils.omegaNIF/const.e)**2*(const.c/vPhL)**2
		w0 = srsUtils.speckle.speckleWidth(6.7,srsUtils.wlVacNIF)
		x0 = srsUtils.speckle.gaussianBeamLengthAtAmp(0.0,math.sqrt(0.5),srsUtils.wnVacNIF,w0)
		E0 = srsUtils.intensityToEField(3.75e15*1e4)*srsUtils.speckle.gaussianBeamAmplitude(x0,y,srsUtils.wnVacNIF,w0)
		E0 = 0.5*np.mean(E0**2)/(const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
		print(E0)
		levels = np.array([0.25,0.5,0.75,1.0])*E0
		ax.contour(energy,levels,extent=extent,origin='lower',linewidths=0.5,colors=['r','y','g','k'])
	elif field.startswith('Electric_') and field.endswith('Ex'):
		eNorm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
	else:
		eNorm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
	#norm =0.5*const.epsilon_0
	energy = energy/eNorm

	if not maxF:
		maxF = np.percentile(energy,maxFPercentile)
	if not minF:
		if log:
			minF = np.percentile(energy,minFPercentile)
		else:
			minF = 0.0
	if not log:
		norm = colors.Normalize(vmin=minF,vmax=maxF)
	else:
		norm = colors.LogNorm(vmin=minF,vmax=maxF)

	#energy[np.where(energy < 1e-5)] = np.nan
	# Downsample array if it is excessively large
	reduceArray = tuple([ max([1,s // 2000]) for s in energy.shape ])
	energy = skimage.measure.block_reduce(energy,reduceArray,np.mean)

	im = ax.imshow(energy,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='viridis',norm=norm)

	if CBar:
		cb = fig.colorbar(im,ax=ax,orientation='vertical')
		if not log:
			cb.formatter.set_powerlimits((-4,4))
		cb.ax.yaxis.set_offset_position('left')
		cb.update_ticks()

	ax.tick_params(reset=True,axis='both',color='w',direction='in')

	# Annotate location of important densities
	if ne is not None:
		if markQC:
			xQC = findDensity(x,ne,0.25)
			xSC = findDensity(x,ne,0.25*0.25)
			if xQC is not None: ax.axvline(xQC/1e-6,color='w',linestyle='--')
			if xSC is not None: ax.axvline(xSC/1e-6,color='w',linestyle='-.')

		if Te is not None:
			bth = math.sqrt(const.k*Te/const.m_e)/const.c
			if markTPDCutoff:
				tpdCutoffNe = srsUtils.tpd.landauCutoffDens(bth,cutoff=landauCutoff)
				diff = np.abs(ne/srsUtils.nCritNIF-tpdCutoffNe)
				xCutoffTPD = x[np.where(diff == np.min(diff))]
				ax.axvline(xCutoffTPD/1e-6,linestyle='--',color='w')

			if markSRSCutoff:
				srsCutoffNe = srsUtils.srs.landauCutoffDens(bth,math.pi,cutoff=landauCutoff)
				diff = np.abs(ne/srsUtils.nCritNIF-srsCutoffNe)
				xCutoffSRS = x[np.where(diff == np.min(diff))]
				ax.axvline(xCutoffSRS/1e-6,linestyle='--',color='r')

		ax2 = ax.twiny()
		srsUtils.misc.addNeScale(ax,ax2,x/1e-6,ne,minVal=minNeTick,interval=neTickInterval,minor=True)

	if xLims:
		ax.set_xlim(xLims)
	if yLims:
		ax.set_ylim(yLims)

	ax.set_xlabel(r'x $/\mu$m')
	ax.set_ylabel('time /ps')
	if CBar and CBarLabel:
		if field.startswith('Magnetic') and field.endswith('Bz'):
			cb.ax.set_title(r'$\frac{e^2}{(m_e\omega_0)^2}\left\langle |B_z|^2 \right\rangle_y$')
		elif field.startswith('Electric_') and field.endswith('Ex'):
			cb.ax.set_title(r'$\frac{e^2}{(m_e\omega_0c)^2}\left\langle |E_x|^2 \right\rangle_y$')
		else:
			cb.ax.set_title(r'$\frac{e^2}{(m_e\omega_0c)^2}\left\langle |E|^2 \right\rangle_y$')

	return fig,ax

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('dataDir')
	parser.add_argument('field')
	parser.add_argument('--prefix',default='regular_')
	parser.add_argument('--space',action='store_true')
	parser.add_argument('--parallel',action='store_true')
	parser.add_argument('--useCached',action='store_true')
	parser.add_argument('--cacheFile')

	parser.add_argument('--densProfFile',default='regular_0000.sdf')
	parser.add_argument('--densitySpecies',default='electrons')
	parser.add_argument('--Te',type=float)

	parser.add_argument('--log',action='store_true')
	parser.add_argument('--xLims',type=float,nargs=2)
	parser.add_argument('--yLims',type=float,nargs=2)
	parser.add_argument('--maxFPercentile',type=float,default=99.9)
	parser.add_argument('--minFPercentile',type=float,default=0.5)
	parser.add_argument('--maxF',type=float)
	parser.add_argument('--minF',type=float)

	parser.add_argument('--noMarkQC',action='store_true')
	parser.add_argument('--markTPDCutoff',action='store_true')
	parser.add_argument('--markSRSCutoff',action='store_true')
	parser.add_argument('--landauCutoff',type=float,default=0.30)
	parser.add_argument('--noCBar',action='store_true')
	parser.add_argument('--noCBarLabel',action='store_true')
	parser.add_argument('--minNeTick',type=float)
	parser.add_argument('--neTickInterval',type=float)

	parser.add_argument('--fontSize',type=float)
	parser.add_argument('-o','--output')
	parser.add_argument('--figSize',type=float,nargs=2)

	args = parser.parse_args()

	if args.Te is not None:
		Te = args.Te*srsUtils.TkeV
	else:
		Te = None

	if args.fontSize:
		import matplotlib as mpl
		mpl.rcParams.update({'font.size':args.fontSize})

	fig = plt.figure()
	ax = fig.add_subplot(111)
	if args.space:
		if args.useCached:
			npCache = np.load(args.cacheFile)
			energy = npCache['energy']
			x      = npCache['x']
			t      = npCache['t']
			ne     = npCache['ne']
		else:
			files = sdfUtils.listFiles(args.dataDir,args.prefix)
			if len(files) == 0:
				raise IOError("Couldn't find any SDF files with prefix {:}".format(args.prefix))
			else:
				print("Found {:} SDF files with prefix {:}".format(len(files),args.prefix))

			# Find quarter critical surface
			sdfProf = sdf.read(os.path.join(args.dataDir,args.densProfFile))
			x  = sdfProf.Grid_Grid.data[0]
			if args.noMarkQC:
				ne = None
			else:
				if args.densitySpecies == '':
					ne = sdfProf.__dict__['Derived_Number_Density'].data
				else:
					ne = sdfProf.__dict__['Derived_Number_Density_'+args.densitySpecies].data

				if len(ne.shape) == 2:
					ne = np.mean(ne,axis=1)
				ne = gaussian_filter1d(ne,sigma=10)

			energy,x,t = calcMeanEnergyVsSpaceTime(files,args.field,parallel=args.parallel)
			if args.cacheFile is not None:
				np.savez_compressed(args.cacheFile,energy=energy,x=x,t=t,ne=ne)

		plotMeanEnergyVsSpaceTime(fig,ax,energy,x,t,args.field,
		    ne=ne,xLims=args.xLims,yLims=args.yLims,maxF=args.maxF,
		    minF=args.minF,maxFPercentile=args.maxFPercentile,
		    minFPercentile=args.minFPercentile,log=args.log,
		    minNeTick=args.minNeTick,neTickInterval=args.neTickInterval,
		    markQC=not args.noMarkQC,CBar=not args.noCBar,
		    CBarLabel=not args.noCBarLabel,Te=Te,
		    markTPDCutoff=args.markTPDCutoff,markSRSCutoff=args.markSRSCutoff,
		    landauCutoff=args.landauCutoff)
	else:
		files = sdfUtils.listFiles(args.dataDir,args.prefix)
		if len(files) == 0:
			raise IOError("Couldn't find any SDF files with prefix {:}".format(args.prefix))
		else:
			print("Found {:} SDF files with prefix {:}".format(len(files),args.prefix))

		meanEnergyVsTime(ax,files,args.field,xLims=args.xLims,log=args.log)

	if args.output:
		if args.figSize:
			fig.set_size_inches(args.figSize)
		fig.tight_layout(pad=0,w_pad=0,h_pad=0)
		fig.savefig(args.output,dpi=600,pad='tight')
	else:
		plt.show()
