#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import scipy.stats
from scipy.ndimage.filters import gaussian_filter1d
import time
import os
import re
import sys
import argparse
import sdf
import threading

import srsUtils
from srsUtils import misc
import sdfUtils
import sdf

mpl.style.use('classic')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('figure', autolayout=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')

# Output directory
_outputDir = 'processed'

# Bodge to remove missing data points from flux vs. time plots
_bodge = False

def plotEnergyDist(ax,fs,eLims,norms,Ts,xlims=None,ylims=None,
                   energyWeight=False,intensityNorm=None,powerNormName=None):
	# Prepare data
	nE = len(fs)
	dE = (eLims[1]-eLims[0])/nE
	bins = eLims[0] + np.arange(nE+1)*dE
	es = 0.5*(bins[1:] + bins[:-1])

	print("Measured intensity above 25keV: {:.3E} W".format(\
	      np.sum((es*fs)[np.where(es > 25.0*srsUtils.keV)])))

	# If weighting by energy vertical scale is intensity, otherwise it is
	# particle flux per unit area. Do energy weighting here for data and apply
	# normalisation. By default use units of s^{-1}cm^{-2} or Wcm^{-2}, hence
	# factors of 1e4.
	if energyWeight:
		fs *= es
		if intensityNorm is not None:
			fs = fs/intensityNorm
		else:
			fs = fs/1e4
	else:
		fs = fs/1e4

	# Calculate expected (fitted) distribution
	dists = np.zeros(norms.shape + fs.shape)
	for i,n,T in zip(range(len(norms)),norms,Ts):
		# This is the `scale' of the distribution (standard deviation?)
		eNorm = 0.5*const.k*T

		# Calculate differences of the cumulative distribution function to get
		# total probability in each energy bin.
		probs = scipy.stats.chi2.cdf(bins[1:] ,df=4,scale=eNorm) \
		       -scipy.stats.chi2.cdf(bins[:-1],df=4,scale=eNorm)

		# If user has requested distribution weighted by energy, do that now.
		if energyWeight:
			dist = n*probs*es
			if intensityNorm is not None:
				dist = dist/intensityNorm
			else:
				dist = dist/1e4
		else:
			dist = n*probs/1e4

		dists[i] = dist
	totDist = np.sum(dists,axis=0)

	# Do actual plotting
	left,right = bins[:-1],bins[1:]
	X = np.array([left,right]).T.flatten()
	Y = np.array([fs,fs]).T.flatten()
	ax.plot(X/srsUtils.keV,Y,'k-',lw=1.0)

	for n,T,dist in zip(norms,Ts,dists):
		#ax.plot(es/srsUtils.keV,dist,label=r'${:.1f}$keV, $\num{{{:.1e}}}n_0$'.format(const.k*T/srsUtils.keV,n/norms[0]),lw=0.50)
		ax.plot(es/srsUtils.keV,dist,label=r'${:.1f}$keV'.format(const.k*T/srsUtils.keV),lw=0.50)
	ax.plot(es/srsUtils.keV,totDist,'--',color='purple',label=r'Fit',lw=0.75)

	ax.set_yscale('log')
	ax.set_xlabel('Energy /keV')
	if energyWeight:
		if intensityNorm is not None:
			ax.set_ylabel('$E\cdot f_e(E)/P_0$')
		else:
			ax.set_ylabel('$E\cdot f_e(E)$ $/\mathrm{W}\cdot\mathrm{cm}^{-2}$')
	else:
		ax.set_ylabel('$f_e(E)$ $/\mathrm{s}^{-1}\cdot\mathrm{cm}^{-2}$')

	if xlims[0] is not None or xlims[1] is not None:
		if xlims[0] is not None and xlims[1] is None:
			ax.set_xlim(xlims[0],bins[-1]/srsUtils.keV)
		elif xlims[0] is None and xlims[1] is not None:
			ax.set_xlim(bins[0]/srsUtils.keV,xlims[1])
		elif xlims[0] is not None and xlims[1] is not None:
			ax.set_xlim(xlims[0],xlims[1])
	else:
		ax.set_xlim(bins[0]/srsUtils.keV,bins[-1]/srsUtils.keV)

	#print(fs)
	fMinLim = float(10**(math.floor(math.log10(np.min(fs[fs != 0.0]))-1.0)))
	if ylims[0] is not None or ylims[1] is not None:
		if ylims[0] is not None and ylims[1] is None:
			ax.set_ylim(ylims[0],ax.get_ylim()[1])
		elif ylims[0] is None and ylims[1] is not None:
			ax.set_ylim(fMinLim,ylims[1])
		elif ylims[0] is not None and ylims[1] is not None:
			ax.set_ylim(ylims[0],ylims[1])
	else:
		ax.set_ylim(fMinLim,ax.get_ylim()[1])
	#ax.set_ylim(ax.get_ylim()[0],1.0)

	#plt.plot(np.linspace(0.0,max(kes)/const.e/1e3,1000),mbpdf*1e0)
	ax.grid()
	return ax

def plotEnergyDistVsTime(ax,fs,eLims,tLims,norms,Ts,xlims=None,ylims=None,flims=None):
	nE = fs.shape[0]
	dE = (eLims[1]-eLims[0])/(nE-1.0)
	ebins = eLims[0] + np.arange(nE+1)*dE
	es = 0.5*(ebins[1:] + ebins[:-1])

	nt = fs.shape[1]
	dt = (tLims[1]-tLims[0])/(nt-1.0)
	tbins = tLims[0] + np.arange(nt+1)*dt
	ts = 0.5*(tbins[1:] + tbins[:-1])
	print(eLims)
	print(tLims)
	extent = misc.getExtent(ts/1e-12,es/srsUtils.keV)
	cMapNorm = colors.LogNorm()
	im = ax.imshow(fs,origin='lower',norm=cMapNorm,interpolation='none',
	extent=extent,aspect='auto',cmap='viridis')

	cb = plt.colorbar(im,orientation='horizontal')

	if xlims[0] or xlims[1]:
		if xlims[0] and not xlims[1]:
			ax.set_xlim(xlims[0],tbins[-1]/1e-12)
		if not xlims[0] and xlims[1]:
			ax.set_xlim(tbins[0]/1e-12,xlims[1])
		if xlims[0] and xlims[1]:
			ax.set_xlim(xlims[0],xlims[1])
	else:
		ax.set_xlim(tbins[0]/1e-12,tbins[-1]/1e-12)

	if ylims[0] or ylims[1]:
		if ylims[0] and not ylims[1]:
			ax.set_ylim(ylims[0],ebins[-1]/srsUtils.keV)
		if not ylims[0] and ylims[1]:
			ax.set_ylim(ebins[0]/srsUtils.keV,ylims[1])
		if ylims[0] and ylims[1]:
			ax.set_ylim(ylims[0],ylims[1])
	else:
		ax.set_ylim(ebins[0]/srsUtils.keV,ebins[-1]/srsUtils.keV)

	ax.set_xlabel('time /ps')
	ax.set_ylabel('Energy /keV')

	return ax

def plotEnergyDistVsSpace(ax,fs,eLims,spaceLims,norms,Ts,xlims=None,ylims=None,flims=None):
	nE = fs.shape[0]
	dE = (eLims[1]-eLims[0])/(ne-1.0)
	ebins = eLims[0] + np.arange(ne+1)*dE
	es = 0.5*(ebins[1:] + ebins[:-1])

	nx = fs.shape[1]
	dx = (spaceLims[1]-spaceLims[0])/(nx-1.0)
	xbins = spaceLims[0] + np.arange(nx+1)*dx
	xs = 0.5*(xbins[1:] + xbins[:-1])
	print(eLims)
	print(spaceLims)
	extent = misc.getExtent(xs/1e-6,es/srsUtils.keV)
	cMapNorm = colors.LogNorm()
	im = ax.imshow(fs,origin='lower',norm=cMapNorm,interpolation='none',
	               extent=extent,aspect='auto',cmap='viridis')

	cb = plt.colorbar(im,orientation='horizontal')

	if xlims[0] or xlims[1]:
		if xlims[0] and not xlims[1]:
			ax.set_xlim(xlims[0],xbins[-1]/1e-6)
		if not xlims[0] and xlims[1]:
			ax.set_xlim(xbins[0]/1e-6,xlims[1])
		if xlims[0] and xlims[1]:
			ax.set_xlim(xlims[0],xlims[1])
	else:
		ax.set_xlim(xbins[0]/1e-6,xbins[-1]/1e-6)

	if ylims[0] or ylims[1]:
		if ylims[0] and not ylims[1]:
			ax.set_ylim(ylims[0],ebins[-1]/srsUtils.keV)
		if not ylims[0] and ylims[1]:
			ax.set_ylim(ebins[0]/srsUtils.keV,ylims[1])
		if ylims[0] and ylims[1]:
			ax.set_ylim(ylims[0],ylims[1])
	else:
		ax.set_ylim(ebins[0]/srsUtils.keV,ebins[-1]/srsUtils.keV)

	ax.set_xlabel('space /$\mu$m')
	ax.set_ylabel('Energy /keV')

	return ax

def plotVsTime(ax,fs,eLims,tLims,norms,Ts,
               #sumBounds=[(25.0,50.0),(50.0,100.0),(100.0,None),(50,None)],
			   sumBounds=[(25.0,50.0),(50.0,100.0),(100.0,None)],
			   #sumBounds=[(10.0,20.0),(20.,40.),(40,None)],
               #sumBounds=[(5.0,10.0),(10.0,20.0),(0.0,None)],
               #sumBounds=[(50.0,100.0),(100.0,None)],
               #sumBounds=[(50.0,None),(100.0,None)],
               xLims=None,fLims=None,plotIntensity=True,intensityNorm=None,
	           subtractFit=True,smoothLen=None):
	'''
	Plots power through boundary as a function of time

	Integrate over variable number of energy bins to get different 'total' energies
	e.g. >25keV, >100keV

	sumBounds specifies bins to integrate over, e.g. [(25.,100.),(100.,None)]
	would cause the function to plot two separate time-series: the total flux of
	particles between 25<E/keV<100 and the flux with E/keV > 100.

	Uses the distributions specified by norms, Ts
	'''
	print("plotVsTime energy bins /keV:")
	print(sumBounds)

	nT = fs.shape[1]
	dT = (tLims[1]-tLims[0])/nT
	ts = np.linspace(tLims[0],tLims[1],nT)

	nE = fs.shape[0]
	dE = (eLims[1]-eLims[0])/nE
	bins = eLims[0] + np.arange(nE+1)*dE
	es = 0.5*(bins[1:] + bins[:-1])

	if plotIntensity:
		fs *= es[:,np.newaxis]
		if intensityNorm is not None:
			fs = fs/intensityNorm
		else:
			fs = fs/1e4
	else:
		fs = fs/1e4

	print(fs.shape)
	print(np.mean(np.sum(fs,axis=0)))
	print(dT*np.sum(fs,axis=(0,1)))

	# Calculate expected fluxes
	expFlux = np.zeros(len(sumBounds))
	for i,bounds in enumerate(sumBounds):
		lLim = bounds[0]*srsUtils.keV
		if lLim is None:
			lLim = 0.0

		for n,T in zip(norms,Ts):
			distWidth = 0.5*const.k*T
			if plotIntensity:
				# Either have output intensity in
				if intensityNorm is not None:
					fac = 1.0/intensityNorm
				else:
					fac = 1.0/1e4

				if bounds[1] is None:
					uLim = 100.0*distWidth
				else:
					uLim = bounds[1]*srsUtils.keV

				integrand = lambda x: x*scipy.stats.chi2.pdf(x,df=4,scale=distWidth)
				flux = fac*n*scipy.integrate.quad(integrand,lLim,uLim)[0]#/Lt*(nT*dT/Lt)
			else:
				lLimCDF = scipy.stats.chi2.cdf(lLim,df=4,scale=distWidth)
				if bounds[1] is None:
					uLimCDF = 1.0
				else:
					uLimCDF = scipy.stats.chi2.cdf(bounds[1]*srsUtils.keV,df=4,scale=distWidth)

				# Convert to cm^-2
				flux = n*(uLimCDF-lLimCDF)/1e4

			expFlux[i] += flux

	print("\nFitted flux in each energy bin / P0:")
	print(expFlux)

	intFs = np.zeros((len(sumBounds),fs.shape[1]))
	for i,bounds in enumerate(sumBounds):
		if bounds[0] is not None and bounds[1] is None:
			ind = np.where(es > bounds[0]*srsUtils.keV)
			label = r'$> {:.0f}$keV'.format(bounds[0])
		elif bounds[0] is None and bounds[1] is not None:
			ind = np.where(es < bounds[1]*srsUtils.keV)
			label = r'$< {:.0f}$keV'.format(bounds[1])
		elif bounds[1] is not None and bounds[1] is not None:
			ind = np.where(np.logical_and(es > bounds[0]*srsUtils.keV,
			                              es < bounds[1]*srsUtils.keV))
			#label = r'${:.0f} < E < {:.0f}$keV'.format(bounds[0],bounds[1])
			label = r'${:.0f}$-${:.0f}$keV'.format(bounds[0],bounds[1])
		else:
			ind = np.where(es == es)
			label = r'Total'

		intFs[i] = np.sum(fs[ind],axis=0)

		if _bodge:
			zeros = np.where(np.logical_and(intFs[i] == 0.0,ts/1e-12 > 1.5))
			print("Removing zero points at times (/ps):")
			print(ts[zeros]/1e-12)
			intFs[i][zeros] = 0.5*(intFs[i][zeros[0]-1] + intFs[i][zeros[0]+1])

		if subtractFit:
			intFs[i] -= expFlux[i]

		intFsPlot = np.copy(intFs[i])
		if smoothLen is not None:
			dt = ts[1]-ts[0]
			intFsPlot = gaussian_filter1d(intFsPlot,sigma=smoothLen/dt)

		if i == 3:
			base_line, = ax.plot(ts/1e-12,intFsPlot,'k--',label=label)
		else:
			base_line, = ax.plot(ts/1e-12,intFsPlot,label=label)

		if not subtractFit:
			color = base_line.get_color()
			ax.axhline(expFlux[i],color=base_line.get_color(),linestyle='--')

	print("\nAverage measured flux in each energy bin / P0:")
	print(np.mean(intFs,axis=1))

	if xlims[0] or xlims[1]:
		if xlims[0] and not xlims[1]:
			ax.set_xlim(xlims[0],tLims[-1]/1e-12)
		if not xlims[0] and xlims[1]:
			ax.set_xlim(tLims[0]/1e-12,xlims[1])
		if xlims[0] and xlims[1]:
			ax.set_xlim(xlims[0],xlims[1])
	else:
		ax.set_xlim(tLims[0]/1e-12,tLims[-1]/1e-12)

	if fLims[0] or fLims[1]:
		if fLims[0] and not fLims[1]:
			ax.set_ylim(fLims[0],ax.get_ylim()[1])
		if not fLims[0] and fLims[1]:
			ax.set_ylim(0.0,fLims[1])
		if fLims[0] and fLims[1]:
			ax.set_ylim(fLims[0],fLims[1])
	else:
		ax.set_ylim(0.0,ax.get_ylim()[1])

	ax.set_xlabel('time /ps')
	if plotIntensity:
		if intensityNorm is not None:
			ax.set_ylabel(r'$I/I_0$')
		else:
			ax.set_ylabel(r'$I$ $/\mathrm{W} \cdot \mathrm{cm}^{-2}$')
	else:
		ax.set_ylabel(r'$\mathrm{Particle}$ $\mathrm{Flux}$ $/\mathrm{s} \cdot \mathrm{m}^{-2}$')
	ax.grid()

	return ax

def printFitDistStats(ne,Te):
	vth = np.sqrt(const.k*Te/const.m_e)

	cutoff = 25.0*srsUtils.keV
	eNorm = 0.5*const.k*Te
	fac = 1.0 - scipy.stats.chi2.cdf(cutoff,df=4,scale=eNorm)
	print(fac)
	#dist = n*probs*es

	# Calculate total electron intensity for each fitted distribution
	Ie = 1e-4*ne*vth*0.5*const.k*Te*math.sqrt(8./math.pi)
	print('Intensity of fitted electron flux through boundary:')
	if Ie.shape == (1,):
		print('I_e = {:.3E} W/cm^2'.format(Ie[0]))
	else:
		print('I_e = {:} W/cm^2'.format(Ie))
	if Ie.shape != (1,):
		print('Total: {:.3E} W/cm^2'.format(np.sum(Ie)))
		print('Total >25keV: {:.3E} W/cm^2\n'.format(np.sum(fac*Ie)))
	else:
		print('')

	# Calculate total electron power for each fitted distribution
	Pe = Ie*(ranges[1][1]-ranges[1][0])*1e4
	print('Power of fitted electron flux through boundary:')
	if Pe.shape == (1,):
		print('P_e = {:.3E} W'.format(Pe[0]))
	else:
		print('P_e = {:} W'.format(Pe))
		print('P_e>25keV = {:} W'.format(fac*Pe))
	if Pe.shape != (1,):
		print('Total: {:.3E} W'.format(np.sum(Pe)))
		print('Total >25keV: {:.3E} W\n'.format(np.sum(fac*Pe)))
		#print('Total (non-thermal): {:.3E} W\n'.format(np.sum(Pe[1:])))
	else:
		print('')
	#exit()

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

	#print(I*Ly)
	return I*Ly

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('dataDir')
	parser.add_argument('boundary')
	parser.add_argument('--probeFile')
	parser.add_argument('--onlyShow',action='store_true')
	parser.add_argument('-T','--temperature',type=float,required=True,nargs='*')
	parser.add_argument('-n','--density',type=float,required=True,nargs='*')

	parser.add_argument('--smoothLen',type=float)

	parser.add_argument('--plotEnergy',action='store_true')
	parser.add_argument('--plotTime',action='store_true')
	parser.add_argument('--plotSpace',action='store_true')
	parser.add_argument('--noEnergyWeight',action='store_true') # Whether to weight by energy
	parser.add_argument('--bodge',action='store_true')
	parser.add_argument('--intensityNorm', '-I',type=float)
	parser.add_argument('--subtractFit',action='store_true')
	parser.add_argument('--minX',type=float)
	parser.add_argument('--maxX',type=float)
	parser.add_argument('--minY',type=float)
	parser.add_argument('--maxY',type=float)
	parser.add_argument('--minF',type=float)
	parser.add_argument('--maxF',type=float)

	parser.add_argument('--minT',type=float)
	parser.add_argument('--maxT',type=float)
	parser.add_argument('--minS',type=float)
	parser.add_argument('--maxS',type=float)

	parser.add_argument('-o','--output')
	parser.add_argument('--fontSize',type=float)
	parser.add_argument('--noTitle' ,action='store_true')
	parser.add_argument('--noLegend',action='store_true')
	parser.add_argument('-f','--figSize',type=float,nargs=2,default=(0.5*0.8*9,2.5))
	args = parser.parse_args()

	numQuants = int(args.plotEnergy) + int(args.plotTime) + int(args.plotSpace)
	if numQuants > 2:
		raise ValueError("Can't histogram more than two quantities")
	if numQuants < 1:
		raise ValueError("Need at least one quantity to plot against!")

	# Enable global flag; signals to remove missing data points
	if args.bodge:
		_bodge = True

	# Read data
	if args.probeFile is not None:
		data = np.load(args.probeFile)
	else:
		outDir = os.path.join(args.dataDir,_outputDir)
		data = np.load(os.path.join(outDir,'probeData_{:}.npz'.format(args.boundary)))

	hist = data['hist']
	ranges = data['ranges']

	# Simulation cell size and lengths
	dx = float(data['dx'])
	dy = float(data['dy'])
	Lx = float(data['Lx'])
	Ly = float(data['Ly'])
	Lt = float(data['Lt'])

	# Histogram data shape
	nE = hist.shape[0]
	nS = hist.shape[1]
	nT = hist.shape[2]
	dE = (ranges[0][1]-ranges[0][0])/nE
	dS = (ranges[1][1]-ranges[1][0])/nS
	dT = (ranges[2][1]-ranges[2][0])/nT

	print('Distribution function shape: {:}\n'.format(hist.shape))
	print("Energy bins:\nMin: {:.2f}keV, max: {:.2f}keV, dE: {:.2f}keV, nE: {:}\n".format(ranges[0][0]/srsUtils.keV,ranges[0][1]/srsUtils.keV,dE/srsUtils.keV,nE))
	print("Space  bins:\nMin: {:.6f}μm, max: {:.6f}μm, dS: {:.2f}nm, nS: {:}\n".format(ranges[1][0]/1e-6,ranges[1][1]/1e-6,dS/1e-9,nS))
	print("Time   bins:\nMin: {:.2f}ps, max: {:.2f}ps, dE: {:.2f}fs, nT: {:}\n".format(ranges[2][0]/1e-12,ranges[2][1]/1e-12,dT/1e-15,nT))

	if args.minT or args.maxT or args.minS or args.maxS:
		print("Truncating histogram")
		# Calculate min and max values from range if not given by user
		if args.minS: minS = args.minS*1e-6
		else: minS = ranges[1][0]

		if args.maxS: maxS = args.maxS*1e-6
		else: maxS = ranges[1][1]

		if args.minT: minT = args.minT*1e-12
		else: minT = ranges[2][0]

		if args.maxT: maxT = args.maxT*1e-12
		else: maxT = ranges[2][1]

		# Convert these limits into indices clamped to dimensions of bin arrays
		minS = max(0, int(math.ceil((minS-ranges[1][0])/dS)))
		maxS = min(nS,int(math.floor((maxS-ranges[1][0])/dS))+1)

		minT = max(0, int(math.ceil((minT-ranges[2][0])/dT)))
		maxT = min(nT,int(math.floor((maxT-ranges[2][0])/dT))+1)

		# Cut down histogram based on calculated indices
		hist = hist[:,minS:maxS,minT:maxT]

		# Cut down bin ranges to match
		ranges[1] = [ranges[1][0]+minS*dS,ranges[1][0]+(maxS-1)*dS]
		ranges[2] = [ranges[2][0]+minT*dT,ranges[2][0]+(maxT-1)*dT]

		# Update histogram shape
		nS = hist.shape[1]
		nT = hist.shape[2]

		print("New bin specification:")
		print("Energy bins:\nMin: {:.2f}keV, max: {:.2f}keV, dE: {:.2f}keV, nE: {:}\n".format(ranges[0][0]/srsUtils.keV,ranges[0][1]/srsUtils.keV,dE/srsUtils.keV,nE))
		print("Space  bins:\nMin: {:.6f}μm, max: {:.6f}μm, dS: {:.2f}nm, nS: {:}\n".format(ranges[1][0]/1e-6,ranges[1][1]/1e-6,dS/1e-9,nS))
		print("Time   bins:\nMin: {:.2f}ps, max: {:.2f}ps, dT: {:.2f}fs, nT: {:}\n".format(ranges[2][0]/1e-12,ranges[2][1]/1e-12,dT/1e-15,nT))


	if args.onlyShow:
		exit()

	# Fitted distribution parameters
	ne = np.array(args.density)*srsUtils.nCritNIF
	Te = np.array(args.temperature)*1e3*const.e/const.k

	# Printed fitted distribution stats
	printFitDistStats(ne,Te)

	if args.fontSize:
		import matplotlib as mpl
		mpl.rcParams.update({'font.size':args.fontSize})

	fig = plt.figure()
	ax = fig.add_subplot(111)

	xlims = (args.minX,args.maxX)
	ylims = (args.minY,args.maxY)
	flims = (args.minF,args.maxF)

	# Calculate normalisation of distributions.
	#
	# Assumes population is symmetric and propagates in both directions. Fits
	# to the superthermal components therefore contain only half the number of
	# particles suggested by the fit density
	vth = np.sqrt(const.k*Te/const.m_e)
	norm = ne*vth/math.sqrt(2.0*math.pi)
	print(norm[0]/(np.sum(hist)/(nS*nT*dS*dT)))

	#if args.intensityNorm is not None:
	#	args.intensityNorm = planeWavePower(srsUtils.wnVacNIF,args.intensity*1e4,
	#	    os.path.join(args.dataDir,'regular_0000.sdf'))

	if args.intensityNorm is not None:
		args.intensityNorm = args.intensityNorm*1e4

	## If we are normalising the output by some power do this here.
	#if args.intensityNorm is not None:
	#	print("Laser power: {:.3e}W\n".format(args.intensityNorm))
	#	norm = norm/args.intensityNorm

	# Average histogram over axes which aren't needed for plotting. Has to be
	# done here as plotting functions don't `know' about other histogram axes.
	# Then call plotting functions.
	if args.plotEnergy and args.plotTime:
		dist = np.sum(hist,axis=(1,))

		plotEnergyDistVsTime(ax,dist,ranges[0],ranges[2],norm,Te,xlims,ylims,flims)
	elif args.plotEnergy and args.plotSpace:
		dist = np.sum(hist,axis=(2,))

		plotEnergyDistVsSpace(ax,dist,ranges[0],ranges[1],norm,Te,xlims,ylims,flims)
	elif args.plotEnergy:
		# Average over space and time to get a flux/intensity for each energy
		# bin. Don't divide by dE as here the total energy in each bin is
		# more useful than an energy density.
		# Or is it? Would make comparison between distributions from different
		# simulations more straightforward (bin size varies). On the other hand
		# sanity checking output vs. plotVsTime is easier this way.
		dist = np.mean(hist,axis=(1,2))/(dS*dT)

		plotEnergyDist(ax,dist,ranges[0],norm,Te,xlims,flims,
			energyWeight=not args.noEnergyWeight,intensityNorm=args.intensityNorm)
		if not args.noLegend:
			if args.fontSize:
				ax.legend(fontsize=args.fontSize-2,loc='best')
			else:
				ax.legend(loc='best')
	elif args.plotTime:
		# If plotting against time it doesn't make sense to plot total energy in
		# each time bin as this depends on the length of the bin, so we will
		# only ever be plotting power/intensity. Therefore divide histogram of
		# particle counts by dT to get a particle flux.

		# Calculate mean to get instantaneous flux/intensity
		dist = np.mean(hist,axis=(1,))/(dS*dT)

		if args.smoothLen is not None:
			args.smoothLen = args.smoothLen*1e-12

		plotVsTime(ax,dist,ranges[0],ranges[2],norm,Te,xLims=xlims,fLims=flims,
		    plotIntensity=not args.noEnergyWeight,intensityNorm=args.intensityNorm,
		    subtractFit=args.subtractFit,smoothLen=args.smoothLen)
		if not args.noLegend:
			if args.fontSize:
				ax.legend(fontsize=args.fontSize-3,loc='best',ncol=2)
			else:
				ax.legend(loc='best')

	if not args.noTitle:
		fig.suptitle('${:}_{{\mathrm{{{:}}}}}$'.format(args.boundary[0],args.boundary[-3:]))

	if args.output:
		fig.set_size_inches(args.figSize[0],args.figSize[1])
		fig.savefig(args.output)
	else:
		plt.show()
