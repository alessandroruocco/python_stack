#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import scipy.constants as const
import scipy.stats

import srsUtils
from srsUtils import speckle
from srsUtils.misc import sizeof_fmt
import pyEPOCH
from pyEPOCH import inputBlock

#def _wnDensityRamp(x,x0,x1,n1,Ln,omega0):
#	''' Calculates the laser wavenumber at points along the density ramp '''
#	if x < x1:
#		return omega0/const.c
#	else:
#		return np.sqrt(omega0**2-n1*const.e**2/(const.m_e*const.epsilon_0)*np.exp((x-x1)/Ln))
#
#def _densityRampCountWls(x,x1,n1,Ln,omega0):
#	if x1 < 0: raise ArithmeticError("x1 is less than zero")
#
#	if x <= x1:
#		return x*omega0/(2*math.pi*const.c)
#	else:
#		# Number of wavelengths up to x1
#		nx1 = _densityRampCountWls(x1,x1,n1,Ln,omega0)
#
#		# Number between x1 and x
#		nxx =
#
#	k0_0 = wnDensityRamp(x0
#	return Ln/(math.pi*const.c)*()

def _speckleLength(x1,n1,Ln,F,omega0,intensFac=0.5):
	'''
	Calculates the length of a speckle
	TODO: Correctly figure out the effects of plasma density on this...

	intensFac: We choose the length based on the point at which the beam
	           intensity on axis drops to a given threshold, which is set
			   with this parameter
	'''
	# Waist based on vacuum wavelength
	k0 = omega0/const.c
	wl0 = 2.0*math.pi/k0
	w0 = F*wl0

	op = np.sqrt(n1*const.e**2/(const.m_e*const.epsilon_0))
	k = np.sqrt(omega0**2 - op**2)/const.c

	return 2.0*speckle.gaussianBeamLengthAtAmp(0.0,math.sqrt(intensFac),k,w0)
	#return 10*2.*math.pi*const.c/omega0*F**2

def singleSpeckle(ne,Ln,Te,                               # Plasma properties
	              I,F,nRange=None,wlVac0=srsUtils.wlVacNIF,omega0=None, # Laser properties
				  theta=0.0,planeWave=False,
				  burst=True,probes=True,particles=True,    # Diagnostics required
                  tEnd=None,ppce=512,ppci=0,grid_res_x=1.2,grid_res_y=1.2,N_acc=1000,
				  bufferWidth=0.0,xBoundIntensFac=0.5,yBoundIntensFac=0.01,sLenFac=None): # Simulation parameters
	'''
	Creates an input deck for a single speckle SRS simulation along a density
	gradient.

	Description
	-----------

	In Yin et al. (2013) the following parameters were used:

	Te     = 4keV
	ne     = 0.14*nCrit
	I      = 2.5*10^15 W/cm^2
	F      = 8.0
	wlVac0 = 351 nm (NIF)
	tEnd   = 9 ns
	ppce    = 512

	Parameters
	----------

	All in SI units unless otherwise specified

	neMid  : Electron density at maximum laser intensity
	Ln     : Density scale-length
	Te     : Electron temperature
	I      : Laser intensity
	wlVac0 : Vacuum wavelength of the laser (ignored if omega0 is set)
	omega0 : Frequency of the laser (takes precedence over wlVac0))
	tEnd   : Simulation end time
	ppce    : Number of particles per cell
	'''
	# Laser parameters
	if not omega0: omega0 = const.c*2.0*math.pi/wlVac0
	nCrit   = const.m_e*const.epsilon_0/const.e**2*omega0**2
	k0 = omega0/const.c
	wl0 = 2*math.pi/k0

	if len(ne) == 1:
		nMid = ne[0]
	elif len(ne) == 2:
		if ne[0] > ne[1]:
			raise ValueError("Density range should have lower density first")
		nMin = ne[0]
		nMax = ne[1]
	else:
		raise ValueError("Density range has more than 2 densities...")

	# Calculate speckle length, or rather calculate the length of box we're
	# going to use for the 'speckle'/plane wave

	# If we've specified a density range then do calculation based on that
	# Warning: if this doesn't give a large enough length to fit the 'speckle'
	# then you might be in trouble...
	if len(ne) == 2:
		Ls = Ln*math.log(ne[1]/ne[0])
		LsVac = Ls
		nMid = ne[0]*np.exp(0.5*Ls/Ln)
	# Otherwise calculate based on getting boundary laser intensity down to a
	# certain fraction (assuming Gaussian beam speckle model)
	else:
		Ls    = _speckleLength(0.0,nMid,Ln,F,omega0,xBoundIntensFac)
		LsVac = _speckleLength(0.0,0.0,  Ln,F,omega0,xBoundIntensFac)

	# If the sLenFac parameter is provided, multiply the above lengths by this
	# factor...
	if sLenFac:
		LsOrig = LsVac
		Ls    *= sLenFac
		LsVac *= sLenFac

	WsVac = speckle.speckleWidth(F,wl0)

	print('Ls: {Ls}μm'.format(Ls=Ls/1e-6))
	nMin = nMid*np.exp(-0.5*Ls/Ln*math.cos(theta))
	nMax = nMid*np.exp( 0.5*Ls/Ln*math.cos(theta))
	print("nMin: {:}m^-3, nMax: {:}m^-3".format(nMin,nMax))
	print("nMin: {:}nCr, nMax: {:}nCr".format(nMin/nCrit,nMax/nCrit))

	opMin = math.sqrt(nMin/(const.m_e*const.epsilon_0))*const.e
	opMid = math.sqrt(nMid/(const.m_e*const.epsilon_0))*const.e
	opMax = math.sqrt(nMax/(const.m_e*const.epsilon_0))*const.e
	vth = math.sqrt(const.k*Te/const.m_e)
	ldMax = math.sqrt(const.k*Te*const.epsilon_0/nMin)/const.e
	ldMin = math.sqrt(const.k*Te*const.epsilon_0/nMax)/const.e
	kLMin = math.sqrt(omega0**2 - opMax**2)/const.c
	kLMid = math.sqrt(omega0**2 - opMid**2)/const.c
	kLMax = math.sqrt(omega0**2 - opMin**2)/const.c

	# Check that SRS can occur at lower density
	try:
		srsWNs = srsUtils.srsWNs(nMin,Te,omega0)
		bSRSGrowthRate = srsUtils.growthRateSRS(nMin,Te,srsUtils.intensityToEField(I),srsWNs['kb'],omega0)
	except AssertionError:
		raise AssertionError("Minimum density too high for SRS to occur")

	# Work out maximum density for SRS
	nUMaxSRS = nMax
	nLMaxSRS = nMin
	maxErr = 1e-6
	nMaxSRS = nUMaxSRS
	while True:
		srsWNs = srsUtils.srsWNs(nMaxSRS,Te,omega0)
		bSRSGrowthRate = srsUtils.growthRateSRS(nMaxSRS,Te,srsUtils.intensityToEField(I),srsWNs['kb'],omega0)

		if math.isnan(bSRSGrowthRate):
			nUMaxSRS = nMaxSRS
		elif (nMaxSRS-nLMaxSRS)/nCrit < maxErr:
			break
		else:
			nLMaxSRS = nMaxSRS

		nMaxSRS = 0.5*(nUMaxSRS + nLMaxSRS)
	print('nMaxSRS: {:}m^-3 = {:}nCr'.format(nMaxSRS,nMaxSRS/nCrit))

	srsWNsMinNe    = srsUtils.srsWNs(nMin,Te,omega0)
	srsOmegasMinNe = srsUtils.srsOmegas(nMin,Te,omega0)
	lwMinPhVel = srsOmegasMinNe['kb']/srsWNsMinNe['kb']

	# Simulation domain
	x0 = 0.0
	x1 = x0 + bufferWidth
	x2 = x1 + Ls

	#print(x1)
	#print(x2)

	y0 = 0.0

	CHRatio = 42.3/57.2
	hProp = 1./(6*CHRatio+1.)
	cProp = CHRatio/(6*CHRatio+1.)

	print("n_h/n_e: {:}".format(hProp))
	print("n_c/n_e: {:}".format(cProp))

	profArgs = {'x0':x0,'x1':x1,'x2':x2,'y0':y0,'Ln':Ln}
	if theta:
		profArgs['angleDep'] = '*cos({theta}) + (y-{y0})*sin({theta})'.format(theta=theta,y0=y0)
	else:
		profArgs['angleDep'] = ''

	# Density profile
	if bufferWidth == 0.0:
		profArgs['buffProf'] = ''
	else:
		profArgs['buffProf'] = '*exp(-(((x-{x1}){angleDep})/{s})^2)'
		#profArgs['buffProf'] = '*if(x lt {xZeroDens},0,exp(-(((x-{x1}){angleDep})/{s})^2))'
		s = bufferWidth/3.
		minDens = 1e-3
		#xZeroDens = x1-s*math.sqrt(math.log(nMin/nCrit/1e-3))
		#print(xZeroDens)
		#exit()
		profArgs['buffProf'] = profArgs['buffProf'].format(s=s,**profArgs)
		#profArgs['buffProf'] = profArgs['buffProf'].format(s=s,xZeroDens=xZeroDens,**profArgs)

	densTemplate = 'if(x lt {x1}, {n1}{buffProf}, {n1}*exp(((x-{x1}){angleDep})/{Ln}))'
	#densTemplate = 'if(x lt {x1}, {n1}{buffProf}, if(x lt {x2},{n1}*exp(((x-{x1}){angleDep})/{Ln}),{n2}))'
	eDensString = densTemplate.format(n1=nMin,n2=nMax,**profArgs)
	if ppci:
		hf = hProp
		cf = cProp
		hDensString = densTemplate.format(n1=hf*nMin,n2=hf*nMax,**profArgs)
		cDensString = densTemplate.format(n1=cf*nMin,n2=cf*nMax,**profArgs)


	xs = x1+Ls/2.0 # Position of speckle centre

	xMin = x0
	xMax = x2

	Lx = xMax-xMin
	# For elongated plane-wave simulations
	if sLenFac:
		Ly = 2.0*speckle.gaussianBeamWidthAtAmp(0.5*LsOrig,
		                                        math.sqrt(yBoundIntensFac),kLMid,WsVac,norm='section')
	else:
		Ly = 2.0*speckle.gaussianBeamWidthAtAmp(max(abs(xMax-xs),abs(xs-xMin)),
		                                        math.sqrt(yBoundIntensFac),kLMid,WsVac,norm='section')

	yMin = -0.5*Ly
	yMax =  0.5*Ly

	# Define preliminary grid spacing
	if nMax > srsUtils.nCritNIF/4.:
		dx = grid_res_x * 2.*vth/srsUtils.omegaNIF
		dy = grid_res_y * 2.*vth/srsUtils.omegaNIF
	else:
		dx = grid_res_x*ldMin
		dy = grid_res_y*ldMin # Could potentially be larger

	# Calculate number of cells from grid spacing
	Nx = int(math.ceil(Lx/dx))
	Ny = int(math.ceil(Ly/dy))
	N  = Nx*Ny

	# Recalculate grid spacing from no. of cells
	dx = Lx/(Nx-1)
	dy = Ly/(Ny-1)

	ppc = ppce + 2*ppci
	Np    = ppc*N

	dt_mult = 0.95
	# Expected dt
	dtExp     = pyEPOCH.calcDt(dx,dy=dy,op=opMax,vth=vth,omega0=omega0,dt_mult=dt_mult)
	nStepsExp = tEnd/dtExp

	# Additional laser parameters
	if not planeWave:
		# Set up gaussian beam
		w0 = F*wlVac0
		z0 = x1+LsVac/2.0
		z  = xMin-z0
		zr = math.pi*w0**2/wl0
		w  = w0*np.sqrt(1.0+(z/zr)**2)
		R  = z*(1.0+(zr/z)**2)
		ez_mod = '{w0}/{w}*exp(-(y/{w})^2)'.format(w0=w0,w=w)
		ez_arg = '-({k}*({z}) + {k}*(y^2)/(2*({R})))'.format(k=kLMax,R=R,z=z)
		y_bc_pcl = 'thermal'
		y_bc_fld = 'open'
	else:
		ez_mod = '1.0'
		ez_arg = '0.0'
		y_bc_pcl = 'periodic'
		y_bc_fld = 'periodic'

	riseTime    = 70.0*2*math.pi/opMax
	initAmpFrac = 1e-3
	wParam      = riseTime/math.sqrt(-math.log(initAmpFrac))


	# Output parameters and data usage

	# First calculate data usage of common output types
	gridVarData = N*8
	partVarData = Np*8
	restartData = 9*gridVarData + 7*partVarData

	# Each diagnostic is defined by a dictionary containing at least:
	# 'name'    : Name of the diagnostic
	# 'dNt'     : Number of steps between outputs
	# 'dNt_acc' : Number of steps to accumulate for (must be the same for all diagnostics)
	# 'N'       : Total number of outputs
	# 'data'    : Data for a single outputs
	# 'physfreq': (Optional) Physical frequency of interest for diagnostic as
	#             angular frequency

	# Diagnostics stored in a dict:
	diag = {}

	# Grid variable regular outputs
	# Interval between outputs - determine from max absolute growth rate
	d = {}
	d['name'] = 'regular'
	d['dNt'] = int(math.floor(1.0/bSRSGrowthRate/5.0/dtExp))
	d['N'] = int(math.floor(tEnd/(d['dNt']*dtExp)))
	d['data']  = 4*gridVarData
	d['physFreq'] = 2.0*math.pi*bSRSGrowthRate
	d['physFreqName'] = 'γ_bSRS'
	diag['reg'] = d

	# Bursts
	d = {}
	d['name'] = 'single burst'
	d['oSample'] = 4.0   # Minimum oversample of laser frequency, i.e. ω_nyq/ω_0
	d['freqRes'] = 0.005 # Maximum Δω/ω_0

	d['dNt'] = int(math.floor(math.pi/(d['oSample']*dtExp*omega0)))
	d['N'] = int(math.ceil(2.0*math.pi/(d['freqRes']*d['dNt']*dtExp*omega0)))
	N_acc = d['N'] # Fix accumulator length to length of burst
	d['dNt_acc'] = d['dNt']*N_acc
	d['data']  = gridVarData
	d['physFreq'] = omega0
	d['physFreqName'] = 'ω_0'
	diag['burst'] = d

	d = {}
	d['name'] = 'all bursts'
	d['N'] = 5
	d['dNt'] = int(math.floor(tEnd/(diag['burst']['dNt_acc']*d['N']*dtExp)))*diag['burst']['dNt_acc']
	d['data'] = diag['burst']['N']*diag['burst']['data']
	if diag['burst']['N']*dtExp > d['dNt']*dtExp:
		raise ValueError("Length of a burst is greater than burst interval")
	diag['aBurst'] = d

	# Boundaries
	d = {}
	d['cellsFromBoundary'] = 5
	d['name'] = 'boundary fields'
	d['dNt'] = diag['burst']['dNt'] # Time interval between samples
	d['dNt_acc'] = d['dNt']*N_acc
	d['N'] = int(math.floor(tEnd/(d['dNt']*dtExp)))
	d['data']  = (6*2*(Nx+Ny) + 4*Nx + 4*Ny)*8 # Latter two due to output of grid
	d['physFreq'] = omega0
	d['physFreqName'] = 'ω_0'
	diag['bound'] = d

	# Strip
	d = {}
	d['name'] = 'strip fields'
	d['dNt'] = diag['burst']['dNt'] # Time interval between samples
	d['dNt_acc'] = d['dNt']*N_acc
	d['N'] = int(math.floor(tEnd/(d['dNt']*dtExp)))
	d['data']  = (2*Nx + 2*Nx)*8 # record Ex and Ey/z so multiplied by 2, plus grids
	d['physFreq'] = omega0
	d['physFreqName'] = 'ω_0'
	diag['strip'] = d

	# Probes
	d = {}
	d['name'] = 'particle probes'
	d['ekMin'] = 5*Te*const.k
	d['partProb'] = 1.0-scipy.stats.chi2.cdf(d['ekMin'],df=4,scale=0.5*const.k*Te)
	d['partRate'] = vth/math.sqrt(2.0*math.pi)*d['partProb']
	d['cellsFromBoundary'] = 5

	d['dNt'] = diag['reg']['dNt']
	d['N'] = int(math.floor(tEnd/(d['dNt']*dtExp)))
	d['data'] = 7*d['partRate']*ppce/(dx*dy)*2.*(Lx+Ly)*d['dNt']*dtExp*8
	diag['probe'] = d

	# Particles
	minV = (lwMinPhVel/vth - 1.0)*vth # This ignores relativity, may be > c
	if minV < 0.0: raise ValueError("Min LW velocity less than vth...")

	d = {}
	d['name'] = 'particles'
	d['exMin'] = 0.5*const.m_e*minV**2 # Calculate KE and use this to re-calculate velocity
	d['vxMin'] = const.c*math.sqrt(1.0-1.0/(1.0+d['exMin']/const.m_e/const.c**2)**2)
	#print('vxMinNR: {:}'.format(minV/vth))
	#print('vxMin: {:}'.format(d['vxMin']/vth))
	d['gammaMin'] = 1.0/math.sqrt(1.0-(d['vxMin']/const.c)**2)
	d['pxMin'] = d['gammaMin']*const.m_e*d['vxMin']
	d['partProb'] = 1.0-scipy.stats.norm.cdf(d['vxMin'],scale=vth)
	#print('partProb: {:}'.format(d['partProb']))
	d['partsExp'] = ppce*Nx*Ny*d['partProb']
	#print('partsExp: {:}'.format(d['partsExp']))

	d['dNt'] = diag['reg']['dNt']
	d['N'] = int(math.floor(tEnd/(d['dNt']*dtExp)))
	d['data'] = 5*d['partsExp']*8
	diag['ptcl'] = d

	totalData = 0.0
	for d in diag.values():
		if not all([ k in d.keys() for k in ['name','dNt','N','data']]):
			raise RuntimeError("Key missing from diagnostic spec")

		d['totData'] = d['data']*d['N']
		d['dt'] = d['dNt']*dtExp
		d['Nyq'] = math.pi/d['dt'] # Nyquist frequency as angular freq.
		d['dO'] = 2.0*math.pi/(d['N']*d['dt']) # Frequency resolution as angular frequency

		if d['name'] == 'single burst': continue
		totalData += d['totData']

	# Come up with simple estimate of CPU time required relative to the
	# reference homogeneous single-speckle simulation
	cellsVsYin = Nx*Ny/(2049.*257.)
	partsVsYin = ppc/512.
	timeStepsVsYin = nStepsExp/(9e-12/4.375789e-17)
	cpuTimeVsYin = timeStepsVsYin*cellsVsYin*partsVsYin

	outStr =  ['Output calculations:\n']
	outStr.append(' - Exp. num. steps     : {:.2f} (dt ~{:})'.format(nStepsExp,dtExp))
	outStr.append(' - End time            : {:.2f}ps = {:.2f}/ω0 = {:.2f}/ωp - {:.2f}/ωp'.format(tEnd/1e-12,tEnd*srsUtils.omegaNIF,tEnd*opMax,tEnd*opMin))
	outStr.append(' - Laser rise time     : {:.2f} ps'.format(riseTime/1e-12))
	outStr.append(' - Grid size           : {:.1f}x{:.1f}μm, {:}x{:} = {:} cells'.format(Lx/1e-6,Ly/1e-6,Nx,Ny,Nx*Ny))
	outStr.append(' - Grid resolution     : {:.1f}x{:.1f}nm'.format(dx/1e-9,dy/1e-9))
	outStr.append(' - Grid var data       : {:}'.format(sizeof_fmt(gridVarData)))
	outStr.append(' - Num particles       : {:} ({:} ppc)'.format(Nx*Ny*ppc,ppc))
	outStr.append(' - Particle var data   : {:}'.format(sizeof_fmt(partVarData)))
	outStr.append(' - Restart data        : {:}'.format(sizeof_fmt(restartData)))
	outStr.append(' - Steps accumulated   : {:}'.format(N_acc))
	outStr.append(' - Data accumulated    : {:}'.format(sizeof_fmt(N_acc*gridVarData*6)))
	outStr.append('')
	outStr.append('   Diagnostic   | Data/dmp |      Interval      | Number  | Tot data |')
	outStr.append('                |          | Nsteps  |   dt     |         |          |')
	outStr.append('----------------------------------------------------------------------')
	for d in diag.values():
		dataStr = sizeof_fmt(d['data'],separate=True)
		totDataStr = sizeof_fmt(d['totData'],separate=True)
		outStr.append(' {:<15.15}| {:>5}{:<3} |{:>8} | {:.2e} |{:>8} | {:>5}{:<3} |'.format(\
		              d['name'],dataStr[0],dataStr[1],
		              str(d['dNt']),d['dt'],str(d['N']),totDataStr[0],totDataStr[1]))

	outStr.append('----------------------------------------------------------------------')
	outStr.append('     TOTAL DATA |  {:<50}|'.format(sizeof_fmt(totalData)))
	outStr.append('----------------------------------------------------------------------')
	outStr.append('')
	outStr.append('   Diagnostic   |   Nyquist frequency    |          Δω               |')
	outStr.append('                |  ω /Hz   |  ω/ω_phys   |   /Hz    |                |')
	outStr.append('----------------------------------------------------------------------')
	for d in diag.values():
		if 'physFreq' not in d.keys(): continue
		outStr.append(' {:<15.15}| {:.2e} | {:.3f}{:<7.7} | {:.2e} | {:.2e}{:<7.7} |'.format(
		              d['name'],
					  d['Nyq'],d['Nyq']/d['physFreq'],d['physFreqName'],
					  d['dO'], d['dO']/d['physFreq'], d['physFreqName']))

	outStr.append('')
	outStr.append(' - Expected walltime vs. Yin (F/4) : {:.2f}x'.format(cpuTimeVsYin))
	outStr.append(' - Expected walltime   : {:.0f} core-hours = {:.0f}kAU'.format(9000.*cpuTimeVsYin,135.*cpuTimeVsYin))

	outStr = '\n'.join(outStr)
	print(outStr)

	# Set up input deck
	deck = pyEPOCH.inputDeck(2)

	boundaries = inputBlock(
		blockName = 'boundaries',

		bc_x_min_field    = 'simple_laser',
		bc_x_min_particle = 'thermal',
		bc_x_max_field    = 'open',
		bc_x_max_particle = 'thermal',
		bc_y_min_field    = y_bc_fld,
		bc_y_min_particle = y_bc_pcl,
		bc_y_max_field    = y_bc_fld,
		bc_y_max_particle = y_bc_pcl
	)

	control = inputBlock(
		blockName = 'control',

		# Domain Size
		x_min = xMin,
		x_max = xMax,
		y_min = yMin,
		y_max = yMax,

		# Domain Size
		nx = Nx,
		ny = Ny,
		npart = ppc*Nx*Ny,

		# Final time of simulation
		dt_multiplier = dt_mult,
		t_end = tEnd,

		stdout_frequency = 25,
		print_eta_string = True,
		max_accumulator_steps = diag['burst']['dNt']*N_acc + 1
	)

	electrons = inputBlock(
		blockName = 'species',

		name    = 'electrons',
		charge  = -1,
		mass    = 1.0,
		npart   = ppce*N,
		temp    = Te,
		density = eDensString
	)

	if ppci:
		carbon = inputBlock(
			blockName = 'species',

			name    = 'carbon',
			charge  = 6,
			mass    = 21895.0,
			npart   = ppci*N,
			temp    = Te/2.,
			density = cDensString
		)

		hydrogen = inputBlock(
			blockName = 'species',

			name    = 'hydrogen',
			charge  = 1,
			mass    = 1837.67,
			npart   = ppci*N,
			temp    = Te/2.,
			density = hDensString
		)

	laser = inputBlock({
		'blockName' : 'laser',

		'boundary'  : 'x_min',
		'id'        : 1,
		'intensity' : I,
		'lambda'    : wlVac0,
		'profile'   : ez_mod,
		'phase'     : ez_arg,
	#	'pol_angle' : math.pi/2.0,
		't_profile' : 'semigauss(time,1.0,{:},{:})'.format(initAmpFrac,wParam)
	})

	# Dummy subset to be applied to fields that are to be accumulated
	subset_allField = inputBlock(
		blockName = 'subset',

		name = 'field'
	)

	# Set up burst output blocks (if burst disabled won't be added to the deck)
	fieldOutput = [ inputBlock(
		blockName = 'output',

		name = 'burst{:}'.format(i),
		file_prefix = 'burst{:}_'.format(i),
		restartable = False,

		dump_first = False,
		dump_last  = False,

		nstep_accumulate = diag['burst']['dNt'],
		nstep_snapshot = diag['burst']['dNt_acc'],
		nstep_start = (i+1)*diag['aBurst']['dNt'],
		nstep_stop  = (i+1)*diag['aBurst']['dNt']+1,

		#ex = 'always',
		ex = 'always + accumulate + field'
	) for i in range(diag['aBurst']['N']) ]

	# Boundary field output
	fieldBounds = inputBlock(
		blockName = 'output',

		name = 'boundary',
		file_prefix = 'boundary_',
		restartable = False,

		dump_first = False,
		dump_last  = False,

		nstep_accumulate = diag['bound']['dNt'],
		nstep_snapshot = diag['bound']['dNt_acc'],
		nstep_start = 0,

		ex = 'always + x_min_ss + x_max_ss + y_min_ss + y_max_ss + accumulate',
		ey = 'always + x_min_ss + x_max_ss + y_min_ss + y_max_ss + accumulate',
		ez = 'always + x_min_ss + x_max_ss + y_min_ss + y_max_ss + accumulate',
		bx = 'always + x_min_ss + x_max_ss + y_min_ss + y_max_ss + accumulate',
		by = 'always + x_min_ss + x_max_ss + y_min_ss + y_max_ss + accumulate',
		bz = 'always + x_min_ss + x_max_ss + y_min_ss + y_max_ss + accumulate',
	)

	# Subset blocks defining above boundary output
	subset_x_min = inputBlock(
		blockName = 'subset',

		name = 'x_min_ss',
		x_min = xMin + diag['bound']['cellsFromBoundary']*dx,
		x_max = xMin + diag['bound']['cellsFromBoundary']*dx + 0.75*dx
	)

	subset_x_max = inputBlock(
		blockName = 'subset',

		name = 'x_max_ss',
		x_min = xMax - diag['bound']['cellsFromBoundary']*dx - 0.75*dx,
		x_max = xMax - diag['bound']['cellsFromBoundary']*dx
	)

	subset_y_min = inputBlock(
		blockName = 'subset',

		name = 'y_min_ss',
		y_min = yMin + diag['bound']['cellsFromBoundary']*dy,
		y_max = yMin + diag['bound']['cellsFromBoundary']*dy + 0.75*dy
	)

	subset_y_max = inputBlock(
		blockName = 'subset',

		name = 'y_max_ss',
		y_min = yMax - diag['bound']['cellsFromBoundary']*dy - 0.75*dy,
		y_max = yMax - diag['bound']['cellsFromBoundary']*dy
	)

	# Strip field output
	fieldStrip = inputBlock(
		blockName = 'output',

		name = 'strip_',
		file_prefix = 'strip_',
		restartable = False,

		dump_first = False,
		dump_last  = False,

		nstep_accumulate = diag['strip']['dNt'],
		nstep_snapshot = diag['strip']['dNt_acc'],
		nstep_start = 0,

		ex = 'always + strip_ss + accumulate',
		ey = 'always + strip_ss + accumulate',
		ez = 'always + strip_ss + accumulate',
		bx = 'always + strip_ss + accumulate',
		by = 'always + strip_ss + accumulate',
		bz = 'always + strip_ss + accumulate'
	)

	# Subset block for the field strip along the laser axis
	subset_strip = inputBlock(
		blockName = 'subset',

		name = 'strip_ss',
		y_min = 0.5*(yMin+yMax) - 0.25*dy + (1 - Ny%2)*0.5*dy,
		y_max = 0.5*(yMin+yMax) + 0.25*dy + (1 - Ny%2)*0.5*dy
	)

	# Set up regular output blocks
	if ppci:
		numDensMask = 'always + species'
	else:
		numDensMask = 'always'
	output = inputBlock(
		blockName = 'output',

		# number of timesteps between output dumps
		name           = 'regular',
		file_prefix    = 'regular_',
		restartable = False,

		nstep_snapshot = diag['reg']['dNt'],
		dump_first     = True,
		dump_last      = True,

		# Properties on grid
		grid             = 'always',
		number_density   = numDensMask,
		temperature      = 'always',
		ex               = 'always',
		ey               = 'always',
		bz               = 'always',
		total_energy_sum = 'always'
	)

	outProbes = inputBlock(
		blockName = 'output',

		# number of timesteps between output dumps
		name           = 'probes',
		file_prefix    = 'probes_',
		restartable = False,

		nstep_snapshot = diag['probe']['dNt'],
		dump_first     = False,
		dump_last      = True,

		particle_probes = 'always'
	)

	frontProbe = inputBlock(
		blockName = 'probe',

		name     = 'x_max',
		include_species = 'electrons',

		point    = '(x_max-{:}*dx,0.0)'.format(diag['probe']['cellsFromBoundary']),
		normal   = '(1.0,0.0)',

		ek_min   = diag['probe']['ekMin'],
		ek_max   = -1,
		dumpmask = 'always'
	)

	sideLProbe = inputBlock(
		blockName = 'probe',

		name     = 'y_max',
		include_species = 'electrons',

		point    = '(0.0,y_max-{:}*dy)'.format(diag['probe']['cellsFromBoundary']),
		normal   = '(0.0,1.0)',

		ek_min   = diag['probe']['ekMin'],
		ek_max   = -1,
		dumpmask = 'always'
	)

	sideRProbe = inputBlock(
		blockName = 'probe',

		name     = 'y_min',
		include_species = 'electrons',

		point    = '(0.0,y_min+{:}*dy)'.format(diag['probe']['cellsFromBoundary']),
		normal   = '(0.0,-1.0)',

		ek_min   = diag['probe']['ekMin'],
		ek_max   = -1,
		dumpmask = 'always'
	)

	backProbe = inputBlock(
		blockName = 'probe',

		name     = 'x_min',
		include_species = 'electrons',

		point    = '(x_min+{:}*dx,0.0)'.format(diag['probe']['cellsFromBoundary']),
		normal   = '(-1.0,0.0)',

		ek_min   = diag['probe']['ekMin'],
		ek_max   = -1,
		dumpmask = 'always'
	)

	restart = inputBlock(
		blockName = 'output',

		name = 'restart',
		file_prefix = 'restart_',
		nstep_snapshot = int(nStepsExp/20),
		dump_first = False,
		dump_last = True,
		restartable = True
	)

	# Subset block defining particles to dump
	# Restricts to positive px near phase velocity of LW
	subset_particles = inputBlock(
		blockName = 'subset',

		name = 'particle_ss',
		px_min = diag['ptcl']['pxMin'],
		include_species = 'electrons'
	)

	particleOutput = inputBlock(
		blockName = 'output',

		# number of timesteps between output dumps
		name           = 'particles',
		file_prefix    = 'particles_',
		restartable    = False,

		nstep_snapshot = diag['ptcl']['dNt'],
		dump_first     = False,
		dump_last      = False,

		# Particle properties to dump
		particle_grid   = 'particle_ss',
		px              = 'particle_ss',
		py              = 'particle_ss',
		particle_weight = 'particle_ss'
	)

	deck += boundaries
	deck += control
	deck += electrons
	if ppci:
		deck += hydrogen
		deck += carbon
	deck += output
	deck += subset_allField
	deck += fieldBounds
	deck += subset_strip
	deck += fieldStrip
	deck += subset_x_min
	deck += subset_x_max
	deck += subset_y_min
	deck += subset_y_max
	deck += restart
	deck += laser
	if particles:
		deck += subset_particles
		deck += particleOutput

	# Add optional diagnostics
	if burst:
		for o in fieldOutput: deck += o
	if probes:
		deck += frontProbe
		deck += sideLProbe
		deck += sideRProbe
		deck += backProbe
		deck += outProbes

	return deck

if(__name__ == '__main__'):
	import argparse
	import pbsUtils
	import sys
	import datetime

	parser = argparse.ArgumentParser()
	parser.add_argument('-n',type=float,required=True,nargs='+')
	parser.add_argument('--nCrit',action='store_true')
	parser.add_argument('-L','--densityScaleLength',type=float,required=True)
	parser.add_argument('-T',type=float,required=True)
	parser.add_argument('--keV',action='store_true')
	parser.add_argument('-I',type=float,required=True,default=0.01)
	parser.add_argument('--IUnit',type=str,choices=['wm2','wcm2','E','vOscOverC'],default='vOscOverC')
	parser.add_argument('--theta',type=float,default=0.0)
	parser.add_argument('-f','--omega0',type=float)
	parser.add_argument('-w','--wlVac0',type=float,default=srsUtils.wlVacNIF)
	parser.add_argument('-F',type=float,required=True)
	parser.add_argument('--planeWave',action='store_true')
	parser.add_argument('-t','--tEnd',type=float,required=True)
	parser.add_argument('--ppce',type=int,default=512)
	parser.add_argument('--ppci',type=int,default=0)
	parser.add_argument('--grid_res',type=float,default=1.2)
	parser.add_argument('--grid_res_y',type=float,default=None)
	parser.add_argument('--sLenFac',type=float)
	parser.add_argument('--nRange',type=float,nargs=2)
	parser.add_argument('--xBoundIntensFac',type=float,default=0.5)
	parser.add_argument('--yBoundIntensFac',type=float,default=0.01)
	parser.add_argument('--N_acc',type=int,default=1000)
	parser.add_argument('-b','--bufferWidth',type=float,required=True)
	parser.add_argument('-o','--outputDir',type=str)
	parser.add_argument('--onHPC',action='store_true',default=False)
	parser.add_argument('--dummy',action='store_true')
	parser.add_argument('-e','--executable')
	args = parser.parse_args()

	if args.omega0 == None:
		args.omega0 = const.c*2.0*math.pi/args.wlVac0

	if args.keV:
		args.T = args.T*1e3*const.e/const.k

	if args.nCrit:
		nCrit = const.m_e*const.epsilon_0/const.e**2*args.omega0**2
		args.n = np.array(args.n)*nCrit

	if args.IUnit == 'wcm2':
		args.I = args.I*1e4
	elif args.IUnit == 'E':
		args.I = srsUtils.intensity(args.I)
	elif args.IUnit == 'vOscOverC':
		args.I = srsUtils.intensity(const.m_e*args.omega0/const.e*args.I*const.c)

	if not args.grid_res_y:
		args.grid_res_y = args.grid_res

	print(args.grid_res_y)
	print('Requested simulation with following parameters:\n')
	print(' - ne: {ne} /m^3'.format(ne=args.n))
	print(' - Te: {Te} /K'.format(Te=args.T))
	print(' - I : {I} /Wcm^-2'.format(I=args.I/1e4))
	print(' - θ : {t}°'.format(t=args.theta))
	print(' - plane wave : '+str(args.planeWave))
	print(' - ω0: {} /Hz'.format(args.omega0))
	print(' - F : {F}\n'.format(F=args.F))

	deck = singleSpeckle(args.n,args.densityScaleLength,args.T,
	                     args.I,args.F,omega0=args.omega0,theta=args.theta,
	                     planeWave=args.planeWave,
	                     tEnd=args.tEnd,ppce=args.ppce,ppci=args.ppci,
	                     grid_res_x=args.grid_res,grid_res_y=args.grid_res_y,
	                     N_acc=args.N_acc,bufferWidth=args.bufferWidth,
	                     sLenFac=args.sLenFac,
						 xBoundIntensFac=args.xBoundIntensFac,yBoundIntensFac=args.yBoundIntensFac)


	runArgs = {'executable':args.executable}
	if args.onHPC:
		wallTime = 48*60*60

		# Request EPOCH to stop 15 minutes before end to write restart dump
		control = deck['control'][0]
		deck -= control
		control['stop_at_walltime'] = wallTime - 15*60
		deck += control

		rr = pbsUtils.resourceRequest(walltime=wallTime,pmem=3882,procs=392)
		runArgs['hpcResources'] = rr
		runArgs['daemon']       = True

		print("\nRequested submission as an HPC job")
		print("Using the following parameters:")
		print(rr)
		print("\nEPOCH set to stop at walltime {:}".format(
		      str(datetime.timedelta(seconds=control['stop_at_walltime']))))

	#print(deck)
	if args.dummy: sys.exit()

	deck.run(args.outputDir,**runArgs)
