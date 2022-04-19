# coding=UTF-8

import math as _math
import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.colors as _colors
import matplotlib.animation as _animation
from scipy.optimize import minimize as _minimize,brentq as _brentq
import scipy.constants as _const
import scipy.special as _special
import numba as _numba
import time as _time
import multiprocessing as _mp
from functools import partial as _partial

from srsUtils import langmuir as _langmuir
from srsUtils import misc as _misc
from srsUtils import filter as _filter

# Define useful constants
wlVacNIF = 351e-9
wnVacNIF = 2*_math.pi/wlVacNIF
omegaNIF = _const.c*wnVacNIF
nCritNIF = omegaNIF**2*_const.m_e*_const.epsilon_0/_const.e**2
keV      = 1e3*_const.e
TkeV     = keV/_const.k

@_numba.jit(nopython=True)
def dispRelEM(ne,k,d=0):
	omegap = _np.sqrt(ne/(_const.m_e*_const.epsilon_0))*_const.e
	K = _const.c/omegap*k
	if(d == 0):
		return omegap*_np.sqrt(1.0 + K**2)
	elif(d == 1):
		return _const.c*K/_np.sqrt(1.0 + K**2)
	elif(d == 2):
		return _const.c**2/omegap*_np.power(1.0 + K**2,-1.5)
	else:
		raise ValueError('Don\'t know how to calculate derivative of requested order :(')
		return 0.0

def grVelEM(ne,k):
	return _const.c**2*k/dispRelEM(ne,k)

def phVelEM(ne,k):
	return dispRelEM(ne,k)/k

def growthRateSRS(ne,Te,E,kEPW,omega0=omegaNIF):
	#vthSquare = _const.k*Te/_const.m_e
	plasmaFreq = _np.sqrt(ne*_const.e**2/(_const.m_e*_const.epsilon_0))
	omegaEPW = _langmuir.reOmega(ne,Te,kEPW)
	vOsc = _const.e*E/(_const.m_e*omega0)
	
	return 0.25*kEPW*plasmaFreq*vOsc/_np.sqrt(omegaEPW*(omega0-omegaEPW))

def matchWaves(k0,omega0,dispRel1,dispRel2,initk1=0.0,rtol=1e-13,maxiter=1000):
	'''
	Solves the three-wave matching conditions
	
	Description
	-----------

	Currently set up for SRS but could be extended to other instabilities.
	
	Output
	------
	
	Returns a list of solutions, with each solution containing the three
	wavenumbers that satisfy the matching conditions. These may be NaN if
	there is no solution.
	'''
	optFunc       = _np.vectorize(lambda k1: (dispRel1(k1*k0) + dispRel2((1.0-k1)*k0))/omega0 - 1.0)
	#cOptFunc      = lambda k1: _np.abs((dispRel1((k1[0]+1j*k1[1])*k0) + dispRel2((1.0-(k1[0]+1j*k1[1]))*k0))/omega0 - 1.0)**2
	
	try:
		kf = _brentq(optFunc,0.0,1.0)*k0
	except ValueError:
		#print("Regular root finding failed")
		#print(kf)
		#print(optFunc(kf))
#		ks = _np.linspace(0,2)
#		y  = optFunc(ks)
#		_plt.plot(ks,y)
#		#_plt.plot([kf/k0,kb/k0],[optFunc(kf/k0),optFunc(kb/k0)],'o')
#		_plt.grid()
#		_plt.show()
		kf = float('NaN')
	
	try:
		kb = _brentq(optFunc,1.0,2.0)*k0
	except ValueError:
#		print("Regular root finding failed")
#		kb = _minimize(cOptFunc,_np.array([1.5,0.1]))
#		print(kb)
		kb = float('NaN')
		
	
	solutions = [[k0,kb,k0-kb],[k0,kf,k0-kf]]
	
	#_plt.plot(_np.linspace(0,2),optFunc(_np.linspace(0,2)))
	#_plt.plot([kf/k0,kb/k0],[optFunc(kf/k0),optFunc(kb/k0)],'o')
	#_plt.grid()
	#_plt.show()
	
#	print("matchWaves")
#	print(_np.abs(solutions[0][1] + solutions[0][2] - solutions[0][0])/k0)
#	print(_np.abs(solutions[1][1] + solutions[1][2] - solutions[1][0])/k0)
#	print(_np.abs(optFunc(solutions[0][1]/k0)))
#	print(_np.abs(optFunc(solutions[1][1]/k0)))
#	print(kf/k0)
#	print(kb/k0)
	
	#assert _np.abs(solutions[0][1] + solutions[0][2] - solutions[0][0])/k0 <= rtol
	#assert _np.abs(solutions[1][1] + solutions[1][2] - solutions[1][0])/k0 <= rtol
	#assert _np.abs(optFunc(solutions[0][1]/k0)) <= rtol
	#assert _np.abs(optFunc(solutions[1][1]/k0)) <= rtol
	
	return solutions

def srsWNs(ne,Te,omega0=omegaNIF,order=None):
	b = _np.broadcast(ne,Te,omega0)
	#print(b.shape)
	matchParams = [ (_np.sqrt(o**2 - (n*_const.e**2/(_const.m_e*_const.epsilon_0)))/_const.c,
	                 o,
	                 lambda k,ne=n,Te=T: _langmuir.reOmega(ne,Te,k,order=order),
	                 lambda k,ne=n     : dispRelEM(ne,k,0)) for (n,T,o) in b ]
	#print(matchParams)
	
	x = [ matchWaves(*i) for i in matchParams ]
	#print(x)
	
	wns = { 'k0'  : _np.empty(b.shape),
	        'kb'  : _np.empty(b.shape),
	        'ksb' : _np.empty(b.shape),
	        'kf'  : _np.empty(b.shape),
	        'ksf' : _np.empty(b.shape) }
	#print(x)
	
	wns['k0'].flat  = [ ks[0][0] for ks in x ]
	wns['kb'].flat  = [ ks[0][1] for ks in x ]
	wns['ksb'].flat = [ ks[0][2] for ks in x ]
	#assert(_np.all((wns['kb'] + wns['ksb'] - wns['k0'])/wns['k0'] < 1e-14))
	
	wns['kf'].flat  = [ ks[1][1] for ks in x ]
	wns['ksf'].flat = [ ks[1][2] for ks in x ]
	#assert(_np.all((wns['kf'] + wns['ksf'] - wns['k0'])/wns['k0'] < 1e-14))
	#print(wns)
	
	return wns

def srsOmegas(ne,Te,omega0=omegaNIF,order=None):
	wns = findWNs(ne,Te,omega0,order=order)
	omegas = {'k0':dispRelEM(ne,wns['k0']),
	          'kb':_langmuir.reOmega(ne,Te,wns['kb'],order=order),
	          'kf':_langmuir.reOmega(ne,Te,wns['kf'],order=order),
	          'ksb':dispRelEM(ne,wns['ksb']),
	          'ksf':dispRelEM(ne,wns['ksf'])}
	
	return omegas

def growthRateSRS2(ne,Te,E,kEPW,omega0,deltaOmega=0.0):
	matchGrowth = growthRateSRS(ne,Te,E,kEPW,omega0)
	newGrowth = _np.sqrt(matchGrowth**2-0.25*deltaOmega**2)
	
	return newGrowth

def srsEPWBeta(n,T,omega0,E):
	'''
	Calculates normalised SRS EPW damping rate β
	
	As defined in Forslund et. al., PoP 18, 1007 (1975).
	β < 2 implies absolute growth.
	'''
	wns = srsWNs(n,T,omega0)
	op = _np.sqrt(n/(_const.m_e*_const.epsilon_0))*_const.e
	
	# Growth and damping rate of SRS and plasma wave
	g0 = growthRateSRS(n,T,E,wns['kb'],omega0)
	gp = _langmuir.imOmega(n,T,wns['kb'])
	
	# Group velocity of EPW and scattered EMW
	vp = _langmuir.groupVel(n,T,wns['kb'])
	vs = _const.c**2*wns['ksb']/_np.sqrt(op**2 + (_const.c*wns['ksb'])**2)
	
	b = gp/g0*_np.sqrt(_np.abs(vs/vp))
	#print(b)
	return b

def srsAbsThreshIntens(n,T,omega0=omegaNIF):
	''' Calculates the threshold intensity for SRS absolute growth '''
	wns    = srsWNs(n,T,omega0)
	omegas = findOmegas(n,T,omega0)
	
	op = _np.sqrt(n/(_const.m_e*_const.epsilon_0))*_const.e
	gp = _langmuir.imOmega(n,T,wns['kb'])
	vp = _langmuir.groupVel(n,T,wns['kb'])
	vs = grVelEM(n,wns['ksb'])
	
	vrat = _np.abs(vs/vp)
	orat = _np.sqrt(omegas['ksb']*omegas['kb'])/op
	
	E = 2.0*(_const.m_e/_const.e)*orat*omega0*gp/wns['kb']*_np.sqrt(vrat)
	
	return intensity(E)

def srsAbsThreshIntens2(nt,omega0=omegaNIF):
	return srsAbsThreshIntens(nt[0],nt[1],omega0)

# Reproduce figure from Winjum thesis (pg. 18)
# TODO: Fix colorbar normalisation
def plotAbsThresh(numTemps,numDens,vthLims=(0.01,0.1),nencrLims=(0.025,0.2)):
	import multiprocessing as mp
	import functools
	
	wlVac0 = 351e-9
	omega0 = _const.c*2.*_math.pi/wlVac0
	nCrit  = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	
	vths   = _np.linspace(vthLims[0],vthLims[1],numTemps)*_const.c
	nencrs = _np.linspace(nencrLims[0],nencrLims[1],numDens)

	nes    = nencrs*nCrit	
	temps  = vths**2*_const.m_e/_const.k
	
	Ns,Ts = _np.meshgrid(nes,temps)
	ns = Ns.flatten()
	ts = Ts.flatten()
	nts = zip(ns,ts)
	
	# Calculate threshold intensities
	pool = mp.Pool()
	threshIntens = pool.map(functools.partial(srsAbsThreshIntens2,omega0=omega0),nts)
	threshIntens = _np.reshape(_np.array(threshIntens),Ns.shape)/1e4
	pool.close()
	#threshIntens = srsAbsThreshIntens(Ns,Ts,omega0)/1e4
	print(threshIntens)
	
	extent = _misc.getExtent(nencrs,vths/_const.c)
	cs = _plt.contour(threshIntens,extent=extent,levels=[1e14,1e15,1e16],_colors='k')
	formatFunc = lambda x: '$10^{{{:}}}$'.format(int(_math.floor(_math.log10(x))))
	_plt.clabel(cs,fmt=formatFunc)
	
	_plt.title('Absolute threshold intensity')
	im = _plt.imshow(threshIntens,extent=extent,aspect='auto',origin='lower',interpolation='none',norm=_colors.SymLogNorm(linthresh=1e9, vmax=_np.max(threshIntens)),cmap='viridis')
	im.cmap.set_under('k')
	_plt.colorbar(im)
	
	ax = _plt.gca()
	ax.set_xlabel('$n_e/n_{\mathrm{cr}}$')
	ax.set_ylabel('$\\frac{v_{\mathrm{th}}}{c}$',rotation=0)
	_plt.show()

def findWNs(ne,Te,omega0=omegaNIF,order=1):
	return srsWNs(ne,Te,omega0,order=order)
	
#	vthSquare = _const.k*Te/_const.m_e
#	omegap = _np.sqrt(ne*_const.e**2/(_const.m_e*_const.epsilon_0))
#	wns = {'k0':_np.sqrt(omega0**2 - omegap**2)/_const.c}
#	
#	maxNumIterations = 10000
#	maxError = 1e-15
#	
#	for sType in ['f','b']:
#		x0 = 0.0
#		numIterations = 0
#		while(True):
#			A = 3*vthSquare - 3*omega0/omegap*vthSquare*(1+x0**2)**(-0.5) - _const.c**2
#			B = 2*_const.c**2*wns['k0']
#			C = omega0**2 - 2*omega0*omegap*(1 + x0**2)**(0.5) + omega0*omegap*(1+x0**2)**(-0.5)*x0**2 - _const.c**2*wns['k0']**2
#			
#			if  (sType == 'f'): k = (-B + _np.sqrt(B**2 - 4*A*C))/(2*A)
#			elif(sType == 'b'): k = (-B - _np.sqrt(B**2 - 4*A*C))/(2*A)
#	
#			x = k*_np.sqrt(3*vthSquare)/omegap
#		
#			#print "Iteration:",numIterations+1,"Error:",x**2-x0**2
#		
#			if(_np.all(_np.abs(x**2-x0**2) < maxError)):
#				break
#	
#			if(numIterations > maxNumIterations):
#				raise RuntimeError("Error: Failed to converge on solution")
#		
#			x0 = x
#			numIterations += 1
#		
#		wns['k'+sType] = k
#		wns['ks'+sType] = wns['k0'] - k
#	
#	return wns

def findOmegas(ne,Te,omega0=omegaNIF,order=1):
	return srsOmegas(ne,Te,omega0,order=order)

def findWNsAnalytic(ne,Te,omega0=omegaNIF,ver=2):
	vthSquare = _const.k*Te/_const.m_e
	omegap = _np.sqrt(ne*_const.e**2/(_const.m_e*_const.epsilon_0))
	k0 = _np.sqrt(omega0**2-omegap**2)/_const.c
	
	if(ver == 1):
		kBack = k0 + omega0/_const.c*_np.sqrt(1.0-2.0*omegap/omega0)
		kForw = k0 - omega0/_const.c*_np.sqrt(1.0-2.0*omegap/omega0)
	else:
		a = 1.0+3.0*vthSquare/_const.c**2*(omega0/omegap-1.0)
		b = -2.0*k0
		c = k0**2 + omega0**2/_const.c**2*(2.0*omegap/omega0 - 1.0)
		
		kForw = (-b-_np.sqrt(b**2-4.0*a*c))/(2.0*a)
	
	ks = k-k0
	
	kArray = _np.array([k0,k,ks]).transpose()
	return kArray

def selfFocusingRate(ne,Te,E,omega0=omegaNIF):
	'''
	Temporal growth rate for the ponderomotive self-focusing instability

	From Kruer, "The Physics of Laser-Plasma Interactions", pg. 93
	'''
	vos = _const.e*E/(_const.m_e*omega0)
	vth = _np.sqrt(_const.k*Te/_const.m_e)

	op = _np.sqrt(ne*_const.e**2/(_const.m_e*_const.epsilon_0))
	
	return (vos/vth)**2*op**2/(8.0*omega0)

def selfFocusingSpatialRate(ne,Te,E,omega0=omegaNIF):
	'''
	**Spatial** growth rate of the ponderomotive self-focusing instability

	From Montgomery, PoP (2016)
	DOI: 10.1063/1.4946016
	'''
	vos = _const.e*E/(_const.m_e*omega0)
	vth = _np.sqrt(_const.k*Te/_const.m_e)
	ncr = _const.m_e*_const.epsilon_0/_const.e**2*omega0**2

	return (vos/vth)**2*(ne/ncr)*omega0/_const.c

def selfFocusingThreshold(ne,Te,L,omega0=omegaNIF):
	'''
	Threshold intensity for the ponderomotive self-focusing instability

	Threshold is where gain is unity over length L, G = g*L where g is the
	spatial growth rate

	From Montgomery, PoP (2016)
	DOI: 10.1063/1.4946016
	'''
	vth = _np.sqrt(_const.k*Te/_const.m_e)
	ncr = _const.m_e*_const.epsilon_0/_const.e**2*omega0**2

	return 4.0/L*(_const.m_e/_const.e)**2*ncr/ne*_const.c**2*omega0*vth**2*_const.epsilon_0

def intensity(E,derivative=False):
	cf = 0.5*_const.c*_const.epsilon_0

	if(derivative):
		return 2.0*cf*E
	else:
		return cf*E**2

def intensityToEField(intensity):
	'''
	Converts intensity to an electric field amplitude

	Intensity given in w/m^2
	'''
	cf = 2.0/(_const.c*_const.epsilon_0)
	
	return _np.sqrt(cf*intensity)

def collisionFreq_ee(v,n,T):
	'''
	Electron-electron collision frequency

	Assumes a Maxwellian background electron distribution

	Parameters
	==========

	- v: velocity of test electron
	- n: density of background electrons
	- T: temperature of background electrons
	'''
	x = 0.5*_const.m_e*v**2/(_const.k*T)
	psiFac = _special.erf(_np.sqrt(x)) - 2.0/_math.sqrt(_math.pi)*_np.exp(-x)*_np.sqrt(x)

	ncc = 1e-6*n
	Tev = T*_const.k/_const.e
	cLog = 23.5-_np.log(_np.sqrt(ncc)*_np.power(Tev,-5./4)) \
	       - _np.sqrt(1e-5 + (_np.log(Tev)-2.0)**2/16.)
	fac = _const.e**4/(4.0*_math.pi*_const.epsilon_0**2)*cLog*n/(_const.m_e**2*v**3)

	return 2.0*fac*psiFac

def collisionFreq_ei(v,n,T,Z,mi):
	'''
	Electron -> ion collision frequency

	Assumes a Maxwellian background ion distribution

	Parameters
	==========

	- v: velocity of test electron
	- n: density of background ions
	- T: temperature of background ions
	- Z: charge state of ions
	- mi: ion mass
	'''
	x = 0.5*_const.m_e*v**2/(_const.k*T)
	psiFac = _special.erf(_np.sqrt(x)) - 2.0/_math.sqrt(_math.pi)*_np.exp(-x)*_np.sqrt(x)

	ncc = 1e-6*n
	Tev = T*_const.k/_const.e
	cLog = 23.0-_np.log(_np.sqrt(ncc)*_np.power(Tev,-5./4)) \
	       - _np.sqrt(1e-5 + (_np.log(Tev)-2.0)**2/16.)
	fac = _const.e**4/(4.0*_math.pi*_const.epsilon_0**2)*cLog*n/(_const.m_e**2*v**3)

	return 2.0*fac*psiFac

def collisionFreq(ne,Te,Zeff=1.0):
	'''
	Electron-ion collision frequency for Maxwellian electron and ion distributions

	All quantities in SI.

	Zeff defined as <Z²>/<Z>

	From NRL plasma formulary 2016, pg. 33
	Zeff factor included in Fien thesis, 2017
	'''
	TeV = Te/TkeV*1e3
	necm = ne/1e6
	
	lei = 24.-_np.log(_np.sqrt(necm)/TeV)
	return 2.9e-6*necm*_np.power(TeV,-1.5)*Zeff*lei

def calcGrowthRate(ts,ys,eps=0.05,diag=False,smooth=False,
	               filtLen=None,filtCutoff=None):
	'''
	Calculates the exponential growth rate from a time series

	The signal should be of form y(t) ~ exp(iωt+γt) so that |y(t)| ~ exp(γt).
	Attempts to measure a growth rate based on finding a continuous region with
	constant gradient of the function f(t) = ln(y). The procedure for deciding
	whether the gradient is constant is described in comments below.
	'''
	dt = ts[-1]-ts[-2]

	logy = _np.log(_np.abs(ys))

	# Apply smoothing filter to log of signal if requested
	if smooth and (filtLen is not None and filtCutoff is not None):
		# Apply Gaussian filter
		oNyq = _math.pi/dt
		coeffs = _filter.gaussianFilter(filtLen,filtCutoff/oNyq)
		logy = _np.convolve(coeffs,logy,mode='valid')
		ts = ts[filtLen/2:-(filtLen/2)]
		ys = ys[filtLen/2:-(filtLen/2)]
	
	Nt = ts.shape[0]

	# Take 1st and 2nd numerical derivatives of the log of the signal
	logGrad = _np.gradient(logy,dt)
	logGrad2 = _np.gradient(logGrad,dt)
	
	minInd = 2
	maxInd = Nt-2
	
	# Calculate threshold on derivative of growth rate
	# Reasoning:
	#
	# Take B(t) = log(A(t)) ~ at^2 + bt + c
	# Then:
	#     B'(t) ~ 2at + b
	#     B"(t) ~ 2a
	#
	# If data is vaguely linear over interval Δt then:
	#     |aΔt^2| < |εbΔt|
	#
	# Where |ε| << 1
	#
	# So need:
	#     |a| ~ |B"(t)/2| < |εb/Δt|
	#
	#     => |B"(t)| < |2εb/Δt|
	b  = _np.mean(logGrad[minInd:maxInd])
	Dt = ts[maxInd]-ts[minInd]
	maxGrad2 = _np.abs(eps*2*b/Dt)
	if diag: print("2nd derivative threshold: {:.3e}s**-2 = {:.3e}ω_0**2".format(maxGrad2,maxGrad2/omegaNIF**2))

	# Find longest continuous section of values where the second derivative is
	# within the threshold
	seq = _np.abs(logGrad2[:maxInd]) < maxGrad2
	result = _misc.maxLenContigSubseqIdxs(seq)
	
	minInd = result[1]
	maxInd = result[2]

	# If no part of the signal shows linear growth, or the segment is too short
	# then return zero, otherwise take the mean and standard deviation of the
	# growth rate
	if (result[0] == 0) or (ts[maxInd-1]-ts[minInd] < 0.05*Dt):
		if diag: print("No linear growth phase identified")
		meanGrad = 0.0
		stdGrad = 0.0
	else:
		meanGrad = _np.mean(logGrad[minInd:maxInd])
		stdGrad = _np.std(logGrad[minInd:maxInd])
	
	# If requested plot diagnostics
	if diag:
		print('γ = {:.5e}±{:.5e} Hz'.format(meanGrad,stdGrad))
		print('γ = {:.5f}±{:.5f} ω_0'.format(meanGrad/omegaNIF,stdGrad/omegaNIF))
		freqs = 2.0*_math.pi*_np.fft.fftshift(_np.fft.fftfreq(Nt,dt))
		
		fig,axes = _plt.subplots(3,2)
		
		eNorm = (_const.m_e*_const.c*omegaNIF)/_const.e
		tNorm = 2.0*_math.pi/omegaNIF
		
		tWindow = _np.hamming(Nt)
		ampNorm = 2./_np.sum(tWindow)
		fft = lambda x: ampNorm*_np.abs(_np.fft.fftshift(_np.fft.fft(tWindow*x)))
		
		axes[0][0].plot(ts/tNorm,_np.abs(ys)/eNorm)
		axes[0][0].set_ylabel(r'$e|E|/m_e\omega_0c$')
		
		axes[0][1].plot(freqs/omegaNIF,fft(_np.log(_np.abs(ys)/eNorm)))
		
		#print(logGrad/omegaNIF)
		axes[1][0].plot(ts/tNorm,logGrad/omegaNIF)
		axes[1][0].set_ylabel(r"$\frac{1}{\omega_0}\frac{|E|'}{|E|}$")
		if result[0] != 0:
			axes[1][0].axhline(meanGrad/omegaNIF,color='k')
			y1 = (meanGrad+stdGrad)/omegaNIF
			y2 = (meanGrad-stdGrad)/omegaNIF
			axes[1][0].axhspan(y1,y2,edgecolor='k',facecolor='g',alpha=0.4)
	
		axes[1][1].plot(freqs/omegaNIF,fft(logGrad)/omegaNIF)
		
		axes[2][0].plot(ts/tNorm,logGrad2/omegaNIF**2)
		axes[2][0].set_ylabel(r'$\frac{\mathrm{d}^2}{\mathrm{d}t^2}\ln{|E|}$')
		axes[2][0].axhspan(-maxGrad2/omegaNIF**2,maxGrad2/omegaNIF**2,
		                   edgecolor='k',facecolor='g',alpha=0.4)
		yMax = _np.max(_np.abs(axes[2][0].get_ylim()))
		axes[2][0].set_ylim(-yMax,yMax)
	
		axes[2][1].plot(freqs/omegaNIF,fft(logGrad2)/omegaNIF**2)
	
		for ax in axes[:,0]:
			ax.set_xlim(ts[0]/tNorm,ts[-1]/tNorm)
			ax.set_xlabel(r'$\omega_0t/2\pi$')
			if result[0] != 0:
				print(ts[minInd]/tNorm)
				print(ts[maxInd-1]/tNorm)
				ax.axvline(ts[minInd]/tNorm,color='k',linestyle='--')
				ax.axvline(ts[maxInd-1]/tNorm,color='k',linestyle='--')
	
		for ax in axes[:,1]:
			ax.set_xlim(freqs[0]/omegaNIF,freqs[-1]/omegaNIF)
			ax.set_xlabel('$\omega/\omega_0$')
			ax.set_yscale('log')
	
		for ax in axes.flatten():
			ax.grid()
		
		axes[0][0].set_yscale('log')
		#axes[1].set_yscale('log')
		
		_plt.show()

	return meanGrad,stdGrad

def plotCollisionFreq(ax,neLims=(0.01*nCritNIF,0.25*nCritNIF),Tes=(2.0*TkeV,),Zeff=1.0):
	nes = _np.linspace(neLims[0],neLims[1],200)
	cFreqs = [ collisionFreq(nes,Te) for Te in Tes ]

	for f,Te in zip(cFreqs,Tes):
		ax.plot(nes/nCritNIF,1./f/1e-12,label=r'${:.1f}$keV'.format(Te/TkeV))
	
	ax.set_xlabel(r'$n_e/n_{\mathrm{cr}}$')
	ax.set_ylabel(r'$\nu_{ei}^{-1}$ /ps')
	ax.grid()

	return ax

def printProperties(n,T,omega0=omegaNIF):
	plasmaFreq = _math.sqrt(n*_const.e**2/(_const.m_e*_const.epsilon_0))
	vth = _math.sqrt(_const.k*T/_const.m_e)
	debyeLength = _math.sqrt(_const.epsilon_0*_const.k*T/(n*_const.e**2))
	eField = _const.m_e*omega0/_const.e*0.01*_const.c
	
	wns = findWNs(n,T,omega0)
	#print(wns)
	#wns[1] = findWNsAnalytic(n,T,omega0,ver=2)
	wls = { wn:2*_math.pi/wns[wn] for wn in wns }
	omegas = findOmegas(n,T,omega0)
	ts = { wn:2*_math.pi/omegas[wn] for wn in wns }

	wlRatios = { wn:abs(wls[wn])/wls['k0'] for wn in wls }
	tRatios = { wn:abs(ts[wn])/ts['k0'] for wn in ts }
	
	print("Laser properties:")
	print("ω:      {:}rad s^-1".format(omega0))
	print("k_vac:  {:}rad m^-1".format(omega0/_const.c))
	print("λ_vac:  {:}nm".format(2*_math.pi/(omega0/_const.c)/1e-9))
	print("n_cr:   {:}m^-3".format(omega0**2*_const.m_e*_const.epsilon_0/_const.e**2))
	print("n_cr/4: {:}m^-3".format(0.25*omega0**2*_const.m_e*_const.epsilon_0/_const.e**2))

	print("\nPlasma properties:")
	print("Tₑ            : {:}keV / {:}K".format(T*_const.k/_const.e*1e-3,T))
	print("nₑ            : {:}m^-3".format(n))
	print("n/n_cr        : {:}".format(n/(omega0**2*_const.m_e*_const.epsilon_0/_const.e**2)))
	print("λ_D           : {:}m".format(debyeLength))
	print("4/3πnₑλ_D³    : {:}".format(4./3.*_math.pi*n*(debyeLength**3)))
	print("ωₚ            : {:}s^-1".format(plasmaFreq))
	print("vₜₕ           : {:}ms^-1 / {:}".format(vth,vth/_const.c))
	print("E (v~0.01c)   : {:e}Vm^-1".format(eField))
	print("I (v~0.01c)   : {:}Wm^-2".format(eField**2*_const.c*_const.epsilon_0/2))
	print("I (v~0.01c)   : {:}Wcm^-2\n".format(eField**2*_const.c*_const.epsilon_0/2/1e4))
	_misc.printCols([["","bSRS","fSRS"],
	                ["kλ_D",wns['kb']*debyeLength,wns['kf']*debyeLength],
	                ["LW wavebreaking E",_langmuir.waveBreakLim(n,T,wns['kb']),
	                                     _langmuir.waveBreakLim(n,T,wns['kf'])    ],
	                ["γ (0.01c):",growthRateSRS(n,T,eField,wns['kb'],omega0),
	                              growthRateSRS(n,T,eField,wns['kf'],omega0) ],
	                ["1/γ",1.0/growthRateSRS(n,T,eField,wns['kb'],omega0),
	                       1.0/growthRateSRS(n,T,eField,wns['kf'],omega0)    ],
	                ["ω₀/2πγ",omega0/(2*_math.pi*growthRateSRS(n,T,eField,wns['kb'],omega0)),
	                          omega0/(2*_math.pi*growthRateSRS(n,T,eField,wns['kf'],omega0)) ],
	                ["ωₚ/2πγ",plasmaFreq/(2*_math.pi*growthRateSRS(n,T,eField,wns['kb'],omega0)),
	                          plasmaFreq/(2*_math.pi*growthRateSRS(n,T,eField,wns['kf'],omega0))]])
	
	fStrings = ['k0','kf','ksf']
	bStrings = ['k0','kb','ksb']
	
	print("\nSolution 1 (fSRS):")
	_misc.printCols([[" ","EMW1","EPW","EMW2"],
		            ["k /rad m^-1"] + [ wns[k] for k in fStrings ],
		            ["λ /nm"] + [ wls[k]/1e-9 for k in fStrings ],
		            ["ω /rad s^-1"] + [ omegas[k] for k in fStrings ],
		            ["T /fs"] + [ ts[k]/1e-15 for k in fStrings ],
		            ["λ/λ₀"] + [ wlRatios[k] for k in fStrings ],
		            ["T/T₀"] + [ tRatios[k] for k in fStrings ],
		            ["v_φ"] + [ omegas[k]/wns[k] for k in fStrings],
		            ["v_g"] + [ grVelEM(n,wns['k0']),_langmuir.groupVel(n,T,wns['kf']),grVelEM(n,wns['ksf'])]])
	
	print("\nω₀-(ω₁+ω₂): {:}".format(omegas['k0']-(omegas['kf']+omegas['ksf'])))
	print("k₀-(k₁+k₂): {:}".format(wns['k0']-(wns['kf']+wns['ksf'])))
	print("ω(k₂)-ω₂  : {:}".format(_langmuir.reOmega(n,T,wns['kf'],order=1) - omegas['kf']))
	print("ω(k₃)-ω₃  : {:}".format(dispRelEM(n,wns['ksf']) - omegas['ksf']))

	print("\nSolution 2 (bSRS):")
	_misc.printCols([[" ","EMW1","EPW","EMW2"],
		            ["k /rad m^-1"] + [ wns[k] for k in bStrings ],
		            ["λ /nm"] + [ wls[k]/1e-9 for k in bStrings ],
		            ["ω /rad s^-1"] + [ omegas[k] for k in bStrings ],
		            ["T /fs"] + [ ts[k]/1e-15 for k in bStrings ],
		            ["λ/λ₀"] + [ wlRatios[k] for k in bStrings ],
		            ["T/T₀"] + [ tRatios[k] for k in bStrings ],
		            ["v_φ"] + [ omegas[k]/wns[k] for k in bStrings],
		            ["v_g"] + [ grVelEM(n,wns['k0']),_langmuir.groupVel(n,T,wns['kb']),grVelEM(n,wns['ksb'])]])

	print("\nω₀-(ω₁+ω₂):",omegas['k0']-(omegas['kb']+omegas['ksb']))
	print("k₀-(k₁+k₂): {:}".format(wns['k0']-(wns['kb']+wns['ksb'])))
	print("ω(k₂)-ω₂  : {:}".format(_langmuir.reOmega(n,T,wns['kb'],order=1) - omegas['kb']))
	print("ω(k₃)-ω₃  : {:}".format(dispRelEM(n,wns['ksb']) - omegas['ksb']))
	
	#k = _np.linspace(-1.2*k2,1.2*k2,100)
	#emDisp = _np.array(map(lambda k:dispRelEM(n,k),k))
	#epwDisp = _np.array(map(lambda k:dispRelEPW(n,T,k),k))

	#_plt.plot(k,emDisp,k,epwDisp)

	#_plt.grid()
	#_plt.show()

def plotRatios(n1,n2,T,omega0=omegaNIF,numNs=100,log=True):
	if(log):
		ns = _np.logspace(_np.log10(n1),_np.log10(n2),numNs)
	else:
		ns = _np.linspace(n1,n2,numNs)
	
	ratioList = []
	for n in ns:
		wns = findWNs(n,T,omega0)
		wls = 2*_math.pi/_np.array(wns)
		omegas = _np.array([[dispRelEM(n,wns[0][0]),_langmuir.reOmega(n,T,wns[0][1],order=1),dispRelEM(n,wns[0][2])],
				           [dispRelEM(n,wns[1][0]),_langmuir.reOmega(n,T,wns[1][1],order=1),dispRelEM(n,wns[1][2])]])
		ts = 2*_math.pi/_np.array(omegas)

		wlRatios = [ abs(i)/i[0] for i in wls ]
	
		ratioList.append(wlRatios)
	
	plasmaFreqs = _np.sqrt(ns*_const.e**2/(_const.m_e*_const.epsilon_0))
	EPWwnsAnalytic = findWNsAnalytic(ns,T,omega0,ver=1)[:,1]
	EPWwnsAnalytic2 = findWNsAnalytic(ns,T,omega0,ver=2)[:,1]
	EPWratiosAnalytic = (_np.sqrt(omega0**2 - plasmaFreqs**2)/_const.c)/EPWwnsAnalytic
	EPWratiosAnalytic2 = (_np.sqrt(omega0**2 - plasmaFreqs**2)/_const.c)/EPWwnsAnalytic2
	#print EPWratiosAnalytic
	#print EPWratiosAnalytic2
	EPWratios = [ r[0][1] for r in ratioList ]
	EM2ratios = [ r[0][2] for r in ratioList ]
	
	backEPWratios = [ r[1][1] for r in ratioList ]
	backEM2ratios = [ r[1][2] for r in ratioList ]
	
	nCrit = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	ncritRatio = ns/nCrit
	
	fig = _plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ncritRatio,EPWratios,label='fSRS EPW')
	ax.plot(ncritRatio,EM2ratios,label='fSRS EMW')
	ax.plot(ncritRatio,backEPWratios,label='bSRS EPW')
	ax.plot(ncritRatio,backEM2ratios,label='bSRS EMW')
	#ax.plot(ncritRatio,EPWratiosAnalytic,label='bSRS EPW analytic')
	ax.plot(ncritRatio,EPWratiosAnalytic2,label='bSRS EPW analytic2')
	#_plt.axhline(2.0,linestyle='--')
	ax.axvline(0.25,linestyle='--',color='k')
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.set_xlim(ax.get_xlim()[0],0.5)
	ax.set_xlim(n1/nCrit,0.5)
	ax.set_ylim(0.5,30.0)
	#ax.set_ylim(0.5,1.0)
	ax.set_xlabel(r'$n_e/n_c\ (n_c = '+_misc.floatToLatexScientific(nCrit)+'\ /\mathrm{m}^{-3})$')
	ax.set_ylabel(r'$\lambda/\lambda_0$')

	ax.legend(loc='upper center',prop={'size':7},ncol=2)
	ax.grid()
	return fig

def plotWns(n1,n2,T,omega0=omegaNIF,numNs=100,log=True,ratio=True):
	if(log):
		ns = _np.logspace(_np.log10(n1),_np.log10(n2),numNs)
	else:
		ns = _np.linspace(n1,n2,numNs)
	
	wns    = srsWNs(ns,T,omega0)
	ratios = { wn:wns[wn]/wns['k0'] for wn in wns }
	#print(ratios['ksb'])
	
	nCrit = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	ncritRatio = ns/nCrit
	
	fig = _plt.figure()
	ax = fig.add_subplot(111)
	if(ratio):
		ax.plot(ncritRatio,ratios['kb'],label='bSRS EPW')
		ax.plot(ncritRatio,_np.abs(ratios['ksb']),label='bSRS EMW')
		ax.plot(ncritRatio,ratios['kf'],label='fSRS EPW')
		ax.plot(ncritRatio,ratios['ksf'],label='fSRS EMW')
	else:
		ax.plot(ncritRatio,wns['kb'],label='bSRS EPW')
		ax.plot(ncritRatio,_np.abs(wns['ksb']),label='bSRS EMW')
		ax.plot(ncritRatio,wns['kf'],label='fSRS EPW')
		ax.plot(ncritRatio,wns['ksf'],label='fSRS EMW')
		ax.plot(ncritRatio,wns['k0'],label='$k_0$')

	#_plt.axhline(2.0,linestyle='--')
	ax.axvline(0.25,linestyle='--',color='k')
	
	if(log):
		ax.set_xscale('log')
		ax.set_yscale('log')
		#ax.set_xlim(ax.get_xlim()[0],0.5)
		ax.set_xlim(ncritRatio[0],0.5)
		#ax.set_ylim(6e-2,3.0)
	else:
		ax.set_xlim(0,0.3)
		#ax.set_ylim(6e-2,3.0)
	ax.set_xlabel(r'$n_e/n_c\ (n_c = '+_misc.floatToLatexScientific(nCrit)+'\ /\mathrm{m}^{-3})$')
	if(ratio):
		ax.set_ylabel(r'$k/k_0$')
	else:
		ax.set_ylabel(r'$k$ /m$^{-1}$')

	_plt.legend(loc='lower left',fontsize=8,ncol=2)
	ax.grid()
	return fig

def plotOmegas(n1,n2,T,omega0=omegaNIF,numNs=100,log=True):
	if(log):
		ns = _np.logspace(_np.log10(n1),_np.log10(n2),numNs)
	else:
		ns = _np.linspace(n1,n2,numNs)
	
	wns = findWNs(ns,T,omega0)
	omegas = findOmegas(ns,T,omega0)
	
	ratios = { wn:omegas[wn]/omegas['k0'] for wn in wns }
	#print(ratios['ksb'])
	
	nCrit = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	ncritRatio = ns/nCrit
	
	fig = _plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ncritRatio,ratios['kb'],label='bSRS EPW')
	ax.plot(ncritRatio,_np.abs(ratios['ksb']),label='bSRS EMW')
	ax.plot(ncritRatio,ratios['kf'],label='fSRS EPW')
	ax.plot(ncritRatio,ratios['ksf'],label='fSRS EMW')

	#_plt.axhline(2.0,linestyle='--')
	ax.axvline(0.25,linestyle='--',color='k')
	if(log):
		ax.set_xscale('log')
		ax.set_yscale('log')
		#ax.set_xlim(ax.get_xlim()[0],0.5)
		ax.set_xlim(ncritRatio[0],0.5)
		#ax.set_ylim(6e-2,3.0)
	else:
		#ax.set_xlim(ax.get_xlim()[0],0.5)
		ax.set_xlim(0.0,0.3)
		#ax.set_ylim(6e-2,3.0)
	ax.set_xlabel(r'$n_e/n_c\ (n_c = '+_misc.floatToLatexScientific(nCrit)+'\ /\mathrm{m}^{-3})$')
	ax.set_ylabel(r'$\omega/\omega_0$')

	ax.legend(loc='lower left',fontsize=8,ncol=2)
	ax.grid()
	return fig

def plotGrowthRates(n1,n2,T,omega0,E=None,numNs=100,log=True):
	if(log):
		ns = _np.logspace(_np.log10(n1),_np.log10(n2),numNs)
	else:
		ns = _np.linspace(n1,n2,numNs)
	
	# If user hasn't specified E field, calculate it such that it gives an
	# electron quiver velocity of 0.01c.
	if(E == None):
		E = _const.m_e*omega0/_const.e*0.01*_const.c
	
	wns = findWNs(ns,T,omega0)
	growthRates = { 'bSRS':growthRateSRS(ns,T,E,wns['kb'],omega0),
	                'fSRS':growthRateSRS(ns,T,E,wns['kf'],omega0) }
	
	nCrit = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	ncritRatio = ns/nCrit
	
	fig = _plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ncritRatio,growthRates['bSRS'],label='bSRS')
	ax.plot(ncritRatio,growthRates['fSRS'],label='fSRS')
	ax = _plt.gca()
	ax.set_xlabel(r'$n_e/n_c\ (n_c = '+_misc.floatToLatexScientific(nCrit)+'\ /\mathrm{m}^{-3})$')
	ax.set_ylabel(r'$\gamma_{\mathrm{SRS}}\ /\mathrm{s}^{-1}$')
	ax.axvline(0.25,linestyle='--',color='k')
	if(log):
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlim(6e-3,0.5)
		ax.set_ylim(1e11,3e13)
	else:
		ax.set_xlim(0.0,0.3)
		ax.set_ylim(0.0,ax.get_ylim()[1])
	_plt.legend(fontsize=10,ncol=2,loc='lower center')
	ax.grid()
	#_plt.show()
	
#	_plt.gcf().set_size_inches(plotWidthIn,1/1.4*plotWidthIn)
#	_plt.tight_layout()
#	_plt.savefig('./srsGrowthRate.pdf')
	return fig

def plotkLambdaD(n1,n2,T,omega0,numNs=100,log=True):
	if(log):
		ns = _np.logspace(_np.log10(n1),_np.log10(n2),numNs)
	else:
		ns = _np.linspace(n1,n2,numNs)
	
	wns = _np.zeros((2,numNs))
	for i,n in enumerate(ns):
		wns[0][i] = findWNs(n,T,omega0)['kb']
		wns[1][i] = findWNs(n,T,omega0)['kf']
	
	debyeLengths = _np.sqrt(_const.epsilon_0*_const.k*T/(ns*_const.e**2))
	kLambdaDs = wns*debyeLengths
	
	nCrit = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	ncritRatio = ns/nCrit
	
	fig = _plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ncritRatio,kLambdaDs[0],label='bSRS')
	ax.plot(ncritRatio,kLambdaDs[1],label='fSRS')
	ax.axvline(0.25,linestyle='--',color='k')
	ax = _plt.gca()
	ax.set_xlabel(r'$n_e/n_c\ (n_c = '+_misc.floatToLatexScientific(nCrit)+'\ \mathrm{m}^{-3})$')
	ax.set_ylabel(r'$k\lambda_D$')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(3.6e-3,0.5)
	ax.set_ylim(1e-2,2)
	ax.axhspan(0.2,ax.get_ylim()[1], color='grey', alpha=0.5,lw=1.0)
	ax.text(0.5,0.8,'strong damping',horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	ax.legend(loc='lower center',prop={'size':10},ncol=2)
	ax.grid()
	return fig

def plotLandauDamping(kld1,kld2,T,n,numNs=100,log=True):
	if(log):
		klds = _np.logspace(_np.log10(kld1),_np.log10(kld2),numNs)
	else:
		klds = _np.linspace(kld1n1,kld2,numNs)
	
	lambdaD = _np.sqrt(_const.k*T*_const.epsilon_0/(n*_const.e**2))
	kEPWs = klds/lambdaD
	
	#print klds
	dampingRates = landauDampingRateMaxwell(n,T,kEPWs)
	#print dampingRates
	dampingRates = dampingRates/max(dampingRates)
	
	fig = _plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(klds,dampingRates)
	ax.set_xlabel(r'$k\lambda_D$')
	ax.set_ylabel(r'Normalised damping rate')
	ax.set_ylim(0,1.1)
	ax.set_xscale('log')
	ax.grid()
	return fig

def plotkLandauDampingVsn(n1,n2,T,omega0,numNs=100,log=True):
	if(log):
		ns = _np.logspace(_np.log10(n1),_np.log10(n2),numNs)
	else:
		ns = _np.linspace(n1,n2,numNs)
	
	dampingRates = _np.zeros((2,numNs))
	for i,n in enumerate(ns):
		wns = findWNs(n,T,omega0)
		dampingRates[0][i] = landauDampingRateMaxwell(n,T,wns['kb'])
		dampingRates[1][i] = landauDampingRateMaxwell(n,T,wns['kf'])
	
	nCrit = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	ncritRatio = ns/nCrit
	
	fig = _plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ncritRatio,dampingRates[0],label='bSRS')
	ax.plot(ncritRatio,dampingRates[1],label='fSRS')
	ax.axvline(0.25,linestyle='--',color='k')
	ax = _plt.gca()
	ax.set_xlabel(r'$n_e/n_c\ (n_c = '+_misc.floatToLatexScientific(nCrit)+'\ /\mathrm{m}^{-3})$')
	ax.set_ylabel(r'$\gamma$')
	#ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.set_xlim(3.6e-3,0.5)
	#ax.set_ylim(1e-2,2)
	#ax.legend()
	ax.grid()
	return fig

#@_numba.jit(nopython=True)
def ratioDiffSum(n,T,omega0,numLasWLs=6,bSRS=True):
	if(bSRS):
		wns = findWNs(n,T,omega0)
		wns = _np.array([wns['k0'],wns['kb'],wns['ksb']])
	else:
		wns = findWNs(n,T,omega0)
		wns = _np.array([wns['k0'],wns['kf'],wns['ksf']])
	
	#print(wns)
	wls = 2*_math.pi/wns
	
	#print(wls)
	wlRatios = numLasWLs*wls[0]/wls
	roundDiff = _np.abs(_np.round(wlRatios)-wlRatios)
	
	sumRatios = _np.sum(roundDiff,axis=0)
		
	ncr = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	#print(n/ncr)
	#print(sumRatios)
	return sumRatios
	
def optimiseRatios(n1,n2,T,omega0,bSRS=True):
	a = n2-n1
	b = n1
	
	n0Scal = 0.5
	n1Scal = 0.0
	n2Scal = 1.0
	
	minFunc = lambda nScal: ratioDiffSum(a*nScal+b,T,omega0,bSRS=bSRS)

	result = _minimize(minFunc,n0Scal,method='L-BFGS-B',bounds=[(n1Scal,n2Scal)])
	
#	nRange = _np.linspace(n1Scal,n2Scal)
#	_plt.plot(nRange,minFunc(nRange))
#	_plt.grid()
#	_plt.show()
	
	#print(a*result['x'][0]+b)
	return a*result['x'][0]+b

def plotRatioDiffSum(fig,ax,n1,n2,T,omega0,numNs=100,log=True):
	if(log):
		ns = _np.logspace(_np.log10(n1),_np.log10(n2),numNs)
	else:
		ns = _np.linspace(n1,n2,numNs)
	
	rds = { 'bSRS': ratioDiffSum(ns,T,omega0,bSRS=True),
	        'fSRS': ratioDiffSum(ns,T,omega0,bSRS=False) }
	
	nCrit = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	ncritRatio = ns/nCrit
	
	ax.plot(ncritRatio,rds['bSRS'],label='bSRS')
	ax.plot(ncritRatio,rds['fSRS'],label='fSRS')
	
	ax.set_xlabel(r'$n_e/n_c\ (n_c = '+_misc.floatToLatexScientific(nCrit)+'\ /\mathrm{m}^{-3})$')
	#ax.set_ylabel(r'$\gamma_{\mathrm{SRS}}\ /\mathrm{s}^{-1}$')
	
	ax.axvline(0.25,linestyle='--',color='k')
	
	if(log):
		ax.set_xscale('log')
		#ax.set_yscale('log')
		ax.set_xlim(6e-3,0.5)
		ax.set_ylim(0.0,1.0)
	else:
		ax.set_xlim(0.0,0.3)
		ax.set_ylim(0.0,1.0)
		#ax.set_ylim(0.0,ax.get_ylim()[1])
	
	fig.legend(fontsize=10,ncol=2,loc='upper center')
	
	ax.grid()
	
	return fig

def plotRatioDiffSum2D(fig,ax,n1,n2,T1,T2,omega0=omegaNIF,numNs=100,numTs=100,log=True):
	nCrit = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	if(log):
		ns = _np.logspace(_np.log10(n1),_np.log10(n2),numNs)
		Ts = _np.logspace(_np.log10(T1),_np.log10(T2),numTs)
		dn = (_np.log10(n2)-_np.log10(n1))/(numNs-1)
		dT = (_np.log10(T2)-_np.log10(T1))/(numTs-1)
		extent=[_np.power(10.,_np.log10(n1)-0.5*dn)/nCrit,
		        _np.power(10.,_np.log10(n2)+0.5*dn)/nCrit,
		        _np.power(10.,_np.log10(T1)-0.5*dT)/TkeV,
				_np.power(10.,_np.log10(T2)+0.5*dT)/TkeV]
	else:
		ns = _np.linspace(n1,n2,numNs)
		Ts = _np.linspace(T1,T2,numTs)
		dn = (n2-n1)/(numNs-1)
		dT = (T2-T1)/(numTs-1)
		extent=_np.array([(n1-0.5*dn)/nCrit,(n2+0.5*dn)/nCrit,
		                 (T1-0.5*dT)/TkeV,(T2+0.5*dT)/TkeV])
	
	print(extent)
	
	nMesh,TMesh = _np.meshgrid(ns,Ts)
	
	print(Ts/TkeV)
	rds = { 'bSRS': ratioDiffSum(nMesh,TMesh,omega0,bSRS=True),
	        'fSRS': ratioDiffSum(nMesh,TMesh,omega0,bSRS=False) }
	
	nCrit = omega0**2*_const.m_e*_const.epsilon_0/_const.e**2
	ncritRatio = ns/nCrit
	
	im = ax.imshow(rds['bSRS'],aspect='auto',extent=extent,origin='lower',cmap=_colormaps.plasma,interpolation='none',vmin=0.0,vmax=1.0)
	cb = fig.colorbar(im,orientation='horizontal',ax=ax)
	ax.contour(nMesh/nCrit,TMesh/TkeV,rds['bSRS'],levels=[0.05],colors='k')
	
	ax.set_xlim(extent[0],extent[1])
	ax.set_ylim(extent[2],extent[3])
	ax.set_xlabel(r'$n_e/n_c\ (n_c = '+_misc.floatToLatexScientific(nCrit)+'\ /\mathrm{m}^{-3})$')
	ax.set_ylabel(r'$T_e$ /KeV')
	#ax.set_yscale('log')
	cb.set_label(r'Mismatch')
	
	ax.axvline(0.25,linestyle='--',color='k')
	
	return fig,ax

