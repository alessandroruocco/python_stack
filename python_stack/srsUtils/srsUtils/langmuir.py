#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import scipy.special as special
from scipy import stats
from scipy.optimize import brentq,basinhopping,brute,minimize_scalar
from scipy.integrate import ode
from scipy import misc
from scipy import signal
import sympy as sp
import numba
import time

from srsUtils import srsUtils
from srsUtils import misc
from srsUtils import sympyStuff

friedConteZeros = np.array([
	1.9914668430e0 - 1.3548101281e0j,
	2.6911490243e0 - 2.1770449061e0j,
	3.2353308684e0 - 2.7843876132e0j,
	3.6973097025e0 - 3.2874107894e0j
])

def deriv(f,x0,N=101,dx=1e-4):
	x = np.linspace(x0-(N-1)/2*dx,x0+(N-1)/2*dx,N)
	y = f(x)
	
	grad = np.polyfit(x,y,1)[0]
	return grad
deriv = np.vectorize(deriv)

# SI wrappers to scale the main dimensionless functions
# Output ω where wave is of form e^{i*(ωt-kx)}

def omega(n,T,k,order=None):
	'''
	Calculates complex LW frequency in SI units
	
	Parameters
	----------

	n,T,k : Density, electron temperature and LW wavenumber (all SI)
	order : Order of the Taylor expansion used. If 'not order == True' then the
	        solution will be calculated numerically.
	
	Output
	------

	Returns a complex frequency ω. Wave has form e^{i*(ωt-kx)}.
	'''
	op = np.sqrt(n/(const.m_e*const.epsilon_0))*const.e
	ld = np.sqrt(const.k*T*const.epsilon_0/n)/const.e
	K  = k*ld
	
	return op*omegaNodim(K,order)
	
def reOmega(n,T,k,order=None):
	''' Calculates LW frequency in SI units. See omega function for details. '''
	op = np.sqrt(n/(const.m_e*const.epsilon_0))*const.e
	ld = np.sqrt(const.k*T*const.epsilon_0/n)/const.e
	K  = k*ld
	
	return op*reOmegaNodim(K,order)

def imOmega(n,T,k,order=None):
	''' Calculates LW damping rate in SI units. See omega function for details. '''
	op = np.sqrt(n/(const.m_e*const.epsilon_0))*const.e
	ld = np.sqrt(const.k*T*const.epsilon_0/n)/const.e
	K  = k*ld
	
	return op*imOmegaNodim(K,order)
	
# Dimensionless functions
# Output ω where wave is of form e^{i*(ωt-kx)}

def omegaNodim(K,order=None):
	'''
	Calculates complex LW frequency in dimensionless units
	
	Parameters
	----------

	K     : Dimensionless wavenumber K=kλ_D
	order : Order of the Taylor expansion used. If 'not order == True' then the
	        solution will be calculated numerically.
	
	Output
	------

	Returns a complex frequency Ω=ω/ω_pe. Wave has form e^{i*(ωt-kx)}.
	'''
	if    not order: return numOmegaNodim(K)
	elif order == 1: return analReOmegaNodimO1(K) + 1j*analImOmegaNodimO1(K)
	elif order == 3: return analReOmegaNodimO3(K) + 1j*analImOmegaNodimO3(K)
	else: raise ValueError('Don\'t know how to calculate EPW of requested order :(')

def reOmegaNodim(K,order=None):
	''' Calculates LW frequency in dimensionless units. '''
	if    not order: return np.real(numOmegaNodim(K))
	elif order == 1: return analReOmegaNodimO1(K)
	elif order == 3: return analReOmegaNodimO3(K)
	else: raise ValueError('Don\'t know how to calculate EPW of requested order :(')

def imOmegaNodim(K,order=None):
	''' Calculates LW damping rate in dimensionless units. '''
	if     not order: return np.imag(numOmegaNodim(K))
	elif  order == 1: return analImOmegaNodimO1(K)
	elif  order == 3: return analImOmegaNodimO3(K)
	else: raise ValueError('Don\'t know how to calculate EPW of requested order :(')

@numba.jit(nopython=True)
def analReOmegaNodimO1(K):
	return np.sqrt(1.0 + 3.0*K**2)

@numba.jit(nopython=True)
def analReOmegaNodimO3(K):
	a = [3.0,6.0,24.0]
	return np.sqrt(1.0 + a[0]*K**2 + a[1]*K**4 + a[2]*K**6)

@numba.jit(nopython=True)
def analImOmegaNodimO1(K):
	return math.sqrt(math.pi/8)*(1./K**3)*np.exp(-(0.5/K**2 + 1.5))

@numba.jit(nopython=True)
def analImOmegaNodimO3(K):
	return math.sqrt(math.pi/8)*(1./K**3 - 6.*K)*np.exp(-(0.5/K**2 + 1.5 + 3.*K**2 + 12.*K**4))

def numOmegaNodim(K):
	'''
	Uses friedConteRoot to calculate the complex frequency of an EPW.
	
	Tested up to K = 1e4, fails somewhere between 1e4 and 1e5
	
	Parameters
	----------
	
	K : Normalised wavenumber (K == kλ_D)
	
	Returns
	-------
	
	Omega : Normalised EPW frequency in units of the plasma frequency
	'''
	
	if(K != 0.0):
		return math.sqrt(2.)*np.abs(K)*np.conj(friedConteRoot(2.*K**2))
	else:
		return 1.0+0.0*1j
numOmegaNodim = np.vectorize(numOmegaNodim,otypes=[np.complex128])

def omegaDeriv(n,T,k,order=None):
	vth = np.sqrt(const.k*T/const.m_e)
	ld = np.sqrt(const.k*T*const.epsilon_0/n)/const.e
	K  = k*ld
	
	return vth*omegaDerivNodim(K,order)

def omegaDerivNodim(K,order=None):
	if     not order: return numOmegaDerivNodim(K)
	elif  order == 1: return analOmegaDerivNodimO1(K)
	elif  order == 3: return analOmegaDerivNodimO3(K)
	else: raise ValueError('Don\'t know how to calculate EPW of requested order :(')

@numba.jit(nopython=True)
def analOmegaDerivNodimO1(K):
	return 3.*K/np.sqrt(1.0 + 3.0*K**2)

@numba.jit(nopython=True)
def analOmegaDerivNodimO3(K):
	return (3.0*K + 12.*K**3 + 72.*K**5)/analReOmegaNodimO3(K)

def numOmegaDerivNodim2(K):
	return deriv(lambda x: np.real(numOmegaNodim(x)),K,N=101,dx=1e-4)

def numOmegaDerivNodim(K):
	'''
	Calculates the group velocity of a Langmuir wave

	Makes use of the implicit function theorem to avoid having to numerically
	differentiate the LW dispersion relation.
	'''
	o = numOmegaNodim(K)
	
	# Write this way so as to keep sign of imaginary component when K < 0
	z = np.real(o)/(math.sqrt(2.0)*K) + 1j*np.imag(o)/np.abs(math.sqrt(2.0)*K)

	fd1  = friedConte(z,d=1)
	fd2  = friedConte(z,d=2)
	dfo  = -fd2/(math.sqrt(2.0)*K)**3
	dfk  = 0.5*(2.0*fd1 + z*fd2)/K**3

	# Not sure why we have to take modulus but seems to work...
	return np.sign(K)*np.abs(dfk/dfo)

def numOmegaDerivNodimTest():
	Ks = np.linspace(-1.0,1.0,100)
	os = numOmegaNodim(Ks)
	print(os)
	zs = np.real(os)/(math.sqrt(2.0)*Ks) + 1j*np.imag(os)/np.abs(math.sqrt(2.0)*Ks)
	print(zs)

	fd1  = friedConte(zs,d=1)
	fd2  = friedConte(zs,d=2)
	dfo  = -fd2/(math.sqrt(2.0)*Ks)**3
	dfk  = 0.5*(2.0*fd1 + zs*fd2)/Ks**3

	dfoInv = 1.0/dfo

	fig,ax = plt.subplots(3,3)

	ax[0][0].set_title(r'$\Re\left[\frac{\partial D}{\partial K}\right]$')
	ax[0][0].plot(Ks,np.real(dfk))
	ax[0][1].set_title(r'$\Im\left[\frac{\partial D}{\partial K}\right]$')
	ax[0][1].plot(Ks,np.imag(dfk))
	ax[0][2].set_title(r'$\left|\frac{\partial D}{\partial K}\right|$')
	ax[0][2].plot(Ks,np.abs(dfk))
	ax[1][0].set_title(r'$\Re\left[\frac{\partial D}{\partial \omega}^{-1}\right]$')
	ax[1][0].plot(Ks,np.real(dfoInv))
	ax[1][1].set_title(r'$\Im\left[\frac{\partial D}{\partial \omega}^{-1}\right]$')
	ax[1][1].plot(Ks,np.imag(dfoInv))
	ax[1][2].set_title(r'$\left|\frac{\partial D}{\partial \omega}^{-1}\right|$')
	ax[1][2].plot(Ks,np.abs(dfoInv))
	ax[2][0].set_title(r'$\Re\left[\frac{\partial D}{\partial K}\right]$')
	ax[2][0].plot(Ks,np.sign(Ks)*np.abs(dfk/dfo))

	for a in ax.flatten():
		a.set_xlabel('$K$')
		a.grid()
	plt.tight_layout()
	plt.show()

# group velocity aliases
groupVel            = omegaDeriv
groupVelNodim       = omegaDerivNodim
analGroupVelNodimO1 = analOmegaDerivNodimO1
analGroupVelNodimO3 = analOmegaDerivNodimO3
numGroupVelNodim    = numOmegaDerivNodim

def Dn(K,O,n):
	'''
	Calculates the nth ω derivative of the Langmuir wave dispersion function D

	Description
	-----------

	The LW dispersion function is defined as D = 1+Χ_e, where Χ_e is the
	kinetic electron susceptibility for a thermal distribution.

	This function calculates its nth partial derivative w.r.t. ω.
	'''
	z = O/(math.sqrt(2.0)*K)

	zD = friedConte(z,d=n+1)

	dn = -zD/(math.sqrt(2.0)*K)**(2+n)
	if n == 0: dn += 1.0
	
	return dn

def symbD(oDeriv,kDeriv):
	O,K,z = sp.symbols('O,K,z')
	ZsF = sp.symbols('Zf:{:}'.format(oDeriv+kDeriv+1),cls=sp.Function)
	Zs = sp.symbols('Z:{:}'.format(oDeriv+kDeriv+1))

	d = 1-ZsF[0](O/sp.sqrt(2)/K)/(2*K**2)
	
	dD = d.diff(O,oDeriv).diff(K,kDeriv)

	for i,Z in enumerate(ZsF):
		if i==0: continue
		ZD = ZsF[0](z).diff(z,i).subs(z,O/sp.sqrt(2)/K)
		dD = dD.replace(ZD,Z(O/sp.sqrt(2)/K))
	
	for i,Z in enumerate(ZsF):
		dD = dD.replace(Z(O/sp.sqrt(2)/K),Zs[i])
	
	return (O,K,)+Zs[oDeriv:],dD

_D = sympyStuff.genFuncHandler(symbD,2)

def D(O,K,oDeriv,kDeriv):
	''' Calculates the required partial derivative wrt ω or k of 1+X_e '''
	z = O/(math.sqrt(2.0)*K)
	Zs = [ friedConte(z,i+1) for i in range(oDeriv,oDeriv+kDeriv+1) ]

	return _D(oDeriv,kDeriv,O,K,*Zs)

def chiEKinetic(omega,k,ne,Te):
	omegapSquared = ne*const.e**2/(const.m_e*const.epsilon_0)
	vthSquared = const.k*Te/const.m_e
	ld = np.sqrt(vthSquared/omegapSquared)
	#print("In X_EKin:, ω: {:}".format(omega))
	return -friedConte(omega/(math.sqrt(2.0)*k*np.sqrt(vthSquared)),d=1)/(2.0*k**2*ld**2)

def chiEKineticNodim(O,K):
	return -friedConte(O/(math.sqrt(2.0)*K),d=1)/(2.0*K**2)

def epsInv(O,K):
	return K**2/(K**2 - 0.5*friedConte(O/(math.sqrt(2.0)*K),d=1))

def friedConte(z,d=0):
	'''
	The Fried-Conte plasma dispersion function and derivatives

	Derivatives calculated using the associated differential equation, see
	pg. 30 of NRL plasma formulary

	Parameters
	==========

	z: argument of the function, in general a complex number
	d: Which derivative to calculate, d=0 is the 0th derivative
	'''
	W = special.wofz(z)
	return friedConteMain(z,W,d)

@numba.jit(nopython=True)
def friedConteMain(z,W,d=0):
	if(d==0):
		return 1j*math.sqrt(math.pi)*W
	else:
		ZDprev = 1j*math.sqrt(math.pi)*W
		ZDcurr = -2.0*(1.0+z*ZDprev)
		for i in range(2,d+1):
			ZDnext = -2.0*((i-1)*ZDprev + z*ZDcurr)
			ZDprev = ZDcurr
			ZDcurr = ZDnext

		return ZDcurr

# Alternative definition using dawson function instead (gives seemingly identical results)
#def friedConte2(z,d=0):
#	#print("In friedConte, z: {:}, d: {:}".format(z,d))
#	
#	if(d==0):
#		return 1j*math.sqrt(math.pi)*np.exp(-z**2) - 2*special.dawsn(z)
#	elif(d == 1):
#		return -2.0*(1.0+z*(1j*math.sqrt(math.pi)*np.exp(-z**2) - 2*special.dawsn(z)))

def friedConteFindImZero(zr):
	'''
	Finds first zi at which Im(Z'(zr + i*zi)) = 0, beginning at zi=0
	
	Note: fails for large zr due to limited precision of friedConte() function
	'''
	f = lambda zi: np.imag(friedConte(zr+1j*zi,d=1))
	
	zi = 0.0
	dz = 0.5
	while(f(zi) < 0.0):
		zi -= dz
		#print(zi)
	
	#print(zi+dz)
	#print(zi)
	
	ziNew = brentq(f,zi+dz,zi)
	#if(ziNew == 0.0):
	#	print('zr: {:}, z1: {:}, z2 {:}, f1: {:}, f2: {:}'.format(zr,zi+dz,zi,f(zi+dz),f(zi)))
	return ziNew

def friedConteRoot(A):
	'''
	Finds the solution to Z'(z) = A, where A∊ℝ > 0 and z∊ℂ, Re(z) > 0
	
	The solution found should be the one with the largest Im(z) (in general Im(z) < 0)
	'''
	if A <= 0:
		return float('NaN')
	#assert(A > 0)
	
	# For A < 0.02 more accurate to use analytic approximation to solution
	if(A < 0.02):
		zr = np.sqrt(1/A + 1.5 + 1.5*A + 3.0*A**2)
		zi = friedConteFindImZero(zr)
		if(zi == 0.0): zi = -(imOmegaNodim(math.sqrt(A/2.),order=3)*zr)/reOmegaNodim(math.sqrt(A/2.),order=3)
		z = zr+1j*zi
	elif(A > 1e4):
		z = -1j*np.sqrt(special.lambertw(A**2/(8.*math.pi),1)/2.)
	# For A >= 0.02 we can get a more accurate numerical result
	elif(A > 0.1):
		z = followRoot(A)
	else:
		# Estimate initial Re(z):
		zr0 = np.real(-1j*np.sqrt(special.lambertw(A**2/(8.*math.pi),1)/2.))
		#zr0 = 1.0/math.sqrt(A) + 0.75*math.sqrt(A)
	
		minFunc = lambda zr: abs(np.real(friedConte(zr + 1j*friedConteFindImZero(zr),d=1)) - A)
	
		#print(minFunc(zr0))
	
		t1 = time.time()
		ub = 7.2
		brtol = 1e-3
		result = minimize_scalar(minFunc,[zr0,zr0+1.],bounds=[0.0,ub],method='Bounded')
		if(abs(result['x']-ub)/ub < brtol):
			raise RuntimeError('Minimizer wants to go past upper bound')	
		#print(result)
		
		zr = result['x']
		zi = friedConteFindImZero(zr)
		if(zi == 0.0): zi = -(imOmegaNodim(math.sqrt(A/2.),order=3)*zr)/reOmegaNodim(math.sqrt(A/2.),order=3)
		z = zr+1j*zi
	#print('elapsed: {:}'.format(time.time()-t1))
	#print(result)
	
	return z

def friedConteRoot2(A,n=0):
	'''
	Finds the solution to Z'(z) = A, where A∊ℝ > 0 and z∊ℂ, Re(z) > 0
	
	The solution found is the nth branch (the n=0th branch gives the EPW
	dispersion relation)
	'''
	if n == 0:
		z = followRoot(A)
	else:
		z = followRoot(A,zInit=friedConteZeros[n-1],AInit=2.0)
	
	return z

def dzdA(A,z):
	return 1.0/friedConte(z,d=2)

def jac(A,z):
	'''
	Jacobian function as used below, makes integration marginally faster as
	fewer function evaluations are needed overall.
	'''
	return -friedConte(z,d=3)/friedConte(z,d=2)**2

def followRoot(A,zInit=1.6828928524910822-0.4020808541999312j,AInit=1.0):
	'''
	Finds the solution to Z'(z) = A, where A∊ℝ > 0 and z∊ℂ, Re(z) > 0

	This is done using the implicit function theorem to find dz/dA:

	   dz   ∂Z'(z) ⎛∂Z'(z)⎞⁻¹
	   ―― = ―――――― ⎜――――――⎟
	   dA     ∂A   ⎝  ∂z  ⎠

	   dz     1
	=> ―― = ―――――
	   dA   Z"(z)
	
	Since this equation is of the form dy/dx = f(x,y) it is an ODE and may be
	solved using standard numerical techniques.
	'''
	# Easier to use ZVODE rather than VODE as it is a complex valued ODE we are
	# solving.
	integrator = ode(dzdA,jac=jac).set_integrator('zvode')
	
	integrator.set_initial_value(zInit,AInit)
	
	# Do numerical integration up to A
	integrator.integrate(A)
	
	return integrator.y[0]

followRoot = np.vectorize(followRoot)

def waveBreakLim(ne,Te,k):
	'''
	Electron plasma wave breaking amplitude
	
	From Kruer pg. 103 (which is from Coffey)
	'''
	vPh = reOmega(ne,Te,k)/k
	vth = np.sqrt(const.k*Te/const.m_e)
	plasmaFreq = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
	
	b = 3.0*vth**2/vPh**2
	X = np.sqrt(1.0+2.0*np.sqrt(b) - 8.0/3.0*np.power(b,0.25) - b/3.0)
	
	return X*const.m_e*plasmaFreq*vPh/const.e
	
def plotFriedConte(fig,ax,zrRange,ziRange,nzr,nzi,overlay=False,noTitle=False):
	'''
	Plots the plasma dispersion function of Fried and Conte in the complex plane
	'''
	# Define range of z values and calculate friedConte function
	zr = np.linspace(zrRange[0],zrRange[1],nzr)
	zi = np.linspace(ziRange[0],ziRange[1],nzi)
	Zr,Zi = np.meshgrid(zr,zi)
	y = friedConte(Zr+1j*Zi,d=1)

	y0 = np.real(y)
	y1 = np.imag(y)
	y2 = np.abs(y)

	if overlay:
		# Pick range of kl_D values and calculate expected zrs and zis using 1st
		# and 3rd order formulae
		Ks = math.sqrt(2)*np.logspace(-5,7,1000)
		
		zsCalc = np.array([ friedConteRoot(K**2) for K in Ks ])
	
		zrs1 = np.sqrt(1/Ks**2 + 1.5)
		zrs2 = np.sqrt(1/Ks**2 + 1.5 + 1.5*Ks**2 + 3.0*Ks**4)
		zrs3 = 1./Ks + 0.75*Ks# + 15./24.*Ks**3 + 147./128.*Ks**5
		zis1 = -math.sqrt(math.pi/8)*(math.sqrt(2)*2./Ks**4)*np.exp(-(1./Ks**2 + 1.5))
		zis2 = -math.sqrt(math.pi/8)*(math.sqrt(2)*2./Ks**4 - 3.*math.sqrt(2))*np.exp(-(1./Ks**2 + 1.5 + 1.5*Ks**2 + 3.*Ks**4))
		zis3 = zis1
		s = 2.
		zs4 = -1j*np.sqrt(0.5*special.lambertw(2.*Ks**4/(math.pi*s**2),1))
		zrs4 = np.real(zs4)
		zis4 = np.imag(zs4)
	
	# Do plotting
	extent = misc.getExtent(zr,zi)
	
	vmax = np.max(y2)
	norm = colors.SymLogNorm(linthresh=0.01, linscale=1,vmin=-vmax, vmax=vmax)
	
	if not noTitle: ax[0].set_title('$Re(Z\'(z))$')
	cs = ax[0].contour(Zr,Zi,y0,levels=[0],linestyles='dashed')
	cs1 = ax[0].contour(Zr,Zi,y1,levels=[0])
	ax[0].imshow(y0,origin='lower',extent=extent,norm=norm,cmap='RdBu_r',interpolation='none')
	ax[0].set_xlabel('$\mathrm{Re}(z)$')
	ax[0].set_ylabel('$\mathrm{Im}(z)$')
	
	if not noTitle: ax[1].set_title('$Im(Z\'(z))$')
	cs = ax[1].contour(Zr,Zi,y0,levels=[0],linestyles='dashed')
	cs1 = ax[1].contour(Zr,Zi,y1,levels=[0])
	ax[1].imshow(y1,origin='lower',extent=extent,norm=norm,cmap='RdBu_r',interpolation='none')
	ax[1].set_xlabel('$\mathrm{Re}(z)$')
	ax[1].set_ylabel('$\mathrm{Im}(z)$')
	
	if not noTitle: ax[2].set_title('$|Z\'(z)|$')
	vmin = np.min(np.abs(y2))
	logMin = np.ceil(np.log10(vmin))
	logMax = np.floor(np.log10(vmax))
	levels = np.power(10.0,np.arange(logMin,logMax+1))
	#cs = ax[2].contour(Zr,Zi,y0,levels=[0])
	#cs1 = ax[2].contour(Zr,Zi,y1,levels=[0],linestyles='dashed')
	norm = colors.LogNorm(vmin=vmin, vmax=vmax)
	cs2 = ax[2].contour(Zr,Zi,y2,levels=levels,colors='k')
	ax[2].imshow(y2,origin='lower',extent=extent,norm=norm,cmap='plasma',interpolation='none')
	ax[2].set_xlabel('$\mathrm{Re}(z)$')
	ax[2].set_ylabel('$\mathrm{Im}(z)$')
	
	if overlay:
		for axis in ax:
			tempXlim = axis.get_xlim()
			tempYlim = axis.get_ylim()
			axis.plot(zrs1,zis1)
			axis.plot(zrs2,zis2)
			#axis.plot(zrs3,zis3)
			axis.plot(np.real(zsCalc),np.imag(zsCalc))
			axis.plot(zrs4,zis4)
			axis.set_xlim(tempXlim)
			axis.set_ylim(tempYlim)
	
	return fig,ax

def plotDispRelVsK(fig,ax,K1,K2,nK,orLims=None,oiLims=None,logK=False,logO=False,plotApprox=False):
	if logK:
		Ks = math.sqrt(2.)*np.logspace(np.log10(K1),np.log10(K2),nK)
	else:
		Ks = math.sqrt(2.)*np.linspace(K1,K2,nK)
	
	zsCalc = np.array([ np.conj(friedConteRoot(K**2)) for K in Ks ])
	zsCalc = Ks*zsCalc
	if plotApprox:
		zrs1 = Ks*np.sqrt(1/Ks**2 + 1.5)
		zrs2 = Ks*np.sqrt(1/Ks**2 + 1.5 + 1.5*Ks**2 + 3.0*Ks**4)
		zis1 = Ks*math.sqrt(math.pi/8)*(math.sqrt(2)*2./Ks**4)*np.exp(-(1./Ks**2 + 1.5))
		zis2 = Ks*math.sqrt(math.pi/8)*(math.sqrt(2)*2./Ks**4 - 3.*math.sqrt(2))*np.exp(-(1./Ks**2 + 1.5 + 1.5*Ks**2 + 3.*Ks**4))

	ax[0].plot(Ks/math.sqrt(2.),np.real(zsCalc))
	ax[1].plot(Ks/math.sqrt(2.),np.imag(zsCalc),'-')
	if plotApprox:
		ax[0].plot(Ks/math.sqrt(2.),zrs1)
		ax[0].plot(Ks/math.sqrt(2.),zrs2)
		ax[1].plot(Ks/math.sqrt(2.),zis1,'-')
		ax[1].plot(Ks/math.sqrt(2.),zis2,'-')
	
	ax[0].set_xlim(K1,K2)
	if logO: ax[0].set_yscale('log')
	ax[1].set_xlim(K1,K2)
	ax[1].set_yscale('log')
	if logK:
		ax[0].set_xscale('log')
		ax[1].set_xscale('log')
	
	ax[0].set_xlabel('$K$')
	ax[0].set_ylabel('$\mathrm{Re}(\Omega)$')
	ax[1].set_xlabel('$K$')
	ax[1].set_ylabel('$\mathrm{Im}(\Omega)$')

	if orLims is not None:
		ax[0].set_ylim(orLims)
	
	if oiLims is not None:
		ax[1].set_ylim(oiLims)

	ax[0].grid()
	ax[1].grid()
	
	return fig,ax

def plotDispRelVsKSquared(K21,K22,nK2):
	'''
	Used to reproduce fig. 1 from McKinstrie et al., PoP 6, 463 (1999)
	'''
	Ks2 = 2*np.linspace(K21,K22,nK2)
	
	zsCalc = np.sqrt(Ks2)*np.array([ np.conj(friedConteRoot(K2)) for K2 in Ks2 ])
	zrs1 = np.sqrt(Ks2)*np.sqrt(1/Ks2 + 1.5)
	zrs2 = np.sqrt(Ks2)*np.sqrt(1/Ks2 + 1.5 + 1.5*Ks2 + 3.0*Ks2**2)
	zis1 = np.sqrt(Ks2)*math.sqrt(math.pi/8)*(math.sqrt(2)*2./Ks2**2)*np.exp(-(1./Ks2 + 1.5))
	zis2 = np.sqrt(Ks2)*math.sqrt(math.pi/8)*(math.sqrt(2)*2./Ks2**2 - 3.*math.sqrt(2))*np.exp(-(1./Ks2 + 1.5 + 1.5*Ks2 + 3.*Ks2**2))
	
	fig,ax = plt.subplots(2,1)
	ax[0].plot(Ks2/2.,np.real(zsCalc))
	ax[0].plot(Ks2/2.,zrs1)
	ax[0].plot(Ks2/2.,zrs2)
	ax[1].plot(Ks2/2.,np.imag(zsCalc))
	ax[1].plot(Ks2/2.,zis1)
	ax[1].plot(Ks2/2.,zis2)

	ax[0].set_xlim(K21,K22)
	#ax[0].set_ylim(1.04,1.25)
	ax[1].set_xlim(K21,K22)
	#ax[1].set_ylim(1e-5,1e-1)

	ax[1].set_yscale('log')
	
	ax[0].set_xlabel('$K^2$')
	ax[0].set_ylabel('$\mathrm{Re}(\Omega)$')
	ax[1].set_xlabel('$K^2$')
	ax[1].set_ylabel('$\mathrm{Im}(\Omega)$')

	ax[0].grid()
	ax[1].grid()
	plt.show()

def plotImFriedConte(zr,ziRange,nzi):
	zis = np.linspace(ziRange[0],ziRange[1],nzi)
	y = np.imag(friedConte(zr+1j*zis,d=1))
	
	plt.plot(zis,y)
	ax = plt.gca()
	ax.set_yscale('symlog')
	plt.grid()
	plt.show()

def plotErrors():
	Ks = np.logspace(-9,3,1000)
	
	y1 = numOmegaNodim(Ks)
	ry1,iy1 = np.real(y1),np.imag(y1)
	ry2,iy2 = reOmegaNodim(Ks,order=1),imOmegaNodim(Ks,order=1)
	ry3,iy3 = reOmegaNodim(Ks,order=3),imOmegaNodim(Ks,order=3)
	
	
	# Deal with real values first
	# Relative errors
	re12 = ry1/ry2 - 1.0
	re13 = ry1/ry3 - 1.0
	re23 = ry2/ry3 - 1.0
	
	# Absolute Errors
	ae12 = ry1-ry2
	ae13 = ry1-ry3
	ae23 = ry2-ry3
	
	fig,ax = plt.subplots(2,2)
	
	# Plot relative erros
	ax[0][0].plot(Ks,re12)
	ax[0][0].plot(Ks,re13)
	ax[0][0].plot(Ks,re23)
	
	ax[0][0].set_xlabel('$K$')
	ax[0][0].set_ylabel('relative error in ReP')
	ax[0][0].set_xscale('log')
	ax[0][0].set_yscale('symlog',linthreshy=1e-16)
	
	ax[0][0].grid()
	
	# Plot absolute errors
	ax[0][1].plot(Ks,ae12)
	ax[0][1].plot(Ks,ae13)
	ax[0][1].plot(Ks,ae23)

	ax[0][1].set_xlabel('$K$')
	ax[0][1].set_ylabel('absolute error in ReP')	
	ax[0][1].set_xscale('log')
	ax[0][1].set_yscale('symlog',linthreshy=1e-16)
	
	ax[0][1].grid()
	
	# Deal with imaginary values
	# Relative errors
	re12 = iy1/iy2 - 1.0
	re13 = iy1/iy3 - 1.0
	re23 = iy2/iy3 - 1.0
	
	# Absolute Errors
	ae12 = iy1-iy2
	ae13 = iy1-iy3
	ae23 = iy2-iy3
	
	# Plot relative erros
	ax[1][0].plot(Ks,re12)
	ax[1][0].plot(Ks,re13)
	ax[1][0].plot(Ks,re23)
	
	ax[1][0].set_xlabel('$K$')
	ax[1][0].set_ylabel('relative error in ImP')
	ax[1][0].set_xscale('log')
	ax[1][0].set_yscale('symlog',linthreshy=1e-16)
	
	ax[1][0].grid()
	
	# Plot absolute errors
	ax[1][1].plot(Ks,ae12)
	ax[1][1].plot(Ks,ae13)
	ax[1][1].plot(Ks,ae23)
	
	ax[1][1].set_xlabel('$K$')
	ax[1][1].set_ylabel('absolute error in ImP')
	ax[1][1].set_xscale('log')
	ax[1][1].set_yscale('symlog',linthreshy=1e-16)
	
	ax[1][1].grid()
	
	plt.show()
	#exit()

def plotGroupVel(kRange,nK):
	K = np.linspace(kRange[0],kRange[-1],nK)
	K = np.logspace(np.log10(kRange[0]),np.log10(kRange[1]),nK)
	
	#vg1 = misc.derivative(reOmegaNodim,K,dx=1e-4,order=9)
	t1 = time.time()
	vg = groupVelNodim(K)#deriv(reOmegaNodim,K)
	#print(time.time()-t1)
	#plt.plot(K,vg1)
	
	plt.title('Langmuir wave $v_g$')
	
	plt.plot(K,vg,label='Numerical')
	plt.plot(K,groupVelNodim(K,order=1),label='$O(1)$')
	plt.plot(K,groupVelNodim(K,order=3),label='$O(3)$')
	#plt.plot(K,numOmegaDerivNodim2(K),label='num2')
	#plt.plot(K,numOmegaDerivNodim2(K)/vg,'bo',label='fac')
	plt.legend(loc='lower right')
	
	ax = plt.gca()
	#ax.set_ylim(0.0,2.0)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('$K$')
	ax.set_ylabel('$\\frac{d\Omega}{dK}$',rotation=0,fontsize='x-large')
	plt.grid()
	plt.show()
	return None

def approxHighZ(z):
	'''
	Approximation of the Fried-Conte function for high z
	
	From Plasma Physics, R. A. Cairns pg 105
	'''
	a = 1./z**2 + 1.5/z**4 + 15./4./z**6
	return a

def approxLowZ(z):
	'''
	Approximation of the Fried-Conte function for low z
	
	From Plasma Physics, R. A. Cairns pg 105
	'''
	a = 1j*math.sqrt(math.pi)*np.exp(-z**2) - 2.*(1. - 2./3.*z**2 + 4./15.*z**4 - 8./105.*z**6)
	a = -2.*(1.+z*a)
	return a

if(__name__ == '__main__'):
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.rc('figure', autolayout=True)
	# Laser properties
	wlVac0 = 351e-9
	omega0 = const.c*wlVac0
	nCrit  = omega0**2*const.m_e*const.epsilon_0/const.e**2
	ey_amp = const.m_e*omega0/const.e*0.01*const.c

	# Plasma properties
	Te = 2e3*const.e/const.k
	ne = 0.15*nCrit
	
	#plotGroupVel([0.0,5.0],100)
	
	#plotErrors()
	
#	L  = 1e-4
#	N  = 100
#	K = np.linspace(1.0,1.0+L,N)
#	y = reOmegaNodim(K)
#	plt.plot(K,y)
#	plt.grid()
#	plt.show()

	#print(D(0,2))

	print(D(1.0,1.0,2,0))
	print(Dn(1.0,1.0,2))

	startTime = time.time()
	for i in xrange(100000):
		D(1.0,1.0,1,0)
	print(time.time()-startTime)
	
	startTime = time.time()
	for i in xrange(100000):
		Dn(1.0,1.0,1)
	print(time.time()-startTime)

	exit()

	K = np.logspace(-1.5,0,1000)
	y = imOmegaNodim(K)
	y1 = imOmegaNodim(K,order=1)
	plt.plot(K,np.abs(y/y1-1.))
	
	ax = plt.gca()
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.grid()
	plt.show()
	
	
	#vg1 = misc.derivative(reOmegaNodim,K,dx=1e-4,order=9)
	t1 = time.time()
	vg = deriv(reOmegaNodim,K)
	#print(time.time()-t1)
	#plt.plot(K,vg1)
	
	plt.title('Langmuir wave $v_g$')
	
	plt.plot(K,vg,label='Numerical')
	plt.plot(K,groupVelNodim(K,order=1),label='$O(1)$')
	plt.plot(K,groupVelNodim(K,order=3),label='$O(3)$')
	plt.legend(loc='lower right')
	
	ax = plt.gca()
	ax.set_ylim(0.0,2.0)
	ax.set_xscale('log')
	ax.set_xlabel('$K$')
	ax.set_ylabel('$\\frac{d\Omega}{dK}$',rotation=0,fontsize='x-large')
	plt.grid()
	plt.show()
	

	wns = srsUtils.findWNs(ne,Te,omega0)
	omegas = srsUtils.findOmegas(ne,Te,omega0)
	growthBs = srsUtils.growthRateSRS(ne,Te,ey_amp,wns['kb'],omega0)
	growthFs = srsUtils.growthRateSRS(ne,Te,ey_amp,wns['kf'],omega0)
	
	plotImFriedConte(5.0,[-5.0,0.0],100)
	
	plotDispRelVsKSquared(0.038,0.122,50)
	plotDispRelVsK(0.05,0.5,50)

	plotFriedConte([0.0,10.0],[-5.0,5.0],200,200,True)
	
