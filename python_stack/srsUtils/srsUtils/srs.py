#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from scipy.optimize import minimize
from scipy.optimize import brentq,basinhopping,brute
import scipy.constants as const
import sympy as sp
from functools import partial
import multiprocessing as mp
import os
import copy
import time
from scipy.integrate import ode
import numba
import warnings

from srsUtils import srsUtils
from srsUtils import langmuir
from srsUtils import sympyStuff
from srsUtils import misc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif=['computer modern roman'])
plt.rc('font', size=12)
plt.rc('figure', autolayout=True)

_scriptDir = os.path.abspath(os.path.dirname(__file__))
_dispRelCacheDir = os.path.join(_scriptDir,'dispRelCache')
_dispRelCache = []
_maxTime = 0.0
_maxParams = []



def lwDispFunc(K,O,bth,a0,O0,div=False):
	'''
	Calculates exactly the LW dispersion function for SRS

	The function's roots make up the LW dispersion relation
	'''
	K0 = bth*np.sqrt(O0**2 - 1.0)
	eps= (0.5*K*(a0/bth))**2

	DM = (O-O0)**2 - 1.0 - ((K-K0)/bth)**2
	DP = (O+O0)**2 - 1.0 - ((K+K0)/bth)**2

	D = langmuir.Dn(K,O,0)

	if div:
		return D + eps*(D-1.0)*(1./DM + 1./DP)
	else:
		return D*DM*DP + eps*(D-1.0)*(DM+DP)



def dispRelSymbolic():
	'''
	Returns symbolic form of the SRS dispersion function

	Written in terms of an arbitrary electron dispersion function:
	D(ω,k) = 1 + Χ_e(ω,k)
	'''
	o,k,O0,a0,bth = sp.symbols("o k oL a0 bth")
	D = sp.symbols('D',cls=sp.Function)

	K0 = sp.sqrt(O0**2 - 1)*bth
	DNeg = (o-O0)**2 - 1 - ((k-K0)/bth)**2
	DPos = (o+O0)**2 - 1 - ((k+K0)/bth)**2

	dr = D(o,k)*DNeg*DPos + (k*a0/bth/2)**2*(D(o,k)-1)*(DPos+DNeg)

	return (o,k,O0,a0,bth,D),dr



def dispRelFunc(oDeriv=0,kDeriv=0):
	'''
	Calculates the value or derivative of the SRS dispersion function
	'''
	o,k,O0,a0,bth = sp.symbols("o k oL a0 bth")
	D = sp.symbols('D',cls=sp.Function)

	K0 = sp.sqrt(O0**2 - 1)*bth
	DNeg = (o-O0)**2 - 1 - ((k-K0)/bth)**2
	DPos = (o+O0)**2 - 1 - ((k+K0)/bth)**2

	dr = D(o,k)*DNeg*DPos + (k*a0/bth/2)**2*(D(o,k)-1)*(DPos+DNeg)
	dr = dr.diff(o,oDeriv).diff(k,kDeriv)

	Ds = np.array(sp.symbols('D:{k}:{o}'.format(k=kDeriv+1,o=oDeriv+1))).reshape((kDeriv+1,oDeriv+1))
	for i in range(oDeriv+1)[::-1]:
		for j in range(kDeriv+1)[::-1]:
			#print(dr)
			if i==0 and j==0:
				dr = dr.subs(D(o,k),Ds[0][0]).doit()
			else:
				dr = dr.subs(D(o,k).diff(o,i).diff(k,j),Ds[j][i]).doit()

	dr = dr.simplify()
	#print(dr)

	return (o,k,O0,a0,bth)+tuple(Ds.flatten()),dr
_dispRel = sympyStuff.genFuncHandler(dispRelFunc,2)



def dispRel(O,K,O0,a0,bth,oDeriv,kDeriv):
	Ds = np.zeros((kDeriv+1,oDeriv+1),dtype=np.complex128)
	for i in range(kDeriv+1):
		for j in range(oDeriv+1):
			Ds[i][j] = langmuir.D(O,K,j,i)

	Ds = Ds.flatten()

	return _dispRel(oDeriv,kDeriv,O,K,O0,a0,bth,*Ds)



def calcDispRelPowSeriesCoeffs(order):
	'''
	Returns functions for each coefficient in a SRS DR power series

	Description
	-----------

	Takes the SRS dispersion function and expands as a power series in ω up to
	an order specified as an argument. The LW dispersion function is expanded
	as a power series to accomplish this. Returns a list of functions that can
	be used to calculate these coefficients. These functions have been
	converted into FORTRAN code which is compiled and wrapped using f2py.

	Dispersion function defined exclusively in terms of normalised parameters.
	Frequencies	normalised to plasma frequency, wavenumbers K = k*λ_D.

	o0   : Complex frequency to expand about
	D[i] : ith derivative of LW dispersion function D = 1+Χ_e
	O0   : Laser frequency
	a0   : Ratio of electron quiver velocity to speed of light
	bth  : Ratio of thermal velocity to speed of light

	Parameters
	----------

	order: highest order term in ω to expand the dispersion function to

	Returns
	-------

	Returns a list of functions. The ith function generates the ith coefficient
	in the power series sum_{i=0}^{i=order}(a_i*ω^i). See below for function
	arguments.
	'''
	# D[i] is the ith derivative of the
	Ds = list(sp.symbols('D:{o}'.format(o=order+1)))

	symbols,dr = dispRelSymbolic()
	o0,k0,o,k,O0,a0,bth = symbols
	print(dr)

	dr = dr.series(o,o0,order+1).doit().removeO()
	print(dr)

	for i in range(len(Ds))[::-1]:
		if i==0:
			dr = dr.subs(D(o0),Ds[0])
			continue
		dr = dr.subs(D(o0).diff(o0,i),Ds[i])

	print(dr)

	dr = dr.subs(o-o0,do)
	print(dr)

	expr = sp.Poly(dr,do)
	print(expr)

	coeffs = expr.all_coeffs()[::-1]
	coeffs = [ c.simplify() for c in coeffs ]

	print("Coefficients:")
	for i in range(order+1): print('ω^{:}:\n'.format(i) + str(coeffs[i]))

	#coeffLambdas = [ sp.lambdify([o0,k,O0,a0,bth] + Ds,coeffs[i]) for i in range(order+1) ]
	#coeffLambdas = [ autowrap(coeffs[i],args=[o0,k,O0,a0,bth]+Ds,backend='cython',tempdir='./tmp') for i in range(order+1) ]
	coeffLambdas = [ misc.genBinFromSympy(coeffs[i],[o0,k,O0,a0,bth]+Ds[:i+1],os.path.join(_dispRelCacheDir,'order_{:}'.format(i))) for i in range(order+1) ]

	return coeffLambdas



# TODO: Clean this up (or don't, it ain't broke so...)
def getDispRelPowSeriesCoeffs(order,cache=True):
	if cache:
		global _dispRelCache
		if order < len(_dispRelCache):
			coeffLambdas = copy.copy(_dispRelCache[:order+1])
		else:
			try:
				print("Attempting to read cached functions from disk")
				coeffLambdas = readCachedDispRelPowSeriesCoeffs(order)
			except IOError:
				print("No cached functions found, generating from scratch")
				if os.path.exists(_dispRelCacheDir):
					import shutil
					shutil.rmtree(_dispRelCacheDir)
				os.makedirs(_dispRelCacheDir)
				coeffLambdas = calcDispRelPowSeriesCoeffs(order)
			except ImportError:
				print("No cached functions found, generating from scratch")
				if os.path.exists(_dispRelCacheDir):
					import shutil
					shutil.rmtree(_dispRelCacheDir)
				os.makedirs(_dispRelCacheDir)
				coeffLambdas = calcDispRelPowSeriesCoeffs(order)

			# Ensure this doesn't get modified erroneously with deepcopy
			_dispRelCache = copy.copy(coeffLambdas)
	else:
		coeffLambdas = calcDispRelPowSeriesCoeffs(order)

	return coeffLambdas



def readCachedDispRelPowSeriesCoeffs(order):
	import imp

	coeffFuncs = []
	for i in range(order+1):
		wrapper_module = imp.load_dynamic('wrapper_module_{:}'.format(i),os.path.join(_dispRelCacheDir,'order_{i}/wrapper_module_{i}.so'.format(i=i)))

		coeffFuncs.append(wrapper_module.autofunc)

	return coeffFuncs



def solveExpansion(o,K,O0,a0,bth,order):
	'''
	Finds roots of the dispersion function power series expansion

	Currently fails for K -> 0. These values aren't normally needed, but would
	be good to fix.
	'''
	coeffs = [ dispRel(o,K,O0,a0,bth,i,0)/math.factorial(i) for i in range(order+1) ][::-1]

	roots = o+np.roots(coeffs)
	validRoots = np.array([ r for r in roots if np.abs(lwDispFunc(K,r,bth,a0,O0)) < 1e-8 ])

	return validRoots



def omegasAtK(k,ne,Te,I,omega0):
	E = srsUtils.intensityToEField(I)
	vos = const.e*E/(const.m_e*omega0)
	op = np.sqrt(ne/(const.m_e*const.epsilon_0))*const.e
	vth = np.sqrt(const.k*Te/const.m_e)
	ld = vth/op

	K = k*ld
	O0 = omega0/op
	a0 = vos/const.c
	bth = vth/const.c

	Os = omegasAtKNodim(K,O0,a0,bth)
	os = { o:op*Os[o] for o in Os }

	return os



def omegasAtKNodim(K,O0,a0,bth):
	#print("Here2")
	K0 = bth*np.sqrt(O0**2-1.0)
	AP = 1.0 + ((K+K0)/bth)**2
	AN = 1.0 + ((K-K0)/bth)**2

	o00 = np.conj(langmuir.omegaNodim(K))
	oP0 = -O0 + np.sqrt(AP)
	oN0 =  O0 - np.sqrt(AN)

	o0 = followRoot(K,O0,bth,oN0,a0)
	oN = followRoot(K,O0,bth,o00,a0)
	oP = followRoot(K,O0,bth,oP0,a0)
	#print('o0: {:} -> {:}'.format(o00,o0))
	#print('oN: {:} -> {:}'.format(oN0,oN))
	#print('oP: {:} -> {:}'.format(oP0,oP))
	return {'o':o0,'oN':oN,'oP':oP}



def dOda0(a0,O,K,O0,bth):
	dDdO = dispRel(O,K,O0,a0,bth,1,0)
	D = langmuir.D(O,K,0,0)
	return dOda0Main(a0,O,K,O0,bth,dDdO,D)



@numba.jit(nopython=True)
def dOda0Main(a0,O,K,O0,bth,dDdO,D):
	bth2 = bth**2
	K2 = K**2
	dDda0 = a0*K2*(D - 1.)*(O**2 - K2/bth2)/bth2
	#print('dO/da0: {:}'.format(-dDda0/dDdO))

	return -dDda0/dDdO



def jacob(a0,O,K,O0,bth):
	chiE    = langmuir.D(O,K,0,0) - 1.0
	dChiEdO = langmuir.D(O,K,1,0)
	d2ChiEdO2 = langmuir.D(O,K,2,0)
	bth2 = bth**2
	K2 = K**2
	K0 = bth*np.sqrt(O0**2-1.)

	dOda0Var = dOda0(a0,O,K,O0,bth)
	print(dOda0Var)
	dDdO = dispRel(O,K,O0,a0,bth,1,0)
	print(dDdO)
	dDda0 = a0*K2*chiE*(O**2 - K2/bth2)/bth2
	print(dDda0)
	d2Dda02 = -K**2*chiE*(-(O - O0)**2 - (O + O0)**2 + 2 + (bth*np.sqrt(O0**2 - 1) - K)**2/bth**2 + (bth*np.sqrt(O0**2 - 1) + K)**2/bth**2)/(2*bth**2)
	print(d2Dda02)
	d2DdOda0 = 2*a0*K**2*O*chiE/bth**2 + a0*K**2*((O - O0)**2 + (O + O0)**2 - 2 - (-bth*np.sqrt(O0**2 - 1) + K)**2/bth**2 - (bth*np.sqrt(O0**2 - 1) + K)**2/bth**2)*dChiEdO/(2*bth**2)
	print(d2DdOda0)

	print(dInvdO)

	jac = -(d2Dda02/dOda0Var + d2DdOda0)/dDdO - dDda0*(dInvda0/dOda0Var + dInvdO)

	print('jacobian: {:}'.format(jac))
	#import sys
	#sys.exit()
	return jac



def followRoot(K,O0,bth,O,a0):
	integrator = ode(dOda0).set_integrator('zvode')
	integrator.set_initial_value(O,0.0).set_f_params(K,O0,bth).set_jac_params(K,O0,bth)
	integrator.integrate(a0)
	return integrator.y[0]
followRoot = np.vectorize(followRoot)



def plotOs(Ks,O0,a0,bth,outFile=None):
	Os = omegasAtKNodim(Ks,O0,a0,bth)
	o0 = Os['o']
	oN = Os['oN']
	oP = Os['oP']

	fig,axes = plt.subplots(2,1)
	axes[0].plot(Ks,np.real(o0))
	axes[0].plot(Ks,np.real(oN))
	axes[0].plot(Ks,np.real(oP))
	axes[0].set_ylabel('$\Omega_r$',rotation=0)

	axes[1].plot(Ks,np.imag(o0))
	axes[1].plot(Ks,np.imag(oN))
	axes[1].plot(Ks,np.imag(oP))
	axes[1].set_ylabel('$\Omega_i$',rotation=0)

	for ax in axes.flatten():
		ax.set_xlabel('$K$')
		ax.grid()
		ax.set_xlim(Ks[0],Ks[-1])

	if outFile:
		fig.savefig(outFile)
	else:
		plt.show()



def wns(ne,Te,I,omega0,bSRS=True):
	E = srsUtils.intensityToEField(I)
	vos = const.e*E/(const.m_e*omega0)
	op = np.sqrt(ne/(const.m_e*const.epsilon_0))*const.e
	vth = np.sqrt(const.k*Te/const.m_e)
	ld = vth/op

	O0 = omega0/op
	a0 = vos/const.c
	bth = vth/const.c

	#print('Dimensionless params: O0: {:}, a0: {:}, bth: {:}'.format(O0,a0,bth))

	Ks = wnsNodim(O0,a0,bth,bSRS=bSRS)
	#print(Ks)
	ks = Ks/ld

	return ks



def wnsNodim(O0,a0,bth,bSRS=True):
	'''
	Finds the wavenumbers of the growing modes due to SRS
	'''
	print("wnsNodim")
	dr1 = langmuir.reOmegaNodim
	dr2 = lambda K: np.sqrt(1.0+(K/bth)**2)

	K0 = bth*math.sqrt(O0**2-1.0)
	sTime = time.time()
	mw1,mw2 = srsUtils.matchWaves(K0,O0,dr1,dr2)
	eTime = time.time() - sTime
	#print(eTime)

	#print(mw1)
	#print(mw2)

	#minFunc = lambda x: -np.imag(omegasAtKNodim(x[0],O0,a0,bth,order)['oN'])
	def minFunc(x):
		#print(x)
		sTime = time.time()
		o = -np.imag(omegasAtKNodim(x[0],O0,a0,bth)['o'])
		eTime = time.time()-sTime

		global _maxTime
		if eTime > _maxTime:
			_maxTime = eTime
			global _maxParams
			_maxParams = [x[0],O0,a0,bth]
			#print(eTime)
		return o
	def derivFunc(x):
		#print(x)
		O = omegasAtKNodim(x[0],O0,a0,bth)['o']
		dfk = dispRel(O,x[0],O0,a0,bth,0,1)
		dfo = dispRel(O,x[0],O0,a0,bth,1,0)
		return np.array([np.imag(dfk/dfo)])

	# bSRS has the largest EPW wavenumber -> think about matching conditions
	if bSRS:
		initK = max(mw1[1],mw2[1])
	else:
		initK = min(mw1[1],mw2[1])

	#dK = 0.001
	#K = initK
	#Ks = np.linspace(K-dK,K+dK,100)
	#ONs = [ omegasAtKNodim(K,O0,a0,bth)['oN'] for K in Ks ]
	#OPs = [ omegasAtKNodim(K,O0,a0,bth)['oP'] for K in Ks ]
	#Os = [ omegasAtKNodim(K,O0,a0,bth)['o'] for K in Ks ]
	#plt.plot(Ks,np.imag(ONs))
	#plt.plot(Ks,np.imag(OPs))
	#plt.plot(Ks,np.imag(Os))
	#ax = plt.gca()
	#ax.set_xlim(Ks[0],Ks[-1])
	#plt.grid()
	#plt.show()
	#exit()

	bounds = [[initK-0.1*initK,initK+0.1*initK]]
	result = minimize(minFunc,[initK],jac=derivFunc,bounds=bounds)
	K = result['x']

	return K
wnsNodim = np.vectorize(wnsNodim)



def omegas(ne,Te,I,omega0,wns=None,bSRS=True):
	E = srsUtils.intensityToEField(I)
	vos = const.e*E/(const.m_e*omega0)
	op = np.sqrt(ne/(const.m_e*const.epsilon_0))*const.e
	vth = np.sqrt(const.k*Te/const.m_e)

	O0 = omega0/op
	a0 = vos/const.c
	bth = vth/const.c

	if wns is not None:
		Os = omegasNodim(O0,a0,bth,wns=wns*vth/op,bSRS=bSRS)
	else:
		Os = omegasNodim(O0,a0,bth,wns=None,bSRS=bSRS)
	os = { o:Os[o]*op for o in Os }

	return os



def omegasNodim(O0,a0,bth,wns=None,bSRS=True):
	if wns is None:
		wns = wnsNodim(O0,a0,bth,bSRS=bSRS)
	Os = omegasAtKNodim(wns,O0,a0,bth)

	return Os



def plotDispRelFuncVsOmega(K,a0,bth,O0,lims=None):
	eps = lambda Or,Oi: lwDispFunc(K,Or+1j*Oi,bth,a0,O0)
	epsv = np.vectorize(eps)
	if not lims:
		Ors = np.linspace(-1.5,1.5,100)
		Ois = np.linspace(-1.5,1.5,100)
	else:
		Ors = np.linspace(lims[0][0],lims[0][1],100)
		Ois = np.linspace(lims[1][0],lims[1][1],100)

	ORs,OIs = np.meshgrid(Ors,Ois)
	y = epsv(ORs,OIs)

	fig,ax = plt.subplots(1,3)
	misc.plotComplexFunc(ax,y,Ors,Ois,
			        xLabel='$\mathrm{Re}(z)$',yLabel='$\mathrm{Im}(z)$',
					funcName=r'\varepsilon(z)')
	return fig



def compareExpansion(K,O0,a0,bth,order):
	o0 = np.conj(langmuir.omegaNodim(K))
	Ds = [ langmuir.Dn(K,o0,i) for i in range(order+1) ]

	coeffLambdas = getDispRelPowSeriesCoeffs(order)

	coeffFunc = lambda O,K,i,Ds: np.complex128(coeffLambdas[i](O,K,O0,a0,bth,*Ds[:i+1]))
	coeffs = [ coeffFunc(o0,K,i,Ds) for i in range(order+1) ][::-1]

	dO = 1.0
	Ors = np.linspace(np.real(o0)-dO,np.real(o0)+dO,100)
	Ois = np.linspace(np.imag(o0)-dO,np.imag(o0)+dO,100)
	ORs,OIs = np.meshgrid(Ors,Ois)
	Os = ORs + 1j*OIs
	drFuncVec = np.vectorize(lwDispFunc)
	dr = drFuncVec(K,Os,bth,a0,O0)
	drApprox = np.zeros(Os.shape,dtype='complex128')
	for i in range(order+1):
		drApprox += coeffs[::-1][i]*(Os-o0)**i
	print(dr)

	drDiff = drApprox - dr

	fig,ax = plt.subplots(3,3)
	misc.plotComplexFunc(ax[0],dr,Ors,Ois,
			        xLabel='$\mathrm{Re}(z)$',yLabel='$\mathrm{Im}(z)$',
					funcName=r'\varepsilon(z)')
	misc.plotComplexFunc(ax[1],drApprox,Ors,Ois,
			        xLabel='$\mathrm{Re}(z)$',yLabel='$\mathrm{Im}(z)$',
					funcName=r'\varepsilon(z)')
	misc.plotComplexFunc(ax[2],drDiff,Ors,Ois,
			        xLabel='$\mathrm{Re}(z)$',yLabel='$\mathrm{Im}(z)$',
					funcName=r'\varepsilon(z)')
	plt.show()



def compareExpansion2(K,O0,a0,bth,order):
	o0 = np.conj(langmuir.omegaNodim(K))

	coeffs = [ dispRel(o0,K,O0,a0,bth,i,0) for i in range(order+1) ][::-1]

	dO = 1.0
	Ors = np.linspace(np.real(o0)-dO,np.real(o0)+dO,100)
	Ois = np.linspace(np.imag(o0)-dO,np.imag(o0)+dO,100)
	ORs,OIs = np.meshgrid(Ors,Ois)
	Os = ORs + 1j*OIs
	drFuncVec = np.vectorize(lwDispFunc)
	dr = drFuncVec(K,Os,bth,a0,O0)
	drApprox = np.zeros(Os.shape,dtype='complex128')
	for i in range(order+1):
		drApprox += coeffs[::-1][i]*(Os-o0)**i/math.factorial(i)
	print(dr)

	drDiff = drApprox - dr

	fig,ax = plt.subplots(3,3)
	misc.plotComplexFunc(ax[0],dr,Ors,Ois,
			        xLabel='$\mathrm{Re}(z)$',yLabel='$\mathrm{Im}(z)$',
					funcName=r'\varepsilon(z)')
	misc.plotComplexFunc(ax[1],drApprox,Ors,Ois,
			        xLabel='$\mathrm{Re}(z)$',yLabel='$\mathrm{Im}(z)$',
					funcName=r'\varepsilon(z)')
	misc.plotComplexFunc(ax[2],drDiff,Ors,Ois,
			        xLabel='$\mathrm{Re}(z)$',yLabel='$\mathrm{Im}(z)$',
					funcName=r'\varepsilon(z)')
	plt.show()



def _animHelper(K,ORs,OIs,bth,a0,O0):
	eps = lambda Or,Oi: lwDispFunc(K,Or+1j*Oi,bth,a0,O0)
	epsv = np.vectorize(eps)
	vals = epsv(ORs,OIs)
	return vals



def animDispRelFunc(Ks,bth,a0,O0,lims=None):
	if not lims:
		Ors = np.linspace(-1.5,1.5,200)
		Ois = np.linspace(-1.5,1.5,200)
	else:
		Ors = np.linspace(lims[0][0],lims[0][1],200)
		Ois = np.linspace(lims[1][0],lims[1][1],200)

	ORs,OIs = np.meshgrid(Ors,Ois)
	func = partial(_animHelper,ORs=ORs,OIs=OIs,bth=bth,a0=a0,O0=O0)

	p = mp.Pool()
	Y = p.map(func,Ks)
	p.close()

#	Y = []
#	for K in Ks:
#		eps = lambda Or,Oi: lwDispFunc(K,Or+1j*Oi,bth,a0,O0)
#		epsv = np.vectorize(eps)
#		vals = epsv(ORs,OIs)
#		Y.append(vals)

	fig,axs = plt.subplots(1,3)
	fig.tight_layout()

	fig.suptitle(r'\varepsilon(K,\Omega,\Omega_0={:.3f},\alpha_0={:.3f},\beta_{{\mathrm{{th}}}}={:.3f})'.format(O0,a0,bth))

	axs[0].set_title(r'$Re(\varepsilon(\Omega))$')
	axs[1].set_title(r'$Im(\varepsilon(\Omega))$')
	axs[2].set_title(r'$|\varepsilon(\Omega)|$')

	extent = misc.getExtent(Ors,Ois)

	vmin = np.min(np.array([ np.abs(y) for y in Y ]))
	vmax = np.max(np.array([ np.abs(y) for y in Y ]))

	norm = colors.SymLogNorm(linthresh=0.01, linscale=1,vmin=-vmax, vmax=vmax)

	# Levels for abs(Y) contour plot
	logMin = np.ceil(np.log10(vmin))
	logMax = np.floor(np.log10(vmax))
	levels = np.power(10.0,np.arange(logMin,logMax+1))

	for ax in axs:
		ax.set_xlabel('$\mathrm{Re}(\Omega)$')
		ax.set_ylabel('$\mathrm{Im}(\Omega)$')
		ax.set_xlim(extent[0],extent[1])
		ax.set_ylim(extent[2],extent[3])

	plots = []
	for i in range(len(Ks)):
		artists = []
		O = langmuir.omegaNodim(Ks[i])
		Or = np.real(O)
		Oi = -np.imag(O)

		K0 = bth*np.sqrt(O0**2 - 1.0)
		OM =  O0 - np.sqrt(1.0 + ((Ks[i]-K0)/bth)**2)
		OP = -O0 + np.sqrt(1.0 + ((Ks[i]+K0)/bth)**2)

		ys = [ np.real(Y[i]),
		       np.imag(Y[i]),
		        np.abs(Y[i])  ]

		#t = plt.figtext(0.5,0.5,r'$K = {:.4f}$'.format(Ks[i]))
		#artists.append(t)

		ims = [ axs[i].imshow(y,origin='lower',extent=extent,
		                      norm=norm,cmap='RdBu_r',interpolation='none',animated=True)
		        for i,y in enumerate(ys) ]

		for ax in axs:
			artists += ax.plot(Or,Oi,'kx')
			artists += ax.plot(OM,0.0,'bx')
			artists += ax.plot(OP,0.0,'rx')

		artists += ims
		for ax in axs[:2]:
			cs0 = ax.contour(ORs,OIs,ys[0],levels=[0])
			cs1 = ax.contour(ORs,OIs,ys[1],levels=[0],linestyles='dashed')
			artists += cs0.collections
			artists += cs1.collections

		cs2 = axs[2].contour(ORs,OIs,ys[2],levels=levels,colors='k')
		artists += cs2.collections

		#print(artists)
		plots.append(artists)

	print('left loop')
	print(plots)

	anim = animation.ArtistAnimation(fig,plots,blit=True)
	anim.save('animation.mp4',writer='ffmpeg_file',dpi=300,savefig_kwargs={'bbox_inches':'tight'},bitrate=5000,fps=30)
	# _animation.FFMpegWriter(),



def wnsMatchSymb():
	'''
	Generates polynomial coefficients for SRS wavenumber

	Uses the fluid EPW dispersion relation
	'''
	k0,dk,o0,op,c,vth = sp.symbols('k_0,dk,o_0,o_p,c,vth',real=True)

	# Wavenumber matching
	k  = k0/2 + dk
	ks = k0/2 - dk

	# Frequencies
	o  = sp.sqrt(op**2 + 3*vth**2*k**2)
	os = sp.sqrt(op**2 + c**2*ks**2)

	# Frequency matching. This is the frequency matching equation squared
	# twice to remove square roots
	expr = (o0**2 - (o**2 + os**2))**2 - (2*o*os)**2
	expr = expr*16

	# Normalise units to o0, c, etc.
	K0,dK,Op,bth = sp.symbols('K_0,dK,O_p,bth',real=True)
	expr = expr.subs(k0,K0*o0/c).subs(dk,dK*o0/c).subs(op,Op*o0).subs(vth,bth*c)

	expr = expr/o0**4
	sp.pprint(expr.expand().collect(dK))
	coeffs = sp.Poly(expr,dK).all_coeffs()

	for i,co in enumerate(coeffs):
		print('\n{:}'.format(i))
		sp.pprint(co.factor())



def _wnsNodimMatch(Op,bth,K0,relativistic=False):
	'''
	Solves the matching conditions to find the SRS wavenumbers

	Description
	===========

	All quantities normalised to light units. Derived using wnsMatchSymb code
	above using the fluid EPW dispersion relation. Attempts relativistic
	correction to the plasma frequency if requested.

	Parameters
	==========

	Op: Plasma frequency normalised to light frequency
	bth: Thermal velocity relative to c
	K0: Light wavenumber normalised to omega0/c
	'''

	if relativistic:
		Op = Op*np.sqrt(1.0 - 2.5*bth**2)

	bth2 = bth**2
	K02 = K0**2

	coeffs = [16.*(3.*bth2 - 1.)**2,
	          32.*K0*(3.*bth2 - 1.)*(3.*bth2 + 1.),
			  8.*(27.*K02*bth2**2 + 6.*K02*bth2 + 3.*K02 - 12.*bth2 - 4.),
	          8.*K0*(3.*bth2 - 1.)*(3.*K02*bth2 + K02 - 4.),
			  9.*(K02*bth2)**2 - 6.*K02**2*bth2 + K02**2 - 24.*K02*bth2 - 8.*K02 - 64.*Op**2 + 16.]

	dK = np.roots(coeffs)
	K  = 0.5*K0+dK
	Ks = 0.5*K0-dK

	correct = np.where(np.logical_and(np.abs(Ks)<1.0,np.imag(Ks) == 0.0))
	if len(correct[0]) == 0:
		return float('NaN'),float('NaN')

	K  =  K[correct]
	Ks = Ks[correct]

	Ksb = np.min(Ks)
	Ksf = np.max(Ks)
	#print(dK)
	#print(K)
	#print(Ks)
	#O  = np.sqrt(Op**2 + 3.*bth**2*K**2)
	#Os = np.sqrt(Op**2 + Ks**2)
	#print('O:  {:}\nOs: {:}'.format(O,Os))
	#print('1 - (O + Os) = {:}'.format(1. - (O+Os)))

	return Ksb,Ksf
_wnsNodimMatchVec = np.vectorize(_wnsNodimMatch)



def wnsNodimMatch(Op,bth,K0,relativistic=False):
	Ksb,Ksf = _wnsNodimMatchVec(Op,bth,K0,relativistic)
	Kb = K0 - Ksb
	Kf = K0 - Ksf

	return {'k0':K0,'kb':Kb,'ksb':Ksb,'kf':Kf,'ksf':Ksf}
wnsNodimMatch.__doc__ = _wnsNodimMatch.__doc__



def wnsMatch(ne,Te,o0=srsUtils.omegaNIF,relativistic=False):
	'''
	Solves the matching conditions to find the SRS wavenumbers

	Description
	===========

	Derived using wnsMatchSymb code above using the fluid EPW dispersion
	relation. Makes relativistic correction to the plasma frequency if
	requested.

	Parameters
	==========

	ne: Electron density
	Te: Electron temperature
	o0: Laser frequency
	'''
	bth = np.sqrt(const.k*Te/const.m_e)/const.c
	Op = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))/o0

	K0 = np.sqrt(1.0 - Op**2)

	wns = wnsNodimMatch(Op,bth,K0,relativistic)
	wns = { k:v*o0/const.c for k,v in wns.items() }

	return wns



def _wnsNodimMatchTheta(Op,bth,theta,relativistic=False):
	'''
	Solves the SRS matching conditions for sidescatter at an arbitrary angle

	Description
	===========

	Uses the fluid EPW dispersion relation. Theta is defined as the angle
	between the scattered light and laser wavevectors.

	Makes relativistic correction to the plasma frequency if requested.

	Returns Kx,Ky for the EPW.
	'''
	if relativistic:
		Op = Op*np.sqrt(1.0 - 2.5*bth**2)
	cos = math.cos(theta)
	sin = math.sin(theta)
	V2 = 3.*bth**2
	K0 = math.sqrt(1.-Op**2)
	coeffs = [(V2 - 1.)**2,
	          -4.*K0*V2*(V2 - 1.)*cos,
			  -4.*K0**2*V2**2*sin**2 + 6.*K0**2*V2**2 - 2.*K0**2*V2 - 2.*V2 - 2.,
			  4.*K0*V2*(-K0**2*V2 + 1.)*cos,
			  K0**4*V2**2 - 2*K0**2*V2 - 4*Op**2 + 1.]

	KsMag = np.roots(coeffs)
	KsMag = KsMag[np.where(np.logical_and(KsMag >= 0.0, KsMag <= 1.0))]
	if len(KsMag) == 1:
		KsMag = KsMag[0]
	elif len(KsMag) > 1:
		#raise ValueError("Multi-valued answer")
		warnings.warn("Multi-valued solution")
		return float('NaN'),float('NaN')
	else:
		warnings.warn("No solution")
		return float('NaN'),float('NaN')

	K0  = math.sqrt(1. - Op**2)
	Ksx = KsMag*np.cos(theta)
	Ky  = KsMag*np.sin(theta)

	Kx  = K0 - Ksx

	return Kx,Ky
wnsNodimMatchTheta = np.vectorize(_wnsNodimMatchTheta)
wnsNodimMatchTheta.__doc__ = _wnsNodimMatchTheta.__doc__



def _wnMatchDens(Te,b,o0=srsUtils.omegaNIF,relativistic=False):
	'''
	Finds the density where SRS matching occurs with given wavenumber ratios

	a: k  /k_0
	b: k_s/k_0

	Given a value for b, we know 'a' from the wavenumber matching condition
	'''
	vth = math.sqrt(const.k*Te/const.m_e)
	bth2 = (vth/const.c)**2

	a = 1.-b

	coeffs = [(3.*a**2*bth2 - b**2)**2,
	          -2.*o0**2*(9.*a**4*bth2**2 - 6.*a**2*b**2*bth2 - 3.*a**2*bth2 + b**4 - b**2 + 2.),
	          o0**4*(3.*a**2*bth2 - b**2 - 2.*b - 1.)*(3.*a**2*bth2 - b**2 + 2.*b - 1.)]

	rts = np.roots(coeffs)
	rts = rts[np.where(rts/o0**2 < 0.25)]

	if len(rts) > 1:
		raise ValueError("No unique solution, multiple places where this would be matched?")
	elif len(rts) == 0:
		raise ValueError("Couldn't find a solution, no density where this wavenumber ratio is possible")

	if relativistic:
		fac = 1.0 - 2.5*bth2
		ne = const.m_e*const.epsilon_0/const.e**2*rts[0]/fac
	else:
		ne = const.m_e*const.epsilon_0/const.e**2*rts[0]
	return ne
wnMatchDens = np.vectorize(_wnMatchDens)
wnMatchDens.__doc__ = _wnMatchDens.__doc__



def wnMatchDens_i(Te,i,N,o0=srsUtils.omegaNIF):
	'''
	Finds the density where k_s/k_0 = i/N

	i<0 for bSRS, i>0 for fSRS and i=0 at n_cr/4
	'''
	b = float(i)/N

	return wnMatchDens(Te,b,o0)



def landauCutoffDens(bth,theta,relativistic=False,cutoff=0.3):
	def getKLd(Op):
		Ld = bth/Op
		Kx,Ky = wnsNodimMatchTheta(Op,bth,theta,relativistic=relativistic)
		kMag = math.sqrt(Kx**2 + Ky**2)

		return kMag*Ld - cutoff

	result = brentq(getKLd,0.01,0.48)

	return result**2



def _growthRateNodim(Op,bth,a0,Kx,Ky,damping=False):
	'''
	Calculates the SRS homogeneous convective growth rate

	Description
	===========

	All parameters normalised to laser parameters

	Op: Plasma frequency (normalised to ω0)
	bth: Thermal velocity (normalised to c)
	a0: Laser amplitude
	Kx: x wavenumber (normalised to ω0/c)
	Ky: y wavenumber (normalised to ω0/c)
	damping: Include Landau damping rate
	'''
	KMag = np.sqrt(Kx**2 + Ky**2)
	Oek  = np.sqrt(Op**2 + 3.*bth**2*KMag**2)
	Os   = 1.0 - Oek

	gL = 0.0
	if damping:
		Ld = bth/Op
		gL = langmuir.imOmegaNodim(KMag*Ld)*Op

	g0 = 0.25*KMag*Op*a0/np.sqrt(Oek*Os)

	g = g0*np.sqrt(1. + 0.25*(gL/g0)**2) - 0.5*gL

	return g
growthRateNodim = np.vectorize(_growthRateNodim)



def rosenbluthGain(ne,Te,I,nDeriv,omega0=srsUtils.omegaNIF,k=None,bSRS=True,kinetic=True):
	'''
	Calculates the Rosenbluth gain exponent for SRS

	Rosenbluth, PRL 29 1972, Parametric Instabilities in Inhomogeneous Media
	https://link.aps.org/doi/10.1103/PhysRevLett.29.565

	TODO: Figure out what's with the factor of pi
	G = 2*π*γ**2/|κ'*V_g,1*V_g,2|
	Intensity of field then goes as I_0*e**G
	Note that this is the exponent for the field intensity, not the amplitude.
	The amplitude gain exponent is G/2
	'''
	op = np.sqrt(ne/(const.m_e*const.epsilon_0))*const.e
	k0 = np.sqrt(omega0**2-op**2)/const.c
	if kinetic:
		if k is None:
			k = wns(ne,Te,I,omega0,bSRS=bSRS)
		omegasSRS = omegas(ne,Te,I,omega0,wns=k,bSRS=bSRS)['o']
		growthRate = np.imag(omegasSRS)
	else:
		if k is None:
			if kinetic:
				if bSRS:
					k = srsUtils.findWNs(ne,Te,omega0)['kb']
				else:
					k = srsUtils.findWNs(ne,Te,omega0)['kf']
			else:
				if bSRS:
					k = wnsMatch(ne,Te,omega0)['kb']
				else:
					k = wnsMatch(ne,Te,omega0)['kf']

		E = srsUtils.intensityToEField(I)
		growthRate = srsUtils.growthRateSRS(ne,Te,E,k,omega0)

	ks = k0-k

	# These are the velocities that appear in the EPW and EMW dispersion relations
	# TODO: replace EPW one with more accurate version
	v0 = const.c
	v1 = math.sqrt(3.0*const.k*Te/const.m_e)
	v2 = const.c

	k_T_inv = 1.0/(v0**2*k0) - 1.0/(v1**2*k) - 1.0/(v2**2*ks)
#	if kinetic:
#		vg1 = langmuir.groupVel(ne,Te,k)
#	else:
#		vg1 = 3.*const.k*Te/const.m_e*k/np.sqrt(op**2 + 3.*const.k*Te/const.m_e*k**2)
#	print(vg1)
	vg1 = langmuir.groupVel(ne,Te,k)
	vg2 = srsUtils.grVelEM(ne,ks)

	dkdn = -0.5*const.e**2/(const.m_e*const.epsilon_0)*nDeriv*k_T_inv
	#print('vg_ek: {:}, vg_s: {:}, dkdn: {:} γ_0: {:}'.format(vg1,vg2,dkdn,growthRate))
	G = 2.0*math.pi*growthRate**2/np.abs(dkdn*vg1*vg2)

	return G



def srsConvThreshInhom(ne,Te,Ln,nDeriv,omega0=srsUtils.omegaNIF,ver=1):
	'''
	Calculates the threshold intensity for growth in an inhomogeneous plasma

	Uses Rosenbluth threshold to do this. The Rosenbluth gain implicitly depends
	on intensity via the instability growth rate.
	'''
	bSRS = True
	op = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
	wns = findWNs(ne,Te,omega0)
	omegas = findOmegas(ne,Te,omega0)

	v0 = const.c
	v1 = math.sqrt(3.0*const.k*Te/const.m_e)
	v2 = const.c

	if bSRS:
		k_T_inv = 1.0/(v0**2*wns['k0']) - 1.0/(v1**2*wns['kb']) - 1.0/(v2**2*wns['ksb'])
	else:
		k_T_inv = 1.0/(v0**2*wns['k0']) - 1.0/(v1**2*wns['kf']) - 1.0/(v2**2*wns['ksf'])

	if bSRS:
		vg1 = _langmuir.groupVel(ne,Te,wns['kb'])
		vg2 = grVelEM(ne,wns['ksb'])
	else:
		vg1 = _langmuir.groupVel(ne,Te,wns['kf'])
		vg2 = grVelEM(ne,wns['ksf'])
	dkdn = -0.5*const.e**2/(const.m_e*const.epsilon_0)*nDeriv*k_T_inv
	EThresh2 = 2.0*math.sqrt(2.0)*const.m_e*omega0*np.sqrt(np.abs(vg1*vg2*dkdn)*omegas['ksb']*omegas['kb'])/(wns['kb']*const.e*op)

	op = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
	k0 = np.sqrt(omega0**2-op**2)/const.c
	l0 = 2.0*math.pi/k0

	EThresh = const.m_e*omega0*const.c/const.e*np.sqrt(2.0/(k0*Ln))

	if ver == 1:
		return intensity(EThresh)
	elif ver == 2:
		return 4e17/(Ln/1e-6)/(l0/1e-6)*1e4
	elif ver == 3:
		return intensity(EThresh2)



def plotRBG(Te,E,Ln,omega0=srsUtils.omegaNIF):
	x = np.linspace(-1000.0,-10.0)*1e-6
	ne = nCritNIF*0.25*np.exp(x/Ln)
	nDerivs = 0.25*nCritNIF/Ln*np.exp(x/Ln)

	G = rosenbluthGainExp(ne,Te,E,nDerivs)
	I = srsConvThreshInhom(ne,Te,Ln,nDerivs,omega0)
	I2 = srsConvThreshInhom(ne,Te,Ln,nDerivs,omega0,ver=2)
	I3 = srsConvThreshInhom(ne,Te,Ln,nDerivs,omega0,ver=3)
	ISF = selfFocusingThreshold(ne,Te,8.0*6.7**2*wlVacNIF,omega0)

	fig = _plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.plot(x,G)
	ax1.grid()

	ax2 = fig.add_subplot(122)
	ax2.plot(x,I/1e4)
	#ax2.plot(x,I2/1e4)
	ax2.plot(x,I3/1e4)
	ax2.plot(x,ISF/1e4)
	ax2.axhline(0.75e15)
	#ax2.set_yscale('log')
	ax2.grid()

	_plt.show()



def afeyanGrowth1985(Te,Ln,I,k,ky):
	'''
	Growth rate of absolute SRS sidescatter in an inhomogeneous plasma

	Based on Eq. 54 of Afeyan & Williams, Phys. Fluids 28 (3397), 1985

	Note that the density (ne) and EPW wavenumnber (k) are determined by ky.
	As far as I can tell k == k0(ne). The density is the maximum possible
	allowed by the matching conditions at a particular temperature. Aside from
	this there is no dependence on temperature aside from in the Landau damping
	factor.

	TODO: Figure out where ky comes into this! Maybe it doesn't??
	'''
	ne = srsTurnPointDens(Te,ky)
	op = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
	E = srsUtils.intensityToEField(I)
	v0 = const.e*E/(const.m_e*srsUtils.omegaNIF)
	a0 = v0/const.c
	o0 = srsUtils.omegaNIF
	Op = op/o0

	nuBar = Op*langmuir.imOmega(ne,Te,k)
	g = np.sqrt(2. - 2.*Op - Op**2)

	return 0.5*(op*(a0*g - Op/((o0*Ln/const.c)*np.sqrt(a0*Op*g))) - nuBar)



def afeyanThresh1985(Te,Ln,k,ky,polarisation='p'):
	'''
	Threshold of absolute SRS sidescatter in an inhomogeneous plasma

	Based on Eqs. 50-52 of Afeyan & Williams, Phys. Fluids 28 (3397), 1985

	Note that the density (ne) and EPW wavenumnber (k) are determined by ky.
	As far as I can tell k == k0(ne). The density is the maximum possible
	allowed by the matching conditions at a particular temperature. Aside from
	this there is no dependence on temperature aside from in the Landau damping
	factor.
	'''
	ne = srsTurnPointDens(Te,ky)
	op = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
	Op = op/srsUtils.omegaNIF

	if polarisation == 's':
		f = np.power(Op**2,1./3)/(2. - 2.*Op - Op**2 + 2.*np.sqrt(1.-Op))
	elif polarisation == 'p':
		f = np.power(Op**2,1./3)/(2. - 2.*Op - Op**2)
	else:
		raise ValueError("Polarisation must be p or s")

	a0 = np.sqrt(f/np.power(srsUtils.wnVacNIF*Ln,4./3))

	E0 = (const.c*a0)*const.m_e*srsUtils.omegaNIF/const.e
	I = srsUtils.intensity(E0)

	return I



def srsTurnPointDens(Te,ky):
	'''
	Density at which SRS scattered light is born at its turning point

	At this point k_s,x = 0 (scattered EM wave x-wavenumber) and k_x=k_0
	(EPW x-wavenumber).
	'''
	vth = np.sqrt(const.k*Te/const.m_e)
	bth = vth/const.c
	#print(ky)
	Ky  = ky*const.c/srsUtils.omegaNIF
	#print(Ky)

	B = 3.*bth**2

	a = B**2
	b = -2.*(B**2*Ky**2 + B**2 - B*Ky**2 - B + 2.)
	c = (B*Ky**2 + B - Ky**2 - 2.*Ky - 1.)*(B*Ky**2 + B - Ky**2 + 2.*Ky - 1)

	# Note: This can be simplified
	#print(b**2-4.*a*c)
	op2 = (-b - np.sqrt(b**2-4.*a*c))/(2.*a)

	ne = srsUtils.nCritNIF*op2

	return ne


if(__name__ == '__main__'):
	K = 0.7
	a0 = 0.01
	bth = 0.17
	O0 = 2.5
	eps = 0.25*(K*a0/bth)**2

	Or = 1.0
	Oi = 0.0
	dO = 1.5

	sp.init_printing(use_unicode=True)
	#fig = plotDispRelFuncVsOmega(K,a0,bth,O0,lims=((Or-dO,Or+dO),(Oi-dO,Oi+dO)))
	#plt.show()
	#exit()
	#omegasAtKNodim(K,O0,a0,bth,25)
	#compareExpansion2(K,O0,a0,bth,10)
	#exit()

	#dispRelFunc(oDeriv=1,kDeriv=1)
	#print(dispRel(1.0,0.2,O0,a0,bth,1,1))
	#exit()

	x = np.linspace(-1000e-6,-7.5e-6,50)
	ne = 0.25*srsUtils.nCritNIF*np.exp(x/300e-6)

	#ne = np.linspace(0.05,0.24,50)*srsUtils.nCritNIF
	Te = 2e3*const.e/const.k
	I = 2.5e15*1e4
	E = srsUtils.intensityToEField(I)
	omega0 = srsUtils.omegaNIF
	#op = math.sqrt(ne/const.m_e/const.epsilon_0)*const.e

	oNs = []
	for n in ne:
		print("n : {:}".format(n/srsUtils.nCritNIF))
		ks = wns(n,Te,I,omega0,5)
		print("wns: {:}".format(wns))
		os = omegas(n,Te,I,omega0,5,wns=ks)
		print("os: {:}".format(os))
		oNs.append(os['kb']['oN'])

	#exit()
	print(_maxTime)
	print(_maxParams)

	wnsOld= srsUtils.srsWNs(ne,Te,omega0=omega0)
	oNOld = srsUtils.growthRateSRS(ne,Te,E,wnsOld['kb'],omega0=omega0)

	print(len(ne))
	print(len(oNs))
	#plt.plot(ne/srsUtils.nCritNIF,np.imag(oNs),label='Full kinetic DR')
	#plt.plot(ne/srsUtils.nCritNIF,oNOld,label='Kruer')
	fix,ax = plt.subplots()
	ax.plot(x/1e-6,np.imag(oNs),label='Full kinetic DR')
	ax.plot(x/1e-6,oNOld,label='Kruer')
	#ax.set_xlabel('$n_e/n_{\mathrm{crit}}$')
	ax.set_xlabel('$x$ /$\mu$m')
	ax.set_ylabel('$\gamma_{\mathrm{SRS}}$')

	ax2 = ax.twinx()
	ax2.plot(x/1e-6,ne/srsUtils.nCritNIF,'k--',label='$n_e$')
	ax2.set_ylabel('$n_e/n_{\mathrm{crit}}$')
	plt.title('$T_e = {:}$ keV, $I = {:.2e}$ Wcm$^2$'.format(Te*const.k/const.e/1e3,I/1e4))

	ax.set_xlim([x[0]/1e-6,x[-1]/1e-6])
	ax.legend(loc='best')
	ax.grid()
	plt.show()

	exit()

	#soltn = omegasAtKNodim(0.0,O0,a0,bth,50)
	#exit()

	Ks = np.linspace(0.175,0.2,500)

	#animDispRelFunc(Ks,bth,a0,O0,lims=((Or-dO,Or+dO),(Oi-dO,Oi+dO)))

	soltn = [ omegasAtKNodim(K,O0,a0,bth,6) for K in Ks ]
	os = np.array([ s['o'] for s in soltn ])
	oPs = np.array([ s['oP'] for s in soltn ])
	oNs = np.array([ s['oN'] for s in soltn ])

	print(os)
	fig,ax = plt.subplots(2,1)
	ax[0].plot(Ks,np.real(os),'b')
	ax[1].plot(Ks,np.imag(os),'b')
	ax[0].plot(Ks,np.real(oPs),'g')
	ax[1].plot(Ks,np.imag(oPs),'g')
	ax[0].plot(Ks,np.real(oNs),'r')
	ax[1].plot(Ks,np.imag(oNs),'r')
	for axis in ax:
		axis.grid()
		axis.set_xlim(np.min(Ks),np.max(Ks))
		axis.set_xlabel('$K$')
	ax[0].set_ylabel('$\Omega_r$',rotation='horizontal')
	ax[1].set_ylabel('$\Omega_i$',rotation='horizontal')
	plt.show()
