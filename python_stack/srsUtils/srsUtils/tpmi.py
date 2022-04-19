#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.stats as stats
import scipy.optimize as optimize
import sympy as sp

from srsUtils import srsUtils
from srsUtils import langmuir

#mpl.style.use('classic')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('figure', autolayout=True)

def dispRelExact():
	'''
	'Exact' dispersion relation for TPMI using the fluid electron susceptibility

	Taken from Kruer, Dawson and Sudan, PRL 23 (1969).

	Symbols
	=======
	
	Frequencies normalised to ω_p, lengths to λ_D

	dO: Frequency shift relative to O0
	dK: Wavenumber shift relative to K0
	O0: Langmuir wave frequency
	K0: Langmuir wave wavenumber
	Ob: Bounce frequency
	Ot: Trapped particle 'plasma frequency' (fT^(1/2)*ω_p, where fT is the
	    trapped fraction of the distribution)
	'''
	O0,K0,dO,dK,Ob,Ot = sp.symbols('O0,K0,dO,dK,Ob,Ot')

	Os = dO - dK*(O0/K0)
	d = dO**2 - 3*dK**2
	x = 1 + d
	y = 2*(O0*dO - 3*K0*dK)

	dr = (Os**2 - Ob**2)*(x**2 - y**2 + 1 - 2*x) - 2*Ot**2*(x**2 - y**2 - x)

	return dr,O0,K0,dO,dK,Ob,Ot

def dispRelApprox():
	'''
	Approximate version of above dispersion relation

	Some higher powers of dO and dK in the above expression have been eliminated
	'''
	O0,K0,dK,Ob,Ot = sp.symbols('O0,K0,dK,Ob,Ot',real=True)
	dO = sp.symbols('dO')

	Os = dO - dK*(O0/K0)
	d = dO**2 - 3*dK**2
	y = 2*(O0*dO - 3*K0*dK)

	dr = y**2*(Ob**2 - Os**2 + 2*Ot**2) - 2*Ot**2*d

	return dr,O0,K0,dO,dK,Ob,Ot

def dispRelMaxGrowth():
	'''
	Failed attempt at calculating the maximum growth rate

	Various things tried, currently assumes Re(dO) goes as dK which empirically
	appears to hold. This could be used to get a scaling for the growth rate
	but we would need to justify it somehow before I'd trust it enough to use.
	'''
	dr,O0,K0,dO,dK,Ob,Ot = dispRelApprox()
	dOr,dOi = sp.symbols('dOr,dOi',real=True)
	dr = dr.subs(dO,dOr+sp.I*dOi)
	#dOf = sp.Function('dOf')(dK)
	#dr = dr.subs(dO,dOf)

	imDr = sp.re(dr.subs(dOr,dK).expand()).collect(dOi)
	
	cs = sp.Poly(imDr.expand().factor().collect(dOi),dOi).all_coeffs()

	fcs = [ sp.lambdify((dK,O0,K0,Ob,Ot),c) for c in cs ]

	return fcs

def dispRelDiscriminant():
	dr,O0,K0,dO,dK,Ob,Ot = dispRelApprox()

	p = sp.Poly(dr.expand().collect(dO),dO)
	return p.discriminant(),O0,K0,dO,dK,Ob,Ot

def genCoeffFuncs(exact=False):
	if exact:
		dr,O0,K0,dO,dK,Ob,Ot = dispRelExact()
	else:
		dr,O0,K0,dO,dK,Ob,Ot = dispRelApprox()
	poly = sp.Poly(dr.expand().collect(dO),dO)
	cs = poly.all_coeffs()
	
	#for c in cs:
	#	print(c)

	fcs = [ sp.lambdify((dK,O0,K0,Ob,Ot),c) for c in cs ]

	return fcs

def calcOmegas(dK,K0,Ob,Ot,fcs=None,sort=True):
	if fcs is None:
		fcs = genCoeffFuncs()

	O0 = math.sqrt(1. + 3.*K0**2)
	cs = [ f(dK,O0,K0,Ob,Ot) for f in fcs ]
	#print(cs)

	dos = np.roots(cs)

	if sort:
		idx = np.lexsort((np.real(dos),np.imag(dos)))[::-1]
		dos = dos[idx]
	#print(dos)

	return dos

def maxdKForGrowth(K0,Ob,Ot):
	'''
	Calculates the maximum wavenumber shift (dK) at which growth can occur

	Description
	===========

	Uses the polynomial coefficients below to calculate the maximum dK**2.

	Based on approximate dispersion relation above

	Coefficients were calculated as follows:
	 1. Calculate the polynomial for the dispersion relation, i.e.
	    sum_i(a_i(dK,O0,K0,...)*dO**i).
	 2. Calculate the determinant of the above polynomial. This is zero where
	    there are multiple roots. We are interested in these points as they
	    occur at dK=0 and at the maximum growing dK. The determinant is
	    another polynomial, this time of order 8 and written in terms of dK:
	    sum_j(b_j(O0,K0,...)*dK**i)
	 3. Simplify this down, first by noting that there are no odd powers of
	    dK, so write in terms of dK**2. We now have a 4th order polynomial.
	    Second, dK**2 can be factored out, indicating as we already know that
	    there are roots at dK**2 = 0. We aren't interested in these so divide
	    out this factor. Now have polynomial of order 3. Finally we can make
	    further simplifications by noting that O**2 = 1 + 3*K**2.
	'''
	O0 = np.sqrt(1. + 3.*K0**2)

	cs = [8,
	      -12*K0**2*(2*O0**2*Ob**2 + 4*O0**2*Ot**2 - Ot**2),
	      6*K0**4*(36*K0**4*O0**4*Ob**4 + 144*K0**4*O0**4*Ob**2*Ot**2 + 144*K0**4*O0**4*Ot**4 + 126*K0**4*O0**2*Ob**2*Ot**2 + 252*K0**4*O0**2*Ot**4 + 9*K0**4*Ot**4 - 24*K0**2*O0**6*Ob**4 - 96*K0**2*O0**6*Ob**2*Ot**2 - 96*K0**2*O0**6*Ot**4 - 30*K0**2*O0**4*Ob**2*Ot**2 - 60*K0**2*O0**4*Ot**4 + 21*K0**2*O0**2*Ot**4 + 4*O0**8*Ob**4 + 16*O0**8*Ob**2*Ot**2 + 16*O0**8*Ot**4 - 4*O0**6*Ob**2*Ot**2 - 8*O0**6*Ot**4 + O0**4*Ot**4),
	      -K0**6*(2*O0**2*Ob**2 + 4*O0**2*Ot**2 - Ot**2)**3]

	# There are two complex roots, we don't care about these. Also remember to
	# take the square root as coefficients above are for powers of dK**2 not dK
	dKs = np.roots(cs)
	maxdK = np.sqrt(dKs[np.where(np.abs(np.imag(dKs)) == np.min(np.abs(np.imag(dKs))))][0])

	maxdK = np.real(maxdK)

	return maxdK

def maxGrowth(K0,Ob,Ot):
	'''
	Calculates the maximum growth rate for TPMI in a given setup

	Description
	===========

	This uses the maxdKForGrowth function to put bounds on where the maximum
	TPMI growth occurs in terms of the wavenumber shift dK.

	Returns dKMax, the shift at maximum growth along with the growth rate

	Parameters
	==========

	K0: Normalised wavenumber of the Langmuir wave (kλ_D)
	Ob: Normalised bounce frequency ω_b/ω_p
	Ot: Normalised trapped particle plasma frequency (described elsewhere)
	'''
	maxdK = maxdKForGrowth(K0,Ob,Ot)
	fcs = genCoeffFuncs(exact=False)

	gFunc = lambda dK: -np.imag(calcOmegas(dK,K0,Ob,Ot,fcs=fcs,sort=True))[0]

	result = optimize.minimize(gFunc,0.5*maxdK,bounds=[(0.0,maxdK)])
	maxGdK = result['x'][0]
	g = -result['fun']

	dKs = np.linspace(0.0,maxdK,200)
	#plt.plot(dKs,[ -gFunc(dK) for dK in dKs ])
	#plt.plot(maxGdK,g,'bo')
	#plt.grid()
	#plt.show()

	return g,maxGdK

maxGrowth = np.vectorize(maxGrowth)

def trappedFraction(k,E,ne,Te):
	'''
	Calculates the fraction of trapped electrons in the distribution

	Assumes an initially Maxwellian plasma where electrons have all been
	trapped by an oscillating 
	'''
	vth = math.sqrt(const.k*Te/const.m_e)
	vTrap = np.sqrt(const.e*E/(const.m_e*k))/vth

	vPh = langmuir.reOmega(ne,Te,k)/k/vth

	fT = stats.norm.cdf(vPh+vTrap)-stats.norm.cdf(vPh-vTrap)

	return fT

def plotDispersionRelation(fig,reAx,imAx,K0,Ob,Ot,exact=False,maxK=None,numK=500):
	fcs = genCoeffFuncs(exact=args.exact)

	if maxK is None:
		maxK = maxdKForGrowth(K0,Ob,Ot)

	dK = np.linspace(-maxK,maxK,500)
	Os = [ calcOmegas(d,K0,Ob,Ot,fcs=fcs) for d in dK ]
	
	for o in zip(*Os):
		reAx.plot(dK,np.real(o))
		imAx.plot(dK,np.imag(o))
	
	reAx.set_ylim(-1.5*maxK,1.5*maxK)
	reAx.set_ylabel(r'$\frac{\Delta \omega}{\omega_{\mathrm{p}}}$',rotation=0)
	imAx.set_ylabel(r'$\frac{\gamma}{\omega_{\mathrm{p}}}$',rotation=0)

	for axis in [reAx,imAx]:
		axis.grid()
		axis.set_xlim(1.5*dK[0],1.5*dK[-1])
		axis.set_xlabel(r'$\Delta k\cdot \lambda_D$')
	
	return fig,reAx,imAx

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	
	parser.add_argument('ne',type=float)
	parser.add_argument('Te',type=float)
	parser.add_argument('k',type=float)
	parser.add_argument('E',type=float)

	parser.add_argument('--fT',type=float)
	parser.add_argument('--useKruerParams',action='store_true')
	parser.add_argument('--exact',action='store_true')
	parser.add_argument('--maxK',type=float)
	
	parser.add_argument('--fontSize',type=float)
	parser.add_argument('-o','--output')
	parser.add_argument('--figSize',type=float,nargs=2)

	args = parser.parse_args()

	ne = args.ne*srsUtils.nCritNIF
	Te = args.Te*srsUtils.TkeV

	op = math.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
	vth = math.sqrt(const.k*Te/const.m_e)
	ld = vth/op

	if args.fT is None:
		fT = trappedFraction(args.k,args.E,ne,Te)
		print("Estimated trapped electron fraction as {:}".format(fT))
	else:
		fT = args.fT
	
	if not args.useKruerParams:
		K0 = args.k*ld
		Ob = math.sqrt(const.e*args.E*args.k/const.m_e)/op
		Ot = math.sqrt(fT)
	else:
		K0 = 0.3
		Ob = 0.07
		Ot = math.sqrt(2.52e-4)

	print("K0: {:}".format(K0))
	print("Ob: {:}".format(Ob))
	print("Ot: {:}".format(Ot))

	fig,ax = plt.subplots(2,1)

	plotDispersionRelation(fig,ax[0],ax[1],K0,Ob,Ot,exact=args.exact,maxK=args.maxK,numK=500)
	
	if args.fontSize:
		import matplotlib as mpl
		mpl.rcParams.update({'font.size':args.fontSize})
	
	if args.output:
		if args.figSize:
			fig.set_size_inches(args.figSize)
		fig.savefig(args.output)
	else:
		plt.show()
