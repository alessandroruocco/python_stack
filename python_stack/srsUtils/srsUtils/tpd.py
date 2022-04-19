#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import scipy.optimize as optimize
from scipy import misc
import time

from srsUtils import srsUtils
from srsUtils import misc
from srsUtils import langmuir
from srsUtils import sympyStuff

def wnsNodim(Bth,O0,theta=None,centredAngle=False):
	'''
	Calculate TPD wavenumbers

	Calculates the wavenumbers of the waves involved in TPD based on normalised
	parameters. Uses the EPW warm-plasma dispersion relation for solving the
	matching conditions.

	It works, but I can't remember how I derived it...

	Parameters
	----------

	Bth: Thermal velocity relative to the speed of light
	O0: Laser frequency relative to the plasma frequency
	theta: angle with respect to propagation direction of pump EM wave
	centredAngle: Decides the point about which to measure the angle. If True
	              the reference point is k = (k0/2,0). This guarantees that a
				  solution will be found regardless of theta. Otherwise (0,0).
	'''
	K0mag = Bth*np.sqrt(O0**2 - 1.)
	K0 = np.outer(K0mag,np.array([1.0,0.0]))

	if theta is None:
		# Calculate wns at angle of maximum growth
		B = 1.0-0.25*O0**2
		A = 0.5*9.*K0mag**2/O0**2 - 3.
		Kperp = np.sqrt(0.5*B/A - 0.25*K0mag**2)
		Kpara = np.sqrt(0.25*K0mag**2 + Kperp**2)
		K1 = 0.5*K0 + np.array([Kpara,Kperp]).transpose()
	else:
		if not centredAngle:
			# Wavenumbers centred at (0,0)
			A = 3.
			B = 3.*K0mag/O0*np.cos(theta)
			C = 0.5*(O0**2 - 3.0*K0mag**2)/O0
			K1mag = np.roots([A-B**2,-2.*B*C,1.-C**2])
			if K1mag[0] < 0.0 and K1mag[1] > 0.0:
				K1mag = K1mag[1]
			elif K1mag[0] > 0.0 and K1mag[1] < 0.0:
				K1mag = K1mag[0]
			else:
				raise ValueError("math domain error")

			K1 = K1mag*np.array([np.cos(theta),np.sin(theta)])
		else:
			# Wavenumbers centred at (K0/2,0), somewhat simplifies the maths
			A = (3.*K0mag*np.cos(theta)/O0)**2 - 3.
			C = 0.75*K0mag**2 + 1. - 0.25*O0**2

			# This isn't the magnitude of K1, it's the magnitude of K1 - K0/2...
			K1mag = np.sqrt(C/A)

			K1 = K0/2. + (K1mag*np.array([np.cos(theta),np.sin(theta)])).transpose()

	K2 = K0 - K1
	return {'k0':K0,'k1':K1,'k2':K2}

def wnsKPerp(kPerp,Te,omega0=srsUtils.omegaNIF):
	'''
	Calculate TPD wavenumbers given the perpendicular k vector and temperature

	Density is then fixed by the matching conditions. Calculation based on one from
	above, but uses light quantities to create dimensionless variables.
	'''
	vth = np.sqrt(const.k*Te/const.m_e)
	bth = vth/const.c

	kNorm = omega0/const.c
	KP = kPerp/kNorm

	coeffs = [3.*bth**4,
	          3.*bth**2*(-KP**2*bth**2 - 2*bth**2 + 2.) - 1,
	          3.*bth**2*(KP**2*bth**2 - 2*KP**2 + bth**2 - 2) + 0.25]

	OP = (-coeffs[1] - np.sqrt(coeffs[1]**2-4.*coeffs[0]*coeffs[2]))/(2.*coeffs[0])
	OP = np.sqrt(OP)

	K0Mag = np.sqrt(1. - OP**2)
	dKPara = np.sqrt(0.25*K0Mag**2 + KP**2)

	K0 = np.outer(K0Mag,np.array([1.0,0.0]))
	K1 = np.array([0.5*K0Mag + dKPara,KP]).transpose()
	K2 = np.array([0.5*K0Mag - dKPara,-KP]).transpose()
	ne = (omega0*OP)**2/const.e**2*(const.m_e*const.epsilon_0)

	return {'k0':K0*kNorm,'k1':K1*kNorm,'k2':K2*kNorm,'ne':ne}

	print(OP)

def wns(ne,Te,theta=None,omega0=srsUtils.omegaNIF,centredAngle=False,
        relativistic=True):
	'''
	Calculate TPD wavenumbers

	Calculates the wavenumbers of the waves involved in TPD. Uses the EPW
	warm-plasma dispersion relation for solving the	matching conditions.

	Units are SI unless otherwise specified.

	Parameters
	----------

	n: Electron density
	T: Electron temperature
	theta: angle with respect to propagation direction of pump EM wave
	omega0: EM wave frequency
	centredAngle: Decides the point about which to measure the angle. If True
	              the reference point is k = (k0/2,0). This guarantees that a
				  solution will be found regardless of theta. Otherwise (0,0).
	'''
	vth = np.sqrt(const.k*Te/const.m_e)
	Bth = vth/const.c
	op  = np.sqrt(ne/const.m_e/const.epsilon_0)*const.e
	if relativistic:
		op = op*math.sqrt(1.-2.5*Bth**2)
	ld  = vth/op

	O0 = omega0/op

	ks = wnsNodim(Bth,O0,theta,centredAngle)
	#print(ks)
	ks = { key: (ks[key].transpose()/ld).transpose() for key in ks }

	return ks

def omegasNodim(Bth,O0,theta=None,centredAngle=False):
	'''
	Calculate TPD wave frequencies

	Calculates the frequencies of the waves involved in TPD based on normalised
	parameters. Uses the EPW warm-plasma dispersion relation for solving the
	matching conditions.

	Units are SI unless otherwise specified.

	Parameters
	----------

	Bth: Thermal velocity relative to the speed of light
	O0: Laser frequency relative to the plasma frequency
	theta: angle with respect to propagation direction of pump EM wave
	centredAngle: Decides the point about which to measure the angle. If True
	              the reference point is k = (k0/2,0). This guarantees that a
				  solution will be found regardless of theta. Otherwise (0,0).
	'''
	ks = wnsNodim(Bth,O0,theta,centredAngle)
	kMags = { key:np.sqrt(np.sum(ks[key]**2,axis=-1)) for key in ks }

	os = {}
	os['k0'] = np.sqrt(1.0+(kMags['k0']/Bth)**2)
	os['k1'] = langmuir.reOmegaNodim(kMags['k1'])
	os['k2'] = langmuir.reOmegaNodim(kMags['k2'])

	return os

def omegas(ne,T,theta=None,omega0=srsUtils.omegaNIF,centredAngle=False):
	'''
	Calculate TPD frequencies

	Calculates the frequencies of the waves involved in TPD. Uses the EPW
	warm-plasma dispersion relation for solving the	matching conditions.

	Units are SI unless otherwise specified.

	Parameters
	----------

	n: Electron density
	T: Electron temperature
	theta: angle with respect to propagation direction of pump EM wave
	omega0: EM wave frequency
	centredAngle: Decides the point about which to measure the angle. If True
	              the reference point is k = (k0/2,0). This guarantees that a
				  solution will be found regardless of theta. Otherwise (0,0).
	'''
	vth = np.sqrt(const.k*T/const.m_e)
	op  = np.sqrt(ne/const.m_e/const.epsilon_0)*const.e

	Bth = vth/const.c
	O0 = omega0/op

	os = omegasNodim(Bth,O0,theta,centredAngle)
	os = { key: os[key]*op for key in os }

	return os

def growthRate(wns,E,omega0=srsUtils.omegaNIF,polAngle=0.0):
	vos = const.e*E/const.m_e/omega0
	#print(vos)

	k0Unit = np.transpose(np.transpose(wns['k0'])/np.linalg.norm(wns['k0'],axis=-1))
	k1Unit = np.transpose(np.transpose(wns['k1'])/np.linalg.norm(wns['k1'],axis=-1))
	theta = np.arccos(np.clip(np.sum(k0Unit*k1Unit,-1), -1.0, 1.0))

	fac = np.abs(np.cos(polAngle)*np.sin(theta))

	k1Mag = np.linalg.norm(wns['k1'],axis=-1)
	k2Mag = np.linalg.norm(wns['k2'],axis=-1)

	#print("fac: {:}, |k1|: {:}, |k2|: {:}".format(fac,k1Mag,k2Mag))

	return 0.25*k1Mag*vos*fac*np.abs(k2Mag**2 - k1Mag**2)/(k1Mag*k2Mag)

def landauCutoffDens(bth,relativistic=False,cutoff=0.3):
	def getKLd(Op):
		wns = wnsNodim(bth,1./Op)
		kfMag = np.sqrt(wns['k1'][0][0]**2 + wns['k1'][0][1]**2)

		return kfMag - cutoff

	Op = optimize.brentq(getKLd,0.01,0.48)
	ne = Op**2
	if relativistic:
		ne = ne/(1 - 2.5*bth**2)

	return ne

#*****************************************************************************#
#                      Absolute Growth Rate Calculations                      #
#*****************************************************************************#

# These are based on the following paper:
# Simon, A., On the Inhomogeneous Two-Plasmon Instability, Phys. Fluids
# Vol. 26 (1983) https://aip.scitation.org/doi/10.1063/1.864037

def omegaSimon(I,Te,Ln,ky,k0,n=0,omega0=srsUtils.omegaNIF,soln='approx'):
	op = 0.5*omega0
	vth = np.sqrt(const.k*Te/const.m_e)

	a = alpha(I,Ln,k0,omega0=omega0)
	b = beta(I,Te,Ln,k0,omega0=omega0)
	q = (ky/k0)**2

	O = omegaNodimSimon(q,a,b,n=n,soln=soln)
	o = op + 3.*vth**2*k0**2/omega0*O

	return o

def reOmegaSimon(I,Te,Ln,ky,k0,n=0,omega0=srsUtils.omegaNIF,soln='approx'):
	return np.real(omegaSimon(I,Te,Ln,ky,k0,n=n,omega0=omega0,soln=soln))

def imOmegaSimon(I,Te,Ln,ky,k0,n=0,omega0=srsUtils.omegaNIF,soln='approx'):
	''' γ = -Im(ω) '''
	return np.imag(omegaSimon(I,Te,Ln,ky,k0,n=n,omega0=omega0,soln=soln))

def omegaNodimSimon(q,a,b,n=0,soln='approx'):
	if soln == 'approx':
		O = highBetaApprox(q,a,b,n=n)
	else:
		O = calcO(q,a,b,n=n,onlySmallest=False)

	return O

omegaNodimSimon = np.vectorize(omegaNodimSimon)

def reOmegaNodimSimon(q,a,b,n=0,soln='approx'):
	return np.real(omegaNodimSimon(q,a,b,n=n,soln=soln))

def imOmegaNodimSimon(q,a,b,n=0,soln='approx'):
	''' Note that -Im(Ω) is the growth rate '''
	return np.imag(omegaNodimSimon(q,a,b,n=n,soln=soln))

def alpha(I,Ln,k0,omega0=srsUtils.omegaNIF):
	# Note the unconventional factor of two. So much time wasted...
	vos = 0.5*const.e*srsUtils.intensityToEField(I)/(const.m_e*omega0)

	a = 4.0*k0**2/omega0*vos*Ln
	return a

def beta(I,Te,Ln,k0,omega0=srsUtils.omegaNIF):
	vos = 0.5*const.e*srsUtils.intensityToEField(I)/(const.m_e*omega0)
	vth = np.sqrt(const.k*Te/const.m_e)

	b = 9.0*(vth**2*k0/omega0/vos)**2
	return b

def findK0(b,q):
	'''
	Finds the roots of the 12th order polynomial for K0 defined by eqs. 23 & 24

	Eq. 23 and 24 are used to eliminate (Ω0-K0), giving an equation in terms of
	β and q.
	'''
	# Array of coefficients
	coeffs = [b,
		      0.0,
			  1.5*b*(4.*q - 1.),
			  0.0,
	          15.*b*q**2 - 4.5*b*q + 15./16.*b + q,
			  0.0,
			  b*(20.0*q**3 - 3.*q**2 + 0.75*q - 5./16.),
			  0.0,
			  15.*b*q**4 + 3.*b*q**3 - 3*b*q**2/8. + 3*b*q/16. + 15.*b/256. - 2.*q**3 - q**2 - q/8.,
			  0.0,
			  3.*b*(1024.*q**5 + 768.*q**4 + 128.*q**3 - 32.*q**2 - 12.*q - 1.)/512.,
			  0.0,
			  b*q**6 + 3.*b*q**5/2. + 15.*b*q**4/16. + 5.*b*q**3/16. + 15.*b*q**2/256. + 3*b*q/512. + b/4096. + q**5 + q**4 + 3*q**3/8. + q**2/16. + q/256.]
	result = np.roots(coeffs)
	return result

def calcO0(K0,b,q):
	'''
	Uses K0 and equation 24 to calculate O0
	'''
	G = (K0**2 + q + 0.25)**2 - K0**2
	O0 = K0 + K0/b*q/G**2*((q + 0.25)**2 - K0**4)
	return O0

def calcd2Qdk2(o,k,q,a,b):
	'''
	Calculates second partial derivative of Q0 (defined in eq. 19) w.r.t. K
	'''
	G = (k**2 + q + 0.25)**2 - k**2
	dGdk = 4.0*k*(k**2 + q + 0.25) - 2.*k
	d2Gdk2 = 12.*k**2 + 4.*q - 1.0

	return a**2*(2.0*b - d2Gdk2*k**2*q/G**2 + 2.*dGdk**2*k**2*q/G**3 - 4.*dGdk*k*q/G**2 + 2*q/G)

def calcdQdO(o,k,a,b):
	'''
	Calculates partial derivative of Q0 (defined in eq. 19) w.r.t. Ω
	'''
	return 2.*a**2*b*(o-k)

def calcQ1(o,k,q,a,b):
	'''
	Calculates Q1 (defined in equation 20)
	'''
	G = (k**2 + q + 0.25)**2 - k**2
	return (1j*a*np.sqrt(b)/k)*(o - 2.*(o-k)*k**2*(q + k**2 - 0.25)/G)

def calcO1(o,k,q,a,b,n):
	'''
	Calculates the first order perturbation to Ω (eq. 25)
	'''
	d2Qdk2 = calcd2Qdk2(o,k,q,a,b)
	dQdO = calcdQdO(o,k,a,b)
	Q1 = calcQ1(o,k,q,a,b)
	return ((n+0.5)*np.sqrt(-2.0*d2Qdk2) - Q1)/dQdO

def calcO(q,a,b,n,onlySmallest=False):
	'''
	Calculates Ω to first order (Ω ~ Ω0 + Ω1)
	'''
	K0 = findK0(b,q)
	O0 = calcO0(K0,b,q)
	O1 = calcO1(O0,K0,q,a,b,n)

	O = O0 + O1
	O = O[np.where(np.real(O) > 0.0)]
	if onlySmallest:
		O = O[np.argpartition(np.abs(np.real(O)-0.5),2)[:2]]


	return O

def calcPsi(q,b):
	'''
	Calculates the (real) parameter Ψ from equation 32
	'''
	coeffs = [1.0,
	          3.0,
	          3.0 + 0.25/(b*q),
	          1.0]

	result = np.roots(coeffs)

	result = result[np.argpartition(np.abs(np.imag(result)),1)[:1]][0]

	return result

def highBetaApprox(q,a,b,n):
	'''
	Calculates Ω using the high-β approximation to the perturbation theory (eq. 36)
	'''
	psi = calcPsi(q,b)
	O = 0.5 + 1j/psi*np.sqrt(-q*psi)*(1 + 2.*psi - 2.*(2.*n + 1)*(1. + psi)**2*np.sqrt(b)/a*np.sqrt((2.*psi-1.)/psi))

	return O

def plotSimonFigs5and6():
	a = 40.0
	b = 25.0
	qs = np.linspace(1e-3,1.0)/b
	n = 0
	Os = [ calcO(q,a,b,n) for q in qs ]
	OsApprox = np.array([ highBetaApprox(q,a,b,n) for q in qs ])

	fig,ax = plt.subplots(1,2)
	for q,O in zip(qs,Os):
		#print(O)
		for o in O:
			ax[0].plot(b*q,math.sqrt(b)*np.real(o),'k.')
			ax[1].plot(b*q,math.sqrt(b)*np.imag(o),'k.')

	ax[0].plot(b*qs,math.sqrt(b)*np.real(OsApprox),'k-')
	ax[1].plot(b*qs,math.sqrt(b)*np.imag(OsApprox),'k-')

	for axis in ax:
		axis.set_xlabel(r'$bq$')
		axis.grid()
		axis.set_xlim(0.0,1.0)

	ax[0].axhline(math.sqrt(b)*0.5,linestyle='--',color='k')
	ax[0].set_ylabel(r'$b^{1/2}\mathrm{Re}(\Omega)$')
	ax[0].set_ylim(2.2,3.2)
	ax[1].set_ylabel(r'$b^{1/2}\mathrm{Im}(\Omega)$')
	ax[1].set_ylim(-0.3,0.2)

	fig.set_size_inches(6,3)
	fig.savefig('./tpd.pdf')

#*****************************************************************************#
#                  Yan Convective Growth Rate Calculations                    #
#*****************************************************************************#

def yanCouplingConstant(kPerp,Te,E,kinetic=True):
	'''
	Coupling constant for TPD convective growth model

	From R. Yan et al., "The linear regime of the two-plasmon decay instability
	in inhomogeneous plasmas", PoP 17 (2010)
	'''
	wns = wnsKPerp(kPerp,Te)
	kMag1 = np.sqrt(wns['k1'][:,0]**2 + wns['k1'][:,1]**2)
	kMag2 = np.sqrt(wns['k2'][:,0]**2 + wns['k2'][:,1]**2)
	ne = wns['ne']

	if kinetic:
		o1 = langmuir.reOmega(ne,Te,kMag1)
		o2 = langmuir.reOmega(ne,Te,kMag2)
	else:
		op  = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
		vth = np.sqrt(const.k*Te/const.m_e)
		ld  = vth/op
		o1 = op*np.sqrt(1. + 3.*(kMag1*ld)**2)
		o2 = op*np.sqrt(1. + 3.*(kMag2*ld)**2)

	vos = const.e*E/(const.m_e*srsUtils.omegaNIF)

	g2 = -1./16.*kPerp**2*vos**2*(o2/o1 - (kMag2/kMag1)**2)*(o1/o2 - (kMag1/kMag2)**2)
	g = np.sqrt(g2)

	return g

def yanDampingRates(kPerp,Te,Ln,kinetic=True,inh=True):
	'''
	Damping rate for each of the two TPD waves

	From R. Yan et al., "The linear regime of the two-plasmon decay instability
	in inhomogeneous plasmas", PoP 17 (2010)
	'''
	wns = wnsKPerp(kPerp,Te)
	kMag1 = np.sqrt(wns['k1'][:,0]**2 + wns['k1'][:,1]**2)
	kMag2 = np.sqrt(wns['k2'][:,0]**2 + wns['k2'][:,1]**2)
	ne = wns['ne']

	o0 = srsUtils.omegaNIF
	if kinetic:
		o1 = langmuir.reOmega(ne,Te,kMag1)
		o2 = langmuir.reOmega(ne,Te,kMag2)
	else:
		op  = np.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
		vth = np.sqrt(const.k*Te/const.m_e)
		ld  = vth/op
		o1 = op*np.sqrt(1. + 3.*(kMag1*ld)**2)
		o2 = op*np.sqrt(1. + 3.*(kMag2*ld)**2)

	# Landau damping rates
	gL1 = langmuir.imOmega(ne,Te,kMag1)
	gL2 = langmuir.imOmega(ne,Te,kMag2)

	# Effective damping due to inhomogeneity
	dndx = ne/Ln/srsUtils.nCritNIF
	kx1 = wns['k1'][:,0]
	kx2 = wns['k2'][:,0]
	gI1 = dndx*(o1*kx1/(2.*kMag1**2*(ne/srsUtils.nCritNIF)**2) + o0**2/(4.*kx1*o1))
	gI2 = dndx*(o2*kx1/(2.*kMag2**2*(ne/srsUtils.nCritNIF)**2) + o0**2/(4.*kx2*o2))

	if inh:
		g1 = gL1 + gI1
		g2 = gL2 + gL2
	else:
		g1 = gL1
		g2 = gL2

	return g1,g2

def yanGrowthRate(kPerp,Te,Ln,E,kinetic=True,inh=True):
	'''
	Convective growth rate for TPD

	Calculated using equation for convective/infinite homogeneous growth rate.
	See K. Nishikawa, "Parametric Excitation of Coupled Waves I: General
	Formulation", Journal of the Physical Society of Japan (24), 1968
	'''
	g0 = yanCouplingConstant(kPerp,Te,E,kinetic=kinetic)
	g1,g2 = yanDampingRates(kPerp,Te,Ln,kinetic=kinetic,inh=inh)

	gT = 0.5*(g1 + g2)
	gD = 0.5*(g2 - g1)

	g = np.sqrt(g0**2 + gD**2) - gT# - 0.5*gT + gD**2/(8.*g0**2)

	return g

def fixYan():
	import sympy as sp
	k1, k2, kp, o1, o2, v0, vth, kx1, kx2, x, t = sp.symbols('k_1, k_2, k_p, o_1, o_2, v_0, v_th, k_x1, k_x2 , x, t', real=True)
	dkdx1,dkdx2, v1, v2 = sp.symbols('dkdx1, dkdx2, v_1, v_2', real=True)
	I = sp.symbols('I')
	F1 = sp.Function('F1')(x,t)
	F2 = sp.Function('F2')(x,t)
	n0 = sp.Function('n_0')(x)

	# Equations 14 & 15
	P1 = -F1.diff(t)/n0 - sp.I*o1/n0*F1 + sp.I*kp*v0/(2*n0)*I*k2**2/k1**2*sp.conjugate(F2) + o1*kx1*F1/(n0*k1)**2*n0.diff(x)
	P2 = -F2.diff(t)/n0 - sp.I*o2/n0*F2 - sp.I*kp*v0/(2*n0)*I*k1**2/k2**2*sp.conjugate(F1) + o2*kx2*F2/(n0*k2)**2*n0.diff(x)

	# Equations 12 & 13
	dP1 = P1.diff(t) + sp.I*o1*P1 - 1/2*sp.I*kp*v0*sp.conjugate(P2)*I - F1 - 3*vth**2/n0*(k1**2*F1 + sp.I*dkdx1*F1 + 2*sp.I*kx1*F1.diff(x))
	dP1 = (sp.I*dP1*n0/(2*o1)).expand().subs(3*kx1*vth**2/o1,v1).subs(3*kx2*vth**2/o2,v2).collect(F1).subs(F1.diff(t,2),0)

	sp.pprint(dP1)

	#return dP1

if(__name__ == '__main__'):
	exit()
