#!/usr/bin/env python
# coding=UTF-8

import math as _math
import numpy as _np
import scipy.constants as _const
import scipy.special as _special
from scipy.optimize import minimize_scalar as _minimize_scalar
import numba as _numba

from srsUtils import srsUtils as _srsUtils
from srsUtils import misc as _misc

def speckleWidth(F,wl):
	return F*wl

def speckleLength(F,wl):
	return 10.0*F**2*wl

def gaussianBeamWidthAtAmp(z,E,k,w0,norm='beam'):
	'''
	Calculates radius from beam axis to reach a given amplitude

	Description
	-----------

	Finds r such that |E(z,r)| - |E| = 0 for given E and z

	Parameters
	----------

	z    : Axial distance from best focus (z0)
	E    : Minimum beam amplitude either as a fraction of the overall beam maximum
	       or the maximum at z.
	k    : Laser wavenumber
	w0   : Beam waist
	z0   : Value of z at best focus
	norm : Amplitude normalisation. If 'beam' then E is a fraction of the
	       maximum beam amplitude. If 'section' then E is a fraction of the
		   maximum amplitude at the given z.
	'''
	w = gaussianBeamWidth(z,k,w0)
	EMax = gaussianBeamAmplitude(z,0.0,k,w0)

	if norm == 'section': E = E*EMax

	return _np.sqrt(-w**2*_np.log(E*w/w0))

# TODO: does increasing r with other params fixed behave as expected?
# Should length should go to 0?
def gaussianBeamLengthAtAmp(r,E,k,w0,norm='beam'):
	'''
	Calculates distance along beam axis to reach a given amplitude

	Description
	-----------

	Finds z such that |E(z,r)| - |E| = 0 for given E and r
	
	Parameters
	----------
	
	r    : Radial distance from beam axis
	E    : Minimum beam amplitude either as a fraction of the overall beam maximum
	       or the maximum at z.
	k    : Laser wavenumber
	w0   : Beam waist
	z0   : Value of z at best focus
	norm : Amplitude normalisation. If 'beam' then E is a fraction of the
	       maximum beam amplitude. If 'section' then E is a fraction of the
		   maximum amplitude at the given r.
	'''
	EMax = gaussianBeamAmplitude(0.0,r,k,w0)

	if norm == 'section': E = E*EMax
	
	R = r/w0

	if R == 0.0:
		W = -1./E
	elif 2.0*(E*R)**2 < 1.0/_math.e:
		W = -1j*_np.sqrt(2./_special.lambertw(-2.0*(E*R)**2))*R
	else:
		raise ValueError("Math domain error")
	
	zr = k*w0**2/2.0

	return _np.real(zr*_np.sqrt(W**2-1))

def gaussianBeamField(z,r,t,k,omega,w0,z0=0.0,phase=0.0):
	'''
	Calculates a Gaussian beam's E-field at (z,r) and time t
	
	Parameters
	----------

	z     : Axial distance from best focus (z0)
	r     : Radial distance from beam axis
	t     : time
	k     : Laser wavenumber
	omega : Laser frequenecy
	w0    : Beam waist
	z0    : Value of z at best focus
	phase : Additional phase factor
	'''
	E = gaussianBeamAmplitude(z,r,k,w0,z0=z0)
	p = gaussianBeamPhase(z,r,t,k,omega0,w0,z0=z0,phase=phase)

	return E*_np.cos(p)

def gaussianBeamPhase(z,r,t,k,omega,w0,z0=0.0,phase=0.0):
	'''
	Calculates a Gaussian beam's phase at (z,r) and time t

	Parameters
	----------

	z     : Axial distance from best focus (z0)
	r     : Radial distance from beam axis
	t     : time
	k     : Laser wavenumber
	omega : Laser frequenecy
	w0    : Beam waist
	z0    : Value of z at best focus
	phase : Additional phase factor
	'''
	z = z-z0

	zr = k*w0**2/2.0
	R  = z*(1.0+(z/zr)**2)
	GP = _np.atan(z/zr) # Gouy phase

	# Correctly handles cases where z = 0.0 which results in division by zero
	return k*(z + _misc.div0(r**2,2.0*R,1.0)) - omega*t - GP + phase

def gaussianBeamAmplitude(z,r,k,w0,z0=0.0):
	'''
	Calculates a Gaussian beam's E-field amplitude at (z,r)

	Parameters
	----------

	z     : Axial distance from best focus (z0)
	r     : Radial distance from beam axis
	k     : Laser wavenumber
	w0    : Beam waist
	z0    : Value of z at best focus
	'''
	w = gaussianBeamWidth(z,k,w0,z0=z0)
	return w0/w*_np.exp(-(r/w)**2)

def gaussianBeamWidth(z,k,w0,z0=0.0):
	''' Calculates width of a Gaussian beam at distance z from best focus z0 '''
	z = z-z0
	zr = k*w0**2/2.0
	
	return w0*_np.sqrt(1.0+(z/zr)**2)

def invCumulAbundance(I,F,wl,S,I0=1.0,d=3,circ=True):
	'''
	Calculates the number of speckles above a given intensity

	From Garnier, PoP (1999)

	Parameters
	----------

	I : Minimum intensity of speckles to count (by default relative to mean intensity)
	I0 : Mean laser intensity (defaults to 1)
	F : F-number of beam
	wl : laser wavelength
	S : "Reference surface" (or volume if 3D), presumably region we're considering
	d : Dimensionality of field. I feel that we should always use 3 since
	    while our simulations are 2D, the experiment is 3D and we are interested
		in the statistics of the experiments...
	circ : If true, calculate for a circular phase plate, otherwise square
	'''
	rc = F*wl
	zc = _math.pi*wl*F**2

	IR = I/I0
	if d == 2:
		func = ((0.5+0.25*_math.pi)*IR + 0.5)*_np.exp(-IR)
	elif d == 3:
		func = (IR - 0.3)*_np.sqrt(IR)*_np.exp(-IR)
	else:
		raise ValueError("Can't calculate anything other than 2D or 3D")
	
	if circ:
		if d == 2:
			raise ValueError("I'm afraid we don't know that.")
			# In paper they give x-y distribution, not x-z
		else:
			norm = 5.0*_math.sqrt(_math.pi)**3/(48*_math.sqrt(6.0)*rc**2*zc)
	else:
		if d == 2:
			norm = _math.pi/(3.0*_math.sqrt(15.0)*rc*zc)
		else:
			norm = _math.sqrt(_math.pi)**3*_math.sqrt(5.0)/(27.0*rc**2*zc)
	
	return norm*S*func
