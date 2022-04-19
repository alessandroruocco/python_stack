#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import scipy.integrate
import time
import numba

import sdfUtils
import sdf

from srsUtils import srsUtils
from srsUtils import langmuir
from srsUtils import misc

#mpl.style.use('classic')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('figure', autolayout=True)

@numba.jit(nopython=True)
def S(K,dx,m=1):
	arg = 0.5*K*dx/math.pi
	return np.sinc(arg)**(m+1)

def epsInvPIC(O,K,dx,maxP,m=1):
	kps = K + np.arange(-maxP,maxP+1)*2.0*math.pi/dx
	kappa = K*np.sinc(K*dx/math.pi)

	zSum = np.sum(S(kps,dx,m)**2*kappa*kps/kps**2*langmuir.friedConte(O/(math.sqrt(2.0)*np.abs(kps)),d=1))

	Kal = K*np.sinc(0.5*K*dx/math.pi)

	return Kal**2/(Kal**2 - 0.5*zSum)

epsInvPIC = np.vectorize(epsInvPIC)

def f(v):
	return np.exp(-0.5*v**2)/math.sqrt(2.0*math.pi)

def spectrum(o,k,dx,dt,w,vth,m=1,maxP=100):
	epsInvAbs = abs(epsInvPIC(o,k,dx,maxP=maxP,m=m))
	
	kps = k + np.arange(-maxP,maxP+1)*2.0*math.pi/dx
	Kal = k*np.sinc(0.5*k*dx/math.pi)

	og = 2.0*math.pi/dt
	opg = o + og
	omg = o - og

	norm = 0.5*w*const.m_e*vth**2/const.epsilon_0
	return norm*epsInvAbs*np.sqrt(np.sum(S(kps,dx,m)**2*(f(o/kps) + f(opg/kps) + f(omg/kps))))/np.abs(Kal)

spectrum = np.vectorize(spectrum)

def spectrumK(k,dx,w,vth,m=1,maxP=100):
	kps = k + np.arange(-maxP,maxP+1)*2.0*math.pi/dx
	sumS = np.sum(S(kps,dx,m)**2)
	Kal = k*np.sinc(0.5*k*dx/math.pi)

	return 0.5*w*const.m_e*vth**2/const.epsilon_0*np.sqrt(sumS/(sumS + Kal**2))

spectrumK = np.vectorize(spectrumK)

def spectrumK2(k,dx,m=1,maxP=100):
	Kal = k*np.sinc(0.5*k*dx/math.pi)

	return Kal*(1.0 - np.imag(epsInvPIC(0.0,k,dx,maxP,m=m)))
#	#return np.abs(k)*0.5*(1.0 - np.real(srsUtils.langmuir.epsInv(0.0,k)))
#	Kal = k*np.sinc(0.5*k*dx/math.pi)
#	return np.sqrt(np.abs(-1j*2.0*0.5*(1.0 - epsInvPIC(0.0,k,dx,maxP=100))))/const.epsilon_0

spectrumK2 = np.vectorize(spectrumK2)

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	
	parser.add_argument('dataDir')
	parser.add_argument('ne',type=float)
	parser.add_argument('Te',type=float)
	parser.add_argument('ppc',type=int)

	parser.add_argument('--kMax',type=float)
	parser.add_argument('--oMax',type=float)

	parser.add_argument('--fontSize',type=float)
	parser.add_argument('-o','--output')
	parser.add_argument('--figSize',type=float,nargs=2)

	args = parser.parse_args()

	ne = args.ne*srsUtils.nCritNIF
	Te = args.Te*srsUtils.TkeV
	op = math.sqrt(ne*const.e**2/(const.m_e*const.epsilon_0))
	vth = math.sqrt(const.k*Te/const.m_e)
	ld = vth/op

	# Read simulation data
	print("\nReading simulation data")
	t1 = time.time()
	files = sdfUtils.listFiles(args.dataDir,prefix='regular_')[10:]
	subset = 'all_ss'
	requiredData = [ 'Electric_Field_Ex_Acc_{ss}'.format(ss=subset) ]
	requiredData = [ 'Electric_Field_Ex' ]

	data = sdfUtils.readMultipleAcc(files,requiredData,subset,spaceSlices=None,onMissing='pad')

	ex = data['Electric_Field_Ex']
	ts = data['time']
	x  = data['space'][0]
	print("Finished reading data, took {:.3f}s".format(time.time()-t1))

	dx = x[1]-x[0]
	dt = ts[1]-ts[0]

	w = ne*dx/args.ppc

	K = 2.0*math.pi*np.fft.fftshift(np.fft.fftfreq(x.shape[0],dx))*ld
	O = 2.0*math.pi*np.fft.fftshift(np.fft.fftfreq(ts.shape[0],dt))/op

	if args.kMax is not None:
		kMax = args.kMax
	else:
		kMax = K[-1]
	
	if args.oMax is not None:
		oMax = args.oMax
	else:
		oMax = O[-1]

	print(K[-1]/ld*dx)
	print(K[-1]*dx/(2.0*math.pi*ld))
	print(O[-1]*op*dt)
	print(O[-1]*dt/(2.0*math.pi/op))
	
	# Calculate simulation noise spectrum
	#xWindow = np.hanning(x.shape[0])
	xWindow = np.ones(x.shape[0])
	tWindow = np.outer(np.hanning(ts.shape[0]),np.ones(x.shape[0]))
	
	print("\nTaking FFTs of simulation data")
	t1 = time.time()
	ex_xFT = np.abs(np.fft.fftshift(np.fft.fft(xWindow*ex,axis=1),axes=1))
	ex_tFT = np.abs(np.fft.fftshift(np.fft.fft(tWindow*ex,axis=0),axes=0))

	ex_xtFT = np.abs(np.fft.fftshift(np.fft.fftn(xWindow*tWindow*ex)))
	print("Finished FFTing data, took {:.3f}s".format(time.time()-t1))

	
	# Calculate theoretical noise spectrum
	print("\nCalculating theoretical noise spectrum")
	t1 = time.time()
	kg = np.linspace(-kMax,kMax,100)
	og = np.linspace(-oMax,oMax,1000)
	Og,Kg = np.meshgrid(og,kg)
	noiseK = spectrumK(kg,dx/ld,w,vth,m=2,maxP=10)
	noise = spectrum(Og,Kg,dx/ld,dt*op,w,vth,m=2,maxP=10)
	print("Finished calculating noise spectrum, took {:.3f}s".format(time.time()-t1))

	fig,ax = plt.subplots(3,2)

	#extent = srsUtils.misc.getExtent(ts/1e-12,x/1e-6)
	#ax[0][0].imshow(ex,cmap='RdBu_r',interpolation='none',origin='lower',extent=extent,aspect='auto')
	
	# Plot mean noise spectrum vs. k
	ax[0][1].set_title(r'$\langle E_x(k) \rangle$')
	ax[0][1].plot(K,np.sqrt(np.mean(ex_xFT**2,axis=0)))
	ax[0][1].plot(kg,noiseK)

	ax[0][1].set_xlabel(r'$k\lambda_D$')
	ax[0][1].set_ylabel(r'$\langle E_x(k) \rangle$')

	ax[0][1].set_xlim(-kMax,kMax)
	ax[0][1].set_ylim(0.0,ax[0][1].get_ylim()[1])
	
	ax[0][1].grid()
	
	# Plot mean noise spectrum vs omega
	ax[1][0].set_title(r'$\langle E_x(\omega) \rangle$')
	ax[1][0].plot(O,np.mean(ex_tFT,axis=1))

	ax[1][0].set_xlabel(r'$\omega/\omega_{\mathrm{pe}}$')
	ax[1][0].set_ylabel(r'$\langle E_x(\omega) \rangle$')

	ax[1][0].set_xlim(-oMax,oMax)
	ax[1][0].set_ylim(0.0,ax[1][0].get_ylim()[1])
	
	ax[1][0].grid()

	# Plot theoretical noise spectrum
	#ax[1][1].plot(kg,np.mean(noise,axis=1))
	ax[1][1].plot(kg,noiseK)
	ax[1][1].set_xlim(-kMax,kMax)
	ax[1][1].set_ylim(0.0,ax[1][1].get_ylim()[1])

	ax[1][1].grid()

	extent = srsUtils.misc.getExtent(K,O)
	vmax = np.max(ex_xtFT)
	norm = colors.LogNorm(vmin=vmax/1e6,vmax=vmax)
	im = ax[2][0].imshow(ex_xtFT,origin='lower',
	                     interpolation='none',aspect='auto',cmap='viridis',
	                     norm=norm,extent=extent)
	cb = fig.colorbar(im,orientation='vertical',ax=ax[2][0])
	
	ax[2][0].set_xlim(-kMax,kMax)
	ax[2][0].set_ylim(0.0,oMax)
	ax[2][0].set_xlabel(r'$k\lambda_D$')
	ax[2][0].set_ylabel(r'$\omega/\omega_p$')
	
	extent = srsUtils.misc.getExtent(kg,og)
	vmax = np.max(noise)
	norm = colors.LogNorm(vmin=vmax/1e6,vmax=vmax)
	im = ax[2][1].imshow(noise.transpose(),origin='lower',
	                     interpolation='none',aspect='auto',cmap='viridis',
	                     norm=norm,extent=extent)
	cb = fig.colorbar(im,orientation='vertical',ax=ax[2][1])
	
	ax[2][1].set_xlim(-kMax,kMax)
	ax[2][1].set_ylim(0.0,oMax)
	ax[2][1].set_xlabel(r'$k\lambda_D$')
	ax[2][1].set_ylabel(r'$\omega/\omega_p$')

	if args.fontSize:
		import matplotlib as mpl
		mpl.rcParams.update({'font.size':args.fontSize})
	
	if args.output:
		if args.figSize:
			fig.set_size_inches(args.figSize)
		fig.savefig(args.output)
	else:
		plt.show()
