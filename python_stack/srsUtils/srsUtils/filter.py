#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import scipy.signal
import scipy.linalg
import scipy.optimize

def butterBandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = scipy.signal.butter(order, [low, high], btype='band')

	if not np.all(np.abs(np.roots(a))<1): print("WARNING: Bandpass filter is unstable")
	return b, a

def butterBandpassFilter(data, lowcut, highcut, fs, order=5):
	b, a = butterBandpass(lowcut, highcut, fs, order=order)
	y = scipy.signal.lfilter(b, a, data)
	return y

def butterLowpass(highcut, fs, order=5):
	nyq = 0.5 * fs
	high = highcut / nyq
	b, a = scipy.signal.butter(order, high, btype='lowpass')
	
	if not np.all(np.abs(np.roots(a))<1): print("WARNING: Lowpass filter is unstable")
	return b, a

def butterLowpassFilter(data, highcut, fs, order=5):
	b, a = butterLowpass(highcut, fs, order=order)
	y = scipy.signal.lfilter(b, a, data)
	return y

# Complex bandpass filter constructed from a real lowpass filter
def butterCBandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	
	# Prototype filter
	bp,ap = scipy.signal.butter(order, low, btype='lowpass',output='ba')
	if not np.all(np.abs(np.roots(ap))<1): print("WARNING: Prototype (lowpass) filter is unstable")
	
	# Frequency map filter
	bm,am = mapFreqToTwo(low,low,high)
	#if not np.all(np.abs(np.roots(am))<1): print("WARNING: Frequency map filter is unstable")
	# TODO: Check whether it is valid to ignore this!
	
	b,a = filtComp(bp,ap,bm,am)
	
	return b, a

def butterCBandpassFilter(data, lowcut, highcut, fs, order=5,fType='lf',padtype='odd',method=None,zi=None):
	b, a = butterCBandpass(lowcut, highcut, fs, order=order)
	if(fType == 'lf'):
		# This filtering scheme has non-linear phase but no ripple at end
		if(zi != None):
			zi = scipy.signal.lfilter_zi(b,a)
			y = scipy.signal.lfilter(b, a, data,zi=zi*data[0])
		else:
			y = scipy.signal.lfilter(b, a, data)
	elif(fType == 'ff'):
		# This filtering scheme has linear phase but ripple at start and end
		if(method == None):
			y = scipy.signal.filtfilt(b, a, data, padlen=len(data)-1,padtype=padtype)
		else:
			y,y1,y2 = _filtfilt_gust(b, a, data,irlen=5628)
	else: raise ValueError("Don't know what type of filter you want")
	return y

# Returns a frequency mapping filter in num,denom representation that maps
# ±ω to f_1,f_2 where f_1 < f_2
# Input:
#  - Old frequency o
#  - new frequencies n1 and n2 (n1 < n2)
def mapFreqToTwo(o,n1,n2):
	assert n1 < n2
	S = np.exp(-1j*0.5*math.pi*(n2+n1))
	alpha = np.sin(0.25*math.pi*(n2-n1-2.*o))/np.sin(0.25*math.pi*(n2-n1+2.*o))/S
	
	b = [S,alpha*S]
	a = [np.conj(alpha),1.]
	
	return b,a

# Returns a frequency mapping filter in num,denom representation that maps
# ω to n
# Input:
#  - Old frequency o
#  - New frequency n
def mapFreq(o,n):
	df = n-o
	b = [np.exp(-1j*math.pi*df)]
	a = [1.]
	
	return b,a

# Return a filter composed of two filters, i.e. f(g(z))
# Takes transfer function (b,a) representation of both filters and applies the
# first to the second
def filtComp(b1,a1,b2,a2):
	assert len(b2) == len(a2)
	
	# Convert b1,a1 to z,p,k representation
	z,p,k = scipy.signal.tf2zpk(b1,a1)
	
	# Calculate new z,p,k representation
	tz = np.array([ np.roots([ b2[i] - zi*a2[i] for i in range(len(b2))]) for zi in z ]).flatten()
	tp = np.array([ np.roots([ b2[i] - pi*a2[i] for i in range(len(b2))]) for pi in p ]).flatten()
	tk = np.prod([ (b2[0]-zi*a2[0])/(b2[0]-pi*a2[0]) for zi,pi in zip(z,p) ])*k
	
	# Convert back into b,a representation
	b,a = scipy.signal.zpk2tf(tz,tp,tk)
	
	return b,a

def convolveAxis(x,kernel,axis,mode='valid'):
	'''
	Convolves a multidimensional array with a 1D kernel along a given axis

	This is just a wrapper around np.convolve
	E.g. array of shape (Nt,Nx,Ny) can be convolved with a kernel of shape (Mt,)
	'''
	convolve = lambda a: np.convolve(a,kernel,mode=mode)

	return np.apply_along_axis(convolve, axis=axis, arr=x)

def winSincFilter(numtaps,cutoff,btype='lowpass'):
	'''
	Generates coefficients for a windowed sinc FIR filter
	
	Parameters
	----------
	
	numtaps : length of filter impulse response
	cutoff : cutoff frequency/frequencies, normalised to the Nyquist frequency
	btype : Type of filter, 'lowpass', 'highpass', 'bandpass', 'bandstop'
	
	Returns
	-------
	
	b : Array of filter coefficients
	
	TODO: Add option for selecting window function
	'''
	
	m = numtaps
	if(m % 2 == 0 and btype not in ('lowpass', 'bandpass', 'cbandpass')):
		raise ValueError('Need odd number of taps')

	if m % 2 == 0:
		ts = np.arange(-m//2,m//2)
	else:
		ts = np.arange(-m//2+1,m//2+1)

	cutoff = np.asarray(cutoff)

	if btype in ('lowpass', 'highpass'):
		if np.size(cutoff) != 1:
			raise ValueError('Must specify a single frequency cutoff')
		
		if btype == 'lowpass':
			b = cutoff*np.sinc(cutoff*ts)
		else:
			b = -cutoff*np.sinc(cutoff*ts)
			b[m//2] += 1.0
	elif btype in ('bandpass', 'bandstop', 'cbandpass', 'cbandstop'):
		if np.size(cutoff) != 2:
			raise ValueError('cutoff must specify start and stop frequencies')
		if cutoff[1] < cutoff[0]:
			raise ValueError('cutoff frequencies must be in ascending order')
		
		if btype == 'bandpass':
			b = cutoff[1]*np.sinc(cutoff[1]*ts) - cutoff[0]*np.sinc(cutoff[0]*ts)
		elif btype == 'bandstop':
			b = cutoff[0]*np.sinc(cutoff[0]*ts) - cutoff[1]*np.sinc(cutoff[1]*ts)
			b[m//2] += 1.0
		elif btype == 'cbandpass':
			f0 = 0.5*(cutoff[1]+cutoff[0])
			df = 0.5*(cutoff[1]-cutoff[0])
			b = df*np.sinc(df*ts)*np.exp(1j*math.pi*f0*ts)
		else:
			f0 = 0.5*(cutoff[1]+cutoff[0])
			df = 0.5*(cutoff[1]-cutoff[0])
			b = -df*np.sinc(df*ts)*np.exp(1j*math.pi*f0*ts)
			b[m//2] += 1.0
	else:
		raise NotImplementedError("{:} not implemented.".format(btype))
	
	b *= np.hanning(m)
	
	# Scale so that gain is 1.0 at pass frequency
	if np.size(cutoff) == 1:
		if btype == 'lowpass':
			scale_frequency = 0.0
		else:
			scale_frequency = 1.0
	else:
		if btype in ('bandpass', 'cbandpass'):
			scale_frequency = 0.5 * (cutoff[0] + cutoff[1])
		# TODO: Deal with cbandstop case
		else:
			scale_frequency = 0.0
	
	if btype in ('cbandpass', 'cbandstop'):
		c = np.exp(-1j * np.pi * ts * scale_frequency)
	else:
		c = np.cos(np.pi * ts * scale_frequency)
	
	s = np.abs(np.sum(b * c))
	#print(np.sum(b))
	#print(s)
	b /= np.conj(s)
	
	return b

def winSincFilter2D(nx,ny,cutoff,btype='lowpass'):
	'''
	Generates coefficients for a windowed sinc FIR filter
	
	Parameters
	----------
	
	nx,ny : length of filter impulse response in first and second dimension
	cutoff : cutoff frequency/frequencies, normalised to the Nyquist frequency
	btype : Type of filter, 'lowpass', 'highpass', 'bandpass', 'bandstop'
	
	Returns
	-------
	
	b : Array of filter coefficients
	
	TODO: Add option for selecting window function
	'''
	
	if(nx % 2 == 0 or ny % 2 == 0): raise ValueError('Need odd number of taps')

	x = np.arange(-nx//2+1,nx//2+1)
	y = np.arange(-ny//2+1,ny//2+1)
	
	X,Y = np.meshgrid(x,y)
	Z = np.sqrt(X**2 + Y**2)
	
	cutoff = np.asarray(cutoff)
	
	if btype in ('lowpass', 'highpass'):
		if np.size(cutoff) != 1:
			raise ValueError('Must specify a single frequency cutoff')
		
		if btype == 'lowpass':
			b = cutoff*np.sinc(cutoff*Z)
		else:
			b = -cutoff*np.sinc(cutoff*Z)
			b[nx//2,ny//2] = 1.0
	elif btype in ('bandpass', 'bandstop'):
		if np.size(cutoff) > 2 or np.size(cutoff) < 2:
			raise ValueError('cutoff must specify start and stop frequencies')
		if cutoff[1] < cutoff[0]:
			raise ValueError('cutoff frequencies must be in ascending order')
		
		if btype == 'bandpass':
			b = cutoff[1]*np.sinc(cutoff[1]*Z) - cutoff[0]*np.sinc(cutoff[0]*Z)
		else:
			b = cutoff[0]*np.sinc(cutoff[0]*Z) - cutoff[1]*np.sinc(cutoff[1]*Z)
			b[nx//2,ny//2] = 1.0
	else:
		raise NotImplementedError("{:} not implemented.".format(btype))
	
	b *= np.outer(np.hamming(nx),np.hamming(ny))
	import matplotlib.pyplot as plt
	plt.imshow(np.outer(np.hamming(nx),np.hamming(ny)))
	plt.show()
	
	# Scale so that gain is 1.0 at pass frequency
	if np.size(cutoff) == 1:
		if btype == 'lowpass':
			scale_frequency = 0.0
		else:
			scale_frequency = 1.0
	else:
		if btype == 'bandpass':
			scale_frequency = 0.5 * (cutoff[0] + cutoff[1])
		else:
			scale_frequency = 0.0
	
	cx = np.cos(np.pi * X * scale_frequency)
	cy = np.cos(np.pi * Y * scale_frequency)
	sx = np.sum(b * cx)
	sy = np.sum(b * cy)
	s = 0.5*(sx+sy)
	b /= s
	
	return b

def gaussianFilter(numtaps,cutoff,btype='lowpass'):
	'''
	Generates coefficients for a gaussian FIR filter
	
	Parameters
	----------
	
	numtaps : length of filter impulse response
	cutoff : cutoff frequency/frequencies, normalised to the Nyquist frequency
	btype : Type of filter, 'lowpass', 'highpass', 'bandpass', 'bandstop'
	
	Returns
	-------
	
	b : Array of filter coefficients
	
	TODO: Add option for selecting window function
	'''
	
	m = numtaps
	if(m % 2 == 0): raise ValueError('Need odd number of taps')

	ts = np.arange(-m//2+1,m//2+1)
	
	cutoff = np.asarray(cutoff)
	
	if btype in ('lowpass', 'highpass'):
		if np.size(cutoff) != 1:
			raise ValueError('Must specify a single frequency cutoff')
		
		if btype == 'lowpass':
			b = cutoff/math.sqrt(2.0*math.pi)*np.exp(-0.5*(cutoff*ts)**2)
		else:
			b = -cutoff/math.sqrt(2.0*math.pi)*np.exp(-0.5*(cutoff*ts)**2)
			b[m//2] += 1.0
	elif btype in ('bandpass', 'bandstop', 'cbandpass', 'cbandstop'):
		if np.size(cutoff) != 2:
			raise ValueError('cutoff must specify start and stop frequencies')
		if cutoff[1] < cutoff[0]:
			raise ValueError('cutoff frequencies must be in ascending order')
		
		if btype == 'bandpass':
			b = cutoff[1]/math.sqrt(2.0*math.pi)*np.exp(-0.5*(cutoff[1]*ts)**2) \
			  - cutoff[0]/math.sqrt(2.0*math.pi)*np.exp(-0.5*(cutoff[0]*ts)**2)
		elif btype == 'bandstop':
			b = cutoff[0]/math.sqrt(2.0*math.pi)*np.exp(-0.5*(cutoff[0]*ts)**2) \
			  - cutoff[1]/math.sqrt(2.0*math.pi)*np.exp(-0.5*(cutoff[1]*ts)**2)
			b[m//2] += 1.0
	else:
		raise NotImplementedError("{:} not implemented.".format(btype))
	
	# Scale so that gain is 1.0 at pass frequency
	if np.size(cutoff) == 1:
		if btype == 'lowpass':
			scale_frequency = 0.0
		else:
			scale_frequency = 1.0
	else:
		if btype in ('bandpass', 'cbandpass'):
			scale_frequency = 0.5 * (cutoff[0] + cutoff[1])
		# TODO: Deal with cbandstop case
		else:
			scale_frequency = 0.0
	
	if btype in ('cbandpass', 'cbandstop'):
		c = np.exp(-1j * np.pi * ts * scale_frequency)
	else:
		c = np.cos(np.pi * ts * scale_frequency)
	
	s = np.abs(np.sum(b * c))
	b /= np.conj(s)
	
	return b

def plotFIRFilterInfo1D(b):
	import matplotlib.pyplot as plt
	
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.rc('figure', autolayout=True)
		
	m = len(b)
	ts = np.arange(-m//2+1,m//2+1)
	bPad = np.pad(b,(m,m),mode='constant',constant_values=0)
	bFT = np.fft.fftshift(np.fft.fft(bPad))
	fs = 2.0*np.fft.fftshift(np.fft.fftfreq(len(bPad)))

	fig,ax = plt.subplots(4,1)
	
	# Plot impulse response
	impulse = np.zeros(m)
	impulse[m//2] = 1.0
	ax[0].set_title('Impulse response')
	ax[0].plot(ts,np.real(b))
	ax[0].plot(ts,np.imag(b))
	ylim = ax[0].get_ylim()
	ax[0].plot(ts,impulse,'k')
	ax[0].set_xlim(ts[0],ts[-1])
	ax[0].set_ylim(ylim)
	ax[0].set_xlabel('Sample no.')
	ax[0].grid()
	
	# Calculate and plot step response
	step = (1+np.sign(np.arange(-m+1,m)+1))//2
	stepResponse = np.convolve(b,step,mode='valid')
	ax[1].set_title('Step reponse')
	ax[1].plot(ts,np.real(stepResponse))
	ax[1].plot(ts,np.imag(stepResponse))
	ax[1].plot(ts,np.real(step[m//2:-m//2+1]),'k')
	ax[1].set_xlim(ts[0],ts[-1])
	ax[1].set_xlabel('Sample no.')
	ax[1].grid()
	
	# Plot frequency response (amplitude)
	ax[2].set_title('Frequency response (amplitude)')
	ax[2].plot(fs,np.abs(bFT))
	ax[2].set_yscale('log')
	ax[2].set_xlim(-1.0,1.0)
	ax[2].set_xlabel('Frequency /$f_{\mathrm{nyq}}$')
	ax[2].grid()
	
	# Plot frequency response (phase) 
	ax[3].set_title('Frequency response (phase)')
	ax[3].plot(fs,np.unwrap(np.angle(bFT)))
	ax[3].set_xlabel('Frequency /$f_{\mathrm{nyq}}$')
	ax[3].grid()
	
	plt.show()
	plt.close()
