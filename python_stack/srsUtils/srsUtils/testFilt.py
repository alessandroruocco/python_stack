#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt

import srsUtils

# Plots signal and its DFT with various types of filter applied

N = 1000
x = np.linspace(0,N-1,N)
dx = x[1]-x[0]
L = x[-1]-x[0]
xFT = np.fft.fftshift(np.fft.fftfreq(N,dx))

f1 = 0.05123134
a1 = 1.5
f2 = 0.1282
a2 = 1.0

# Generate original signal
signal = a1*np.sin(math.pi/dx*f1*x) + a2*np.cos(math.pi/dx*f2*x)
signalFT = (2./N)*np.abs(np.fft.fftshift(np.fft.fft(signal)))

# Generate filter coefficients
nTaps = 201
lowPass = srsUtils.filter.winSincFilter(nTaps,0.5*(f1+f2),btype='lowpass')
highPass = srsUtils.filter.winSincFilter(nTaps,0.5*(f1+f2),btype='highpass')
bandPass = srsUtils.filter.winSincFilter(nTaps,(0.5*(f1+f2),f2+0.5*(f2-f1)),btype='bandpass')
bandStop = srsUtils.filter.winSincFilter(nTaps,(0.5*(f1+f2),f2+0.5*(f2-f1)),btype='bandstop')
cBandPass = srsUtils.filter.winSincFilter(nTaps,(0.5*(f1+f2),f2+0.5*(f2-f1)),btype='cbandpass')
cBandStop = srsUtils.filter.winSincFilter(nTaps,(0.5*(f1+f2),f2+0.5*(f2-f1)),btype='cbandstop')

#srsUtils.filter.plotFIRFilterInfo1D(lowPass)

# Apply filters to signal
signalLP = np.convolve(signal,lowPass,mode='same')
signalHP = np.convolve(signal,highPass,mode='same')
signalBP = np.convolve(signal,bandPass,mode='same')
signalBS = np.convolve(signal,bandStop,mode='same')
signalCBP = np.convolve(signal,cBandPass,mode='same')
signalCBS = np.convolve(signal,cBandStop,mode='same')

# Take DFT of filtered signals
signalLPFT = (2./N)*np.abs(np.fft.fftshift(np.fft.fft(signalLP)))
signalHPFT = (2./N)*np.abs(np.fft.fftshift(np.fft.fft(signalHP)))
signalBPFT = (2./N)*np.abs(np.fft.fftshift(np.fft.fft(signalBP)))
signalBSFT = (2./N)*np.abs(np.fft.fftshift(np.fft.fft(signalBS)))
signalCBPFT = (2./N)*np.abs(np.fft.fftshift(np.fft.fft(signalCBP)))
signalCBSFT = (2./N)*np.abs(np.fft.fftshift(np.fft.fft(signalCBS)))

filtSignals = np.zeros((7,N),dtype=np.complex128)
filtSignals[0] = signal
filtSignals[1] = signalLP
filtSignals[2] = signalHP
filtSignals[3] = signalBP
filtSignals[4] = signalBS
filtSignals[5] = signalCBP
filtSignals[6] = signalCBS

filtSignalsFT = np.zeros((7,N),dtype=np.complex128)
filtSignalsFT[0] = signalFT
filtSignalsFT[1] = signalLPFT
filtSignalsFT[2] = signalHPFT
filtSignalsFT[3] = signalBPFT
filtSignalsFT[4] = signalBSFT
filtSignalsFT[5] = signalCBPFT
filtSignalsFT[6] = signalCBSFT


fig,axes = plt.subplots(3,7)

# Plot filtered signals in time domain
for i,ax in enumerate(axes[0,:]):
	ax.plot(x,np.real(filtSignals[i]))
	ax.plot(x,np.imag(filtSignals[i]))
	ax.set_ylim(-3.,3.)

# Plot DFT of signals
for i,ax in enumerate(axes[1,:]):
	ax.plot(xFT,filtSignalsFT[i])
	ax.set_xlim(-0.15,0.15)
	ax.set_ylim(0,2.0)

# Plot log(DFT) of signals
for i,ax in enumerate(axes[2,:]):
	ax.plot(xFT,filtSignalsFT[i])
	ax.set_xlim(-0.15,0.15)
	ax.set_yscale('log')

for ax in axes.flatten():
	ax.grid()

axes[0][0].set_title('\#nofilter')
axes[0][1].set_title('Lowpass')
axes[0][2].set_title('Highpass')
axes[0][3].set_title('Bandpass')
axes[0][4].set_title('Bandstop')
axes[0][5].set_title('cBandpass')
axes[0][6].set_title('cBandstop')

axes[0][0].set_ylabel('Signal')
axes[1][0].set_ylabel('Signal FT')
axes[2][0].set_ylabel('Signal FT (logged)')

fig.set_size_inches((14,6))
fig.savefig('./testFilter.pdf')
