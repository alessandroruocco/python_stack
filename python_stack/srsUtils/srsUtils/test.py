#!/usr/bin/env python
# coding=UTF-8
import srsUtils
import langmuir
import misc
from matplotlib import pyplot as plt, colors
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('figure', autolayout=True)

bth = 0.2
a0  = 0.4
O0  = 2.5

#k1,k2 = (0.08,0.12)
k1,k2 = (0.1,0.8)
o1,o2 = (0.0,O0)

Nk = 200
No = 200

Ks = np.linspace(k1,k2,Nk)
Os = np.linspace(o1,o2,No)

KS,OS = np.meshgrid(Ks,Os)
dr = lambda k,o: srsUtils._srsLWDispRelFunc(k,o,bth,a0,O0)
drV = np.vectorize(dr)
Z = dr(KS,OS)
zr = np.real(Z)
zi = np.imag(Z)
za = np.abs(Z)
extent = misc.getExtent(Ks,Os)

fig,ax = plt.subplots(1,3)

norm = colors.SymLogNorm(linthresh=0.01, linscale=1)
im = ax[0].imshow(zr,origin='lower',interpolation='none',extent=extent,aspect='auto',norm=norm,cmap='RdBu_r')

norm = colors.SymLogNorm(linthresh=0.01, linscale=1)
im = ax[1].imshow(zi,origin='lower',interpolation='none',extent=extent,aspect='auto',norm=norm,cmap='RdBu_r')

norm = colors.LogNorm()
im = ax[2].imshow(za,origin='lower',interpolation='none',extent=extent,aspect='auto',norm=norm,cmap='inferno_r')

#plt.colorbar(im,orientation='horizontal')
plt.show()
#exit()

dO = 0.6
O = langmuir.reOmegaNodim(np.mean(Ks))
srsUtils.animSRSLWDispRelFunc(Ks,bth,a0,O0,lims=((O-dO,O+dO),(-dO,dO)))
exit()

#
#for K in Ks:
#	O = langmuir.reOmegaNodim(K)
#	fig = srsUtils.plotSRSLWDispRelFuncVsOmega(K,bth,a0,O0,lims=((O-0.1,O+0.1),(-0.1,0.1)))
#	fig.suptitle('$K={:.3f}$'.format(K))
#	fig.savefig('{:.3f}.pdf'.format(K))
#	plt.close()

Ks = np.linspace(k1,k2,1000)
Os = [ srsUtils.srsLWDispRelNodim(K,bth,a0,O0) for K in Ks ]

fig,ax = plt.subplots(1,2)
ax[0].plot(Ks,np.real(Os))
ax[0].grid()

ax[1].plot(Ks,np.imag(Os))
ax[1].grid()

plt.show()
