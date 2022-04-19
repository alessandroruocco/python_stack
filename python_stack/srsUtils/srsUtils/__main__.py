
def main():
	import argparse
	import matplotlib.pyplot as plt
	import numpy as np
	import scipy.constants as const

	import srsUtils

	parser = argparse.ArgumentParser()
	parser.add_argument('--sType',type=str)
	parser.add_argument('-ne',type=float)
	parser.add_argument('--neRange',type=float,nargs=2)
	parser.add_argument('-nencr',type=float)
	parser.add_argument('--nencrRange',type=float,nargs=2)
	parser.add_argument('-Te',type=float)
	parser.add_argument('-TekeV',type=float)
	parser.add_argument('-omega0',type=float)
	parser.add_argument('-wlVac0',type=float)
	parser.add_argument('--plot',action='store_true')
	args = parser.parse_args()

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.rc('font', serif=['computer modern roman'])
	plt.rc('font', size=10)
	plt.rc('figure', autolayout=True)
	
	plotWidthIn = 2.93192
	
	if(args.sType != None):
		if(args.sType == 'fSRS' or args.sType == 'f'):
			bSRS = False
		elif(args.sType == 'bSRS' or args.sType == 'b'):
			bSRS = True
		else:
			raise ValueError('Unrecognised value for sType argument')
	else:
		bSRS = True
	
	if(args.wlVac0 != None):
		args.omega0 = const.c*2*math.pi/args.wlVac0
	
	if(args.omega0 != None):
		omega0 = args.omega0
	else:
		omega0 = srsUtils.omegaNIF
	
	nCrit = omega0**2*const.m_e*const.epsilon_0/const.e**2
	
	if(args.Te != None):
		T = args.Te
	elif(args.TekeV != None):
		T = args.TekeV*1e3*const.e/const.k
	else:
		T = 1e3*const.e/const.k
	
	if(args.ne != None):
		n = args.ne
	elif(args.nencr != None):
		n = args.nencr*nCrit
	elif(args.neRange != None):
		n = srsUtils.optimiseRatios(args.neRange[0],args.neRange[1],T,omega0,bSRS)
	elif(args.nencrRange != None):
		n = srsUtils.optimiseRatios(args.nencrRange[0]*nCrit,args.nencrRange[1]*nCrit,T,omega0,bSRS)
	else:
		n = srsUtils.optimiseRatios(0.17*nCrit,0.21*nCrit,T,omega0,bSRS)
		#n = srsUtils.optimiseRatios(0.1*nCrit,0.2*nCrit,T,omega0)

	srsUtils.printProperties(n,T,omega0)
	
	if not args.plot: exit()
	
	n1 = 1e-10*nCrit
	n2 = 0.249*nCrit
	fig = srsUtils.plotRatioDiffSum2(n1,n2,0.1*T,10.0*T,omega0,numNs=1000,log=False)
	fig.set_size_inches(4.0,3.5)
	fig.tight_layout()
	fig.savefig('./rdsVsn.pdf')

if __name__ == "__main__":
	main()
