# coding=UTF-8

import sympy as sp
import numpy as np
import os
import imp
import inspect
import re

from srsUtils import srsUtils

_scriptDir = os.path.abspath(os.path.dirname(__file__))
_funcCacheDir = os.path.join(_scriptDir,'_funcCache')

_funcCache = {}


# Generate numerical function handler for arbitrary symbolic function
def genFuncHandler(f,nSymArgs):
	prefix = inspect.getmodule(f).__name__+'_'+f.__name__+'_'
	def fNum(*args):
		numArgs = args[nSymArgs:]
		symArgs = args[:nSymArgs]
		funcKey = prefix+'_'.join([ repr(a) for a in symArgs ])
		try:
			return _funcCache[funcKey](*numArgs)
		except KeyError:
			try:
				_funcCache[funcKey] = readFuncFromDisk(funcKey)
				return _funcCache[funcKey](*numArgs)
			except (ImportError,IOError):
				symbols,expr = f(*symArgs)
				#print(symbols)
				#print(expr)
				outDir = os.path.join(_funcCacheDir,funcKey)
				_funcCache[funcKey] = genBinFromSympy(expr,symbols,outDir)
				return _funcCache[funcKey](*numArgs)
	
	return fNum

def readFuncFromDisk(funcKey):
	#print(_funcCacheDir)
	#print(funcKey)
	funcDir = os.path.join(_funcCacheDir,funcKey)
	modFile = [ f for f in os.listdir(funcDir) ]
	modFile = [ f for f in modFile if f.endswith('.so') ]
	if len(modFile) == 1:
		modFile = modFile[0]
	else:
		raise IOError('Couldn\'t find shared object file')
	
	wrapper_module = imp.load_dynamic('wrapper_module',modFile)
	
	return wrapper_module.autofunc

def genBinFromSympy(expr,args,outDir):
	'''
	Wrapper to sympy autowrap to enable complex calculations

	Description
	-----------

	Uses autowrap to generate fortran code for an expression, and then replaces
	all REAL variable declarations with COMPLEX ones and re-compiles using f2py

	Parameters
	----------

	These are simply passed to autowrap. See autowrap for reference.
	'''
	import os
	import imp
	from sympy.utilities import autowrap
	import subprocess

	os.makedirs(outDir)
	codeWrapper = autowrap.CodeWrapper
	counter = codeWrapper._module_counter
	try:
		autowrap.autowrap(expr,args=args,backend='f2py',tempdir=outDir)
	except IOError:
		pass
	names = { 'h':os.path.join(outDir,'wrapped_code_{c}.h'.format(c=counter)),
			  'f90':os.path.join(outDir,'wrapped_code_{c}.f90'.format(c=counter)) }
	
	files = {}
	for ext,name in names.items():
		with open(name,'r') as nameFile:
			lines = nameFile.readlines()
			files[ext] = lines
	
	for ext,lines in files.items():
		lines = [ l.replace('REAL*8','COMPLEX*16') for l in lines ]
		files[ext] = lines
	
	for ext,lines in files.items():
		os.remove(names[ext])
		with open(names[ext],'w') as nameFile:
			nameFile.writelines(lines)
	
	modFile = [ f for f in os.listdir(outDir) if f.endswith('.so') ]
	if len(modFile) == 1:
		modFile = modFile[0]
	else:
		raise IOError('Couldn\'t find shared object file')

	os.remove(os.path.join(outDir,modFile))

	process = subprocess.Popen(['f2py','-c',os.path.basename(names['f90']),'-m','wrapper_module'],cwd=outDir)
	process.wait()
	if process.returncode != 0:
		raise EnvironmentError("f2py failed to generate *.so file")

	modFile = [ f for f in os.listdir(outDir) if f.endswith('.so') ]
	if len(modFile) == 1:
		modFile = modFile[0]
	else:
		raise IOError('Couldn\'t find shared object file')
	wrapper_module = imp.load_dynamic('wrapper_module',os.path.join(outDir,modFile))

	return wrapper_module.autofunc
