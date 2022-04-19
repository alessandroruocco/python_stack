# coding=UTF-8

class inputBlock(dict):
	''' A class representing a block in an input deck

	Description
	-----------

	A simple subclass of dict, with extra attribute blockName
	'''
	def __init__(self,*args,**kwargs):
		if len(args) != 0 and 'blockName' in args[0]:
			self.blockName = args[0].pop('blockName')
		elif 'blockName' in kwargs:
			self.blockName = kwargs.pop('blockName')
		else:
			self.blockName = None

		dict.__init__(self,*args,**kwargs)
		#print(self.blockName)

	def __str__(self):
		newLineChar = '\n'
		indent = '    '

		lines = []
		# Define block start
		lines.append('begin:'+self.blockName)

		# Add parameters (in alphabetical order)
		# Hack to put laser block boundary parameter before other properties
		# TODO: Sort this out (maybe do rest of PhD first...)
		for p in sorted(self.keys()):
			val = self[p]
			# If we have boolean type then make sure to print 'T' or 'F'
			if(type(val) == bool):
				line = indent+p+' = '+str(val)[0]
			else:
				line = indent+p+' = '+str(val)

			# Make sure lines aren't more than 80 characters long by splitting them up
			# Split at following characters (not + and - as can appear in e.g. 1e-6)
			breakChars = ['=',',','*','/','^',' ']
			exithere = False
			while len(line) > 80:
				breakInds = [ line.rfind(c,len(indent)-1,79) for c in breakChars
				                  if line.rfind(c,len(indent)-1,79) != -1 ]
				breakInd = max(breakInds)

				lines.append(line[:breakInd].ljust(79)+'\\')
				line = indent+line[breakInd:]
			lines.append(line)

		#for l in lines: print(l)
		# Define block end
		lines.append('end:'+self.blockName)

		return newLineChar.join(lines)

