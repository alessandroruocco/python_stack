from .knownSystems import knownSystems

class resourceRequest:
	''' Class for specifying a slurm resoure request '''

	def __init__(self,
		         walltime,
				 pmem,
		         nodes=None,ppn=None,procs=None,
		         queue=None,
				 system='tinis'):
		''' Object initialisation function
	
		Input
		-----
	
		walltime : number of seconds requested, rounded to nearest integer
		pmem     : memory per processor
		nodes    : number of nodes requested
		ppn      : number of processors per node
		procs    : total number of processors requested (overrides nodes and ppn)
		queue    : queue on which to run
		system   : the system on which this will be run
		           If this is set to None then no checks will be made on the validity
		           of the resource requests
		'''
		# Check arguments
		# Basic sanity checks first
		if walltime <= 0:
			raise ValueError("Requested walltime must be greater than zero")
		if pmem <= 0:
			raise ValueError("Amount of memory requested must be greater than 0MB")
	
		if nodes == None and ppn == None and procs == None:
			raise ValueError("Please either specify the number of nodes and processors per node or a total number of processors")
		
		if not procs:
			if nodes <= 0 or ppn <= 0:
				raise ValueError("Number of nodes and processors per node must be greater than zero")
	
		# Record resource request data
		self._resources = {
			'walltime' : int(round(walltime)),
			'pmem'     : pmem,
			'nodes'    : nodes,
			'ppn'      : ppn,
			'procs'    : procs
		}
		self.queue  = queue
		self.system = system
		# Convenience flag indicating whether we want cores grouped on specific nodes
		self.nodes  = not procs
	
		# Check that the resources requested are available on the specified system/queue
		if self.system:
			self.checkSystemResources()

	def __str__(self):
		''' String conversion function

		Converts resource requests into the PBS script comment format
		'''
		wtime = _wtimeToString(self._resources['walltime'])
		
		if self.nodes:
			nodes = self._resources['nodes']
			ppn   = self._resources['ppn']
		else:
			procs = self._resources['procs']
		pmem = self._resources['pmem']

		time='#PBS -l walltime={wtime}'.format(wtime=wtime)
		if self.nodes:
			cpu='#PBS -l nodes={nodes}:ppn={ppn}'.format(nodes=nodes,ppn=ppn)
		else:
			cpu='#PBS -l procs={procs}'.format(procs=procs)
		mem ='#PBS -l pmem={pmem}mb'.format(pmem=pmem)

		return '\n'.join([time,cpu,mem])
	
	def comArgs(self):
		''' Returns command line arguments for the pbsUtils module '''
		args = ['-t',_wtimeToString(self._resources['walltime']),
		        '-m',str(self._resources['pmem'])]
		if self.nodes:
			args += ['-n',str(self._resources['nodes']),
			         '--ppn',str(self.resources['ppn'])]
		else:
			args += ['-p',str(self._resources['procs'])]

		if self.queue:  args += ['-q',self.queue]

		return args

	@staticmethod
	def lookupResourceLims(system,queue,noerror=False):
		''' Looks up the resource limits for a queue/system combination
		
		Input
		-----

		system  : Name of the system
		queue   : Name of the queue (default queue specified by None)
		noerror : If this is true then no errors will be raised if the limits can't be found

		Output
		------

		Resource limits in a dictionary if they can be found
		None if lookup fails
		'''
		try:
			knownQueues = knownSystems[system]
		except:
			if not noerror:
				raise ValueError("Couldn't find details of system \'{system}\'".format(system))
			return None

		try:
			queueLims = knownQueues[queue]
		except:
			if not noerror:
				raise ValueError("Couldn't find details of queue \'{queue}\' on system \'{system}\'".format(queue=queue,system=system))
			return None

		return queueLims

	def checkSystemResources(self,noerror=False):
		''' Checks whether the specified resources are available on the specified system/queue
		
		Input
		-----

		noerror: If True, the function will not throw errors on finding invalid resource requests

		Output
		------

		Returns True if the resources requested are available
		Otherwise returns False
		'''
		maxResources = resourceRequest.lookupResourceLims(self.system,self.queue)

		for res,val in self._resources.iteritems():
			if val > maxResources[res]:
				if noerror:
					return False
				else:
					if self.queue:
						raise ValueError("For the {queue} queue on {system}, {res} must be less than or equal to {maxRes}".format(queue=self.queue,system=self.system,res=res,maxRes=maxResources[res]))
					else:
						raise ValueError("On {system}, {res} must be less than or equal to {maxRes}".format(system=self.system,res=res,maxRes=maxResources[res]))

		return True

def _wtimeToString(time):
	# Convert walltime into string format
	m,s = divmod(time,60)
	h,m = divmod(m,60)
	d,h = divmod(h,24)

	wtime = '{:02d}:{:02d}:{:02d}:{:02d}'.format(d,h,m,s)

	return wtime
