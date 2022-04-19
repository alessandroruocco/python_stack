#!/usr/bin/env python
# coding=UTF-8

from __future__ import print_function
import os as _os
import time as _time
import datetime as _dt
import subprocess as _sp
import threading as _threading
import sys as _sys

from .resourceRequest import resourceRequest as _resourceRequest

def _wallTime(wtString):
	if wtString.count(':') == 2:
		if('0d' in wtString):
			dat = _dt.datetime.strptime(wtString,"0d%H:%M:%S")
			wt = _dt.timedelta(hours=dat.hour,minutes=dat.minute,seconds=dat.second)
		elif('d' in wtString):
			dat = _dt.datetime.strptime(wtString,"%dd%H:%M:%S")
			wt = _dt.timedelta(days=dat.day,hours=dat.hour,minutes=dat.minute,seconds=dat.second)
		else:
			dat = _dt.datetime.strptime(wtString,"%H:%M:%S")
			wt = _dt.timedelta(hours=dat.hour,minutes=dat.minute,seconds=dat.second)
	elif wtString.count(':') == 3:
		if wtString.startswith('00'):
			dat = _dt.datetime.strptime(wtString,"00:%H:%M:%S")
			wt = _dt.timedelta(hours=dat.hour,minutes=dat.minute,seconds=dat.second)
		else:
			dat = _dt.datetime.strptime(wtString,"%d:%H:%M:%S")
			wt = _dt.timedelta(days=dat.day,hours=dat.hour,minutes=dat.minute,seconds=dat.second)
	
	return wt

# From http://stackoverflow.com/a/377028
def _which(program):
	def is_exe(fpath):
		return _os.path.isfile(fpath) and _os.access(fpath, _os.X_OK)

	fpath, fname = _os.path.split(program)
	if fpath:
		if is_exe(program):
			return program
	else:
		for path in _os.environ["PATH"].split(_os.pathsep):
			path = path.strip('"')
			exe_file = _os.path.join(path, program)
			if is_exe(exe_file):
				return exe_file
	
	return None

def _timeStampStr():
	return _dt.datetime.now().replace(microsecond=0).isoformat()

def _logMessage(printStr,fileName):
	message = _timeStampStr()+': '+printStr
	with open(fileName,'a') as f:
		f.write(message + '\n')
		f.flush()

def _logJobOutput(jobID,outFile,errFile=None,pollInterval=0.5):
	''' Watches output files of a running job and logs them to a separate file
	
	Description
	-----------
	
	Monitors stdout and stderr files. If the files are modified it will log
	the additions. This is done by polling the files for changes - crude but
	simple and works.
	'''
	# If output and error files are the same, don't print twice!
	if errFile == outFile:
		errFile = None
	
	# Location in output files of last read
	outLoc = 0
	errLoc = 0
	# Time at which last write occurred
	prevOutStamp = 0
	prevErrStamp = 0
	while not (isComplete(jobID) or isVacated(jobID)):
		if outFile and _os.path.exists(outFile):
			# Record time file was last modified
			outStamp = _os.stat(outFile).st_mtime
	
			if outStamp != prevOutStamp:
				prevOutStamp = outStamp
				with open(outFile,'r') as out:
					with open(outFile+'.tmp','a') as outLog:
						out.seek(outLoc)
						outLog.write(out.read())
						outLog.flush()
						outLoc = out.tell()

		if errFile and _os.path.exists(errFile):
			errStamp = _os.stat(errFile).st_mtime	
			
			if errStamp != prevErrStamp:
				prevErrStamp = errStamp
				with open(errFile,'r') as err:
					with open(errFile+'.tmp','a') as errLog:
						err.seek(errLoc)
						errLog.write(err.read())
						errLog.flush()
						errLoc = err.tell()
	
		_time.sleep(pollInterval)


def _printJobOutput(jobID,outFile,errFile=None,pollInterval=0.5):
	''' Watches output files of a running job and prints them
	
	Description
	-----------
	
	Monitors stdout and stderr files. If the files are modified it will print
	the additions. This is done by polling the files for changes - crude but
	simple and works.
	'''

	# If output and error files are the same, don't print twice!
	if errFile == outFile:
		errFile = None
	
	# Location in output files of last read
	outLoc = 0
	errLoc = 0
	# Time at which last write occurred
	prevOutStamp = 0
	prevErrStamp = 0
	while not (isComplete(jobID) or isVacated(jobID)):
		if outFile and _os.path.exists(outFile):
			# Record time file was last modified
			outStamp = _os.stat(outFile).st_mtime
	
			if outStamp != prevOutStamp:
				prevOutStamp = outStamp
				with open(outFile,'r') as out:
					out.seek(outLoc)
					for line in out.read().splitlines():
						print(line)
						_sys.stdout.flush()
					outLoc = out.tell()

		if errFile and _os.path.exists(errFile):
			errStamp = _os.stat(errFile).st_mtime	
			
			if errStamp != prevErrStamp:
				prevErrStamp = errStamp
				with open(errFile,'r') as err:
					err.seek(errLoc)
					for line in err.read().splitlines():
						print(line,file=_sys.stderr)
						_sys.stderr.flush()
					errLoc = err.tell()
	
		_time.sleep(pollInterval)

def onHPC():
	''' Hacky method for checking whether we\'re on an HPC '''
	hpcCommands = ['msub','checkjob','canceljob']
	if(all(map(_which,hpcCommands))):
		return True
	else:
		return False

def genPBSScript(task,resources,openMP=False,
                 outputFile=None,forceWrite=False,
				 stdoutLog=None,stderrLog=None):
	''' Generates a PBS script for submission to an HPC '''
	if not isinstance(resources,_resourceRequest):
		raise TypeError("resources parameter should be an object of type resourceRequest")
	resourceRequestString = str(resources)

	if(openMP):
		task = 'srun -n 1 -c '+str(ppn)+' '+task
	
	if stdoutLog: outputFileString = '#PBS -o '+stdoutLog+'\n'
	else:         outputFileString = ''

	if stderrLog: errorFileString = '#PBS -e '+stderrLog+'\n'
	else:         errorFileString = ''

	pbsScript = \
	"#!/bin/bash\n\n"+\
	"# Resource requests\n"+\
	resourceRequestString+'\n\n'+\
	outputFileString+\
	errorFileString+\
	"# Make sure local (more up to date) executables are used"+'\n'+\
	"export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:$HOME/.local/lib\""+'\n'+\
	"export LIBRARY_PATH=\"$LIBRARY_PATH:$HOME/.local/lib\""+'\n\n'+\
	"# Load intel libraries"+'\n'+\
	"module load intel"+'\n'+\
	"module load impi"+'\n'+\
	"module load imkl"+'\n\n'+\
	"# Run main task"+'\n'+\
	task+'\n'

	if(outputFile != None):
		if(_os.path.exists(outputFile)):
			if(forceWrite):
				_os.remove(outputFile)
			else:
				raise IOError("PBS file "+outputFile+" already exists. Set forceWrite=True if you don't care.")

		with open(outputFile,'w') as f:
			f.write(pbsScript)
	
	return pbsScript

def submit(pbsFile,queue=None,echoID=False):
	''' Submit a task to the HPC queue
	
	Submits the task and then returns the job ID 
	'''
	try:
		if(queue != None):
			command=['msub','-q',queue,pbsFile]
		else:
			command=['msub',pbsFile]
		
		output = _sp.check_output(command)
	except _sp.CalledProcessError as cpe:
		print("Error: command \""+' '.join(command)+"\" failed with code "+str(cpe.returncode))
		exit()
	except EnvironmentError:
		print("Error: msub not found on your system, is this really tinis?")
		exit()

	jobID = int(output.decode('utf-8').replace('\n',''))
	if(echoID): print("Job successfully submitted with ID "+str(jobID))
	
	return jobID

def run(task,resources,pbsFile=None,openMP=False,
        stdoutLog=None,stderrLog=None,printQueueStatus=True):
	''' Submit a task to the HPC queue and watch its output

	This is designed to function like running a command locally, output will be
	copied to stdout.

	Input
	-----

	task      : The command to be run
	resources : A resourceRequest object
	pbsFile   : The PBS file name to write (optional)
	openMP    : Boolean, submit as an openMP job
	stdoutLog : The file to log stdout to. If not specified the output
	            will be lost (stdout file will be deleted)
	stderrLog : The file to log stderr to. If not specified the error output
	            will be lost (stderr file will be deleted)
	'''
	if not pbsFile:
		pbsFile = './job-{}.pbs'.format(int(_time.time()))
		# This may fail if run() is executed more than once simultaneously
		i = 0
		while _os.path.exists(pbsFile):
			pbsFile = './job-{}-{}.pbs'.format(int(_time.time()),i)
			i += 1
	
	if stdoutLog and _os.path.exists(stdoutLog):
		raise EnvironmentError("stdout logfile already exists")
	
	if stderrLog and _os.path.exists(stderrLog):
		raise EnvironmentError("stderr logfile already exists")

	# Attempt to generate the PBS script for launching the task
	genPBSScript(task,resources,openMP,
	             outputFile=pbsFile,forceWrite=False,
				 stdoutLog=stdoutLog,stderrLog=stderrLog)

	# Now run the task
	jobID = submit(pbsFile,resources.queue)
	
	# If no output files given then use default
	if not stdoutLog:
		outFile = './slurm-{job}.out'.format(job=jobID)
	else:
		outFile = stdoutLog
	if not stderrLog:
		errFile = './slurm-{job}.err'.format(job=jobID)
	else:
		errFile = stderrLog
	outFileTmp = outFile+'.tmp'
	errFileTmp = errFile+'.tmp'
	
	try:
		pollTime = 0.5 # Poll interval
		
		# Now that the task is running, monitor its output and error files
		jobLog = _threading.Thread(target=_logJobOutput,  args=(jobID,outFile,errFile,pollTime))
		jobMon = _threading.Thread(target=_printJobOutput,args=(jobID,outFileTmp,errFileTmp,pollTime))
		jobLog.start()
		jobMon.start()

		if printQueueStatus:
			_logMessage('Successfully submitted job (ID: {ID})'.format(ID=jobID),outFileTmp)
		
		# Watch the queue and wait until it begins to run
		while not isRunning(jobID): _time.sleep(pollTime)
		
		if printQueueStatus: 
			_logMessage('Job now running',outFileTmp)

		# Now wait for job to complete
		while not (isComplete(jobID) or isVacated(jobID)):
			_time.sleep(pollTime)

		_os.remove(outFile)
		if _os.path.exists(errFile): _os.remove(errFile)
		
		_os.rename(outFile+'.tmp',outFile)
		if _os.path.exists(errFile+'.tmp'): _os.rename(errFile+'.tmp',errFile)

		# If we don't want to keep the output files then remove them
		if not stdoutLog: _os.remove(outFile)
		if not stderrLog and _os.path.exists(errFile): _os.remove(errFile)

		jobLog.join()
		jobMon.join()

	# If we receive Ctrl-C then kill the job
	except KeyboardInterrupt:
		killJob(jobID)

		_os.remove(outFile)
		if _os.path.exists(errFile): _os.remove(errFile)
		
		_os.rename(outFile+'.tmp',outFile)
		if _os.path.exists(errFile+'.tmp'): _os.rename(errFile+'.tmp',errFile)

		# If we don't want to keep the output files then remove them
		if not stdoutLog: _os.remove(outFile)
		if not stderrLog and _os.path.exists(errFile): _os.remove(errFile)
		
		jobLog.join()
		jobMon.join()



def getStatus(jobID):
	try:
		proc = _sp.Popen(['checkjob',str(jobID)],stdout=_sp.PIPE,stderr=_sp.PIPE,bufsize=1)
		stdout,stderr = proc.communicate()
	except EnvironmentError:
		raise EnvironmentError("Error: checkjob not found on your system, is this really tinis?")

	outLines = [ line.strip().split() for line in stdout.decode('utf-8').splitlines() if line.strip() != '' ]
	#print outLines
	errLines = [ line.strip() for line in stderr.decode('utf-8').splitlines() if line.strip() != '' ]
	#print errLines

	if(any([ line[:30] == "ERROR:  invalid job specified:" for line in errLines ])):
		raise ValueError("Error: invalid job specified")

	for line in outLines:
		if(line[0] == 'State:'):
			return line[1]
	
	raise RuntimeError("Error: Didn't find state in checkjob output")

def jobExists(jobID):
	try:
		getStatus(jobID)
	except ValueError as message:
		if(message.args == "Error: invalid job specified"):
			return False
	
	return True

def isQueued(jobID):
	status = getStatus(jobID)
	if(status == 'Idle'): # TODO: Check that this is the correct status
		return True
	else:
		return False

def isRunning(jobID):
	status = getStatus(jobID)
	if(status == 'Running'): # TODO: Starting?
		return True
	else:
		return False

def isComplete(jobID):
	status = getStatus(jobID)
	if(status == 'Completed'):
		return True
	else:
		return False

def isCanceling(jobID):
	status = getStatus(jobID)
	if(status == 'Canceling'):
		return True
	else:
		return False

def isVacated(jobID): # Cancelled?
	status = getStatus(jobID)
	if(status == 'Vacated'):
		return True
	else:
		return False

def getStartTime(jobID):
	try:
		proc = _sp.Popen(['checkjob',str(jobID)],stdout=_sp.PIPE,stderr=_sp.PIPE,bufsize=1)
		stdout,stderr = proc.communicate()
	except EnvironmentError:
		raise EnvironmentError("Error: checkjob not found on your system, is this really tinis?")
	
	outLines = [ line.strip() for line in stdout.decode('utf-8').splitlines() if line.strip() != '' ]
	errLines = [ line.strip() for line in stderr.decode('utf-8').splitlines() if line.strip() != '' ]

	if(any([ line[:30] == "ERROR:  invalid job specified:" for line in errLines ])):
		raise ValueError("Error: invalid job specified")
	
	# Get start time string from command output
	# NOTE: This is in local time, whereas time.time() is UTC
	startTime = ''
	for line in outLines:
		if(line.startswith('StartTime:')):
			startTime = line.replace('StartTime:','').strip()
	
	if(startTime):
		# Attempt parsing the string we received
		startTime = _dt.datetime.strptime(startTime,'%a %b %d %H:%M:%S')
		# String doesn't include the year, so put that in just in case
		startTime = startTime.replace(year = _dt.datetime.now().year)
		# Convert to seconds since the epoch and return
		return (startTime-(startTime.now()-startTime.utcnow()) - _dt.datetime(1970,1,1,0,0,0)).total_seconds()
	else:
		raise RuntimeError("Error: Didn't find start time in checkjob output")

def killJob(jobID,graceTime=60):
	''' Kills a job submitted to the queue '''
	try:
		# Variable keeping track of elapsed time
		elapsed = 0
		
		# Check whether job has already been cancelled
		if(isVacated(jobID)):
			raise ValueError("Error: job already killed")
		
		# Run canceljob and record stdout & stderr 
		try:
			proc = _sp.Popen(['canceljob',str(jobID)],stdout=_sp.PIPE,stderr=_sp.PIPE,bufsize=1)
			stdout,stderr = proc.communicate()
		except EnvironmentError:
			raise EnvironmentError("Error: canceljob not found on your system, is this really tinis?")
		
		# Clean up stdout & stderr
		outLines = [ line.strip() for line in stdout.decode('utf-8').splitlines() if line.strip() != '' ]
		#print(outLines)
		errLines = [ line.strip() for line in stderr.decode('utf-8').splitlines() if line.strip() != '' ]
		#print(errLines)
	
		# Job may not exist, in which case raise error
		if(any([ line.startswith("ERROR:  invalid job specified") for line in errLines ])):
			raise ValueError("Error: invalid job specified")
		
		# Look through stdout & stderr to check if we have successfully sent the command
		cancelled = False
		for line in outLines:
			if(line.startswith('job') and line.endswith('cancelled')):
				cancelled = True
		for line in errLines:
			if '(Job/step already completing or completed)' in line:
				cancelled = True
			elif '(previous failure - try again in' in line:
				# If this happens assume we've previously tried to cancel the
				# job and wait to see if it does get cancelled.
				cancelled = True
		
		if not cancelled:
			raise EnvironmentError("Error: canceljob doesn't seem to have worked...")
		
		# Canceling doesn't seem to happen instantaneously, so wait for graceTime
		# while it is canceling
		startTime = _time.time()
		while(not (isVacated(jobID) or isComplete(jobID)) and elapsed < graceTime):
			_time.sleep(0.5)
			elapsed = _time.time()-startTime
		
		# If we haven't successfully killed it by now then raise an error
		if(not (isVacated(jobID) or isComplete(jobID)) and elapsed > graceTime):
			raise EnvironmentError("Error: job not cancelled despite waiting {:d} seconds since canceljob command issued".format(graceTime))
	# If we receive Ctrl-C then continue trying to kill the job
	except KeyboardInterrupt:
		# Not a great way to deal with this
		# TODO: Improve
		killJob(jobID,graceTime-elapsed)


def getOutput(jobID):
	if not (isRunning(jobID) or isComplete(jobID) or isVacated(jobID)):
	
		raise ValueError("Error: Job no. {:d} has not started running".format(jobID))
	else:
		stdout = None
		stdoutLoc = './slurm-{:d}.out'.format(jobID)
		stderr = None
		stderrLoc = './slurm-{:d}.err'.format(jobID)
		
		# Check for stdout logfile, throw error if it isn't found
		if(_os.path.exists(stdoutLoc)):
			stdout = open(stdoutLoc,'r')
		else: raise EnvironmentError("Error: can\'t find slurm output file")
		
		# Check for stderr logfile, breathe a sigh of relief if it isn't found
		if(_os.path.exists(stderrLoc)):
			stderr = open(stderrLoc,'r')
		#else:
		#	breathe.sigh('relief')
		
		return stdout,stderr

def getError(jobID):
	if not (isRunning(jobID) or isComplete(jobID) or isVacated(jobID)):
		raise ValueError("Error: Job no. {:d} has not started running".format(jobID))
	else:
		stderr = None
		stderrLoc = './slurm-{:d}.err'.format(jobID)
		
		# Check for stderr logfile, throw error if it isn't found
		if(_os.path.exists(stderrLoc)):
			stderr = open(stderrLoc,'r')
		else: raise EnvironmentError("Error: can\'t find slurm error file")
		
		return stderr
