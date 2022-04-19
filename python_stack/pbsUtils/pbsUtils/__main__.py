
def main():
	import argparse
	import time
	import os
	import subprocess as sp

	import pbsUtils
	from resourceRequest import resourceRequest

	# Handle command line arguments.
	# These should direct us as to the resource requirements of the job and the
	# tasks that need to be performed.
	parser = argparse.ArgumentParser()
	parser.add_argument('task',type=str)
	parser.add_argument('-t','--walltime',type=pbsUtils._wallTime,required=True)
	parser.add_argument('-n','--nodes',type=int,default=None)
	parser.add_argument('--ppn',type=int,default=None)
	parser.add_argument('-p','--procs',type=int,default=None)
	parser.add_argument('-m','--pmem',type=int,default=3882)
	parser.add_argument('--openMP',action='store_true')
	parser.add_argument('--pbsFile')
	parser.add_argument('-o','--output')
	parser.add_argument('-e','--error')
	parser.add_argument('-f','--forceOutput',action='store_true')
	parser.add_argument('-q','--queue',type=str,default=None)
	parser.add_argument('--echo',action='store_true')
	parser.add_argument('--noSubmit',action='store_true')
	parser.add_argument('--noSave',action='store_true')
	parser.add_argument('--watch',action='store_true')
	args = parser.parse_args()
	
	args.walltime = args.walltime.total_seconds()

	resources = resourceRequest(args.walltime,
	                            nodes  = args.nodes,
	                            ppn    = args.ppn,
	                            procs  = args.procs,
	                            pmem   = args.pmem,
	                            system = 'tinis',
								queue  = args.queue)

	pbsScript = pbsUtils.genPBSScript(args.task,resources,args.openMP)
	
	if args.echo:
		print("\nGenerated script:")
		print("-------START--------")
		print(pbsScript)
		print("--------END--------\n")

	if not args.pbsFile:
		timeStamp = int(time.time())
		pbsFile = 'job-'+str(timeStamp)+'.pbs'
	else:
		pbsFile = args.pbsFile

	if os.path.exists(pbsFile):
		if args.forceOutput:
			os.remove(pbsFile)	
		else:
			raise IOError("PBS file "+pbsFile+" already exists...")

	if (not args.noSave and not args.noSubmit):
		if args.watch:
			pbsUtils.run(args.task,resources,pbsFile,args.openMP,
			             stdoutLog=args.output,stderrLog=args.error)
		else:
			if not args.noSave:
				with open(pbsFile,'w') as f:
					f.write(pbsScript)
			pbsUtils.submit(pbsFile,args.queue,True,
			                stdoutLog=args.output,stderrLog=args.error)

if __name__ == '__main__':
	main()
