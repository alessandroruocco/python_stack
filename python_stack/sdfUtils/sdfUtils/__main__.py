def sizeof_fmt(num, suffix='B',separate=False):
	'''
	Converts a number of bytes into a readable expression

	From https://stackoverflow.com/a/1094933/2622765

	Parameters
	----------

	num : Number of bytes/bits
	suffix : Unit of measure, default 'B' for bytes. E.g. 'b' for bits
	separate : Return a tuple containing the number followed by the unit rather
	           than a single string
	'''
	foundUnit = False
	for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
		if abs(num) < 1024.0:
			sizeTup = ('{:3.1f}'.format(num),'{:}{:}'.format(unit, suffix))
			foundUnit = True
			break
		num /= 1024.0
	if not foundUnit: sizeTup = ('{:.1f}'.format(num),'{:}{:}'.format('Yi',suffix))

	if separate:
		return sizeTup
	else:
		return sizeTup[0]+sizeTup[1]


def main():
	import argparse
	import os
	import sdf

	parser = argparse.ArgumentParser()
	parser.add_argument('sdfFile')
	args = parser.parse_args()

	sdfFile = sdf.read(args.sdfFile)
	sdfDict = sdfFile.__dict__

	header = sdfFile.Header
	print("SDF file metadata:")
	for k in sorted(header.keys()): print(' - {:}: {:}'.format(k,header[k]))

	print("\nSDF file contents:")
	contentKeys = sorted(sdfFile.__dict__.keys())
	contentKeys.remove('Header')
	#for k in contentKeys:
	#	print(' - {:}'.format(k))

	infoList = {}
	print('       Variable Name        |   Dimensions   |   Data   |   Value   |   Units   ')
	print('================================================================================')
	for k in contentKeys:
		val = sdfDict[k]
		dType = type(val)

		dimStr  = ''
		units   = 'N/A'
		value   = ''
		size    = sizeof_fmt(val.data_length,separate=True)
		if dType == sdf.BlockPlainVariable:
			dimStr = 'x'.join([str(i) for i in val.dims])
			units  = val.units
		elif dType == sdf.BlockPointVariable:
			dimStr = 'x'.join([str(i) for i in val.dims])
			units  = val.units
		elif dType == sdf.BlockArray:
			dimStr = 'x'.join([str(i) for i in val.dims])
		elif dType == sdf.BlockConstant:
			if val.data == 0.0:
				value = '0'
			elif type(val.data) == int and abs(val.data) < 1000000000:
				value = '{:>d}'.format(val.data)
			elif abs(val.data) > 1000.0 or abs(val.data) < 1e-3:
				value = '{:.3e}'.format(val.data)
			else:
				value = '{:5.4f}'.format(val.data)
		elif dType == sdf.BlockNameValue:
			continue
		elif dType == sdf.BlockPlainMesh:
			dimStr = ','.join([str(i) for i in val.dims])
			units  = val.units
		elif dType == sdf.BlockPointMesh:
			dimStr = ','.join([str(i) for i in val.dims])
			units  = val.units
		else:
			print(dType)
			raise RuntimeError("Unrecognised SDF variable type")

		infoList[k] = {'dims':dimStr,'units':units,'value':value,'size':size,'dType':str(dType)}
		#print(infoList[k])

	lengths = { 'dims':max([ len(infoList[k]['dims'])  for k in infoList ]),
	           'units':max([ len(infoList[k]['units']) for k in infoList ]),
			   'value':max([ len(infoList[k]['value']) for k in infoList ]),
			    'size':max([ len(infoList[k]['size'][0])+len(infoList[k]['size'][1]) for k in infoList ]),
			   'dType':max([ len(infoList[k]['dType']) for k in infoList ])}

	for k in sorted(infoList.keys()):
		v = infoList[k]
		print(' {:27.27}|{:>15} | {:>5}{:<3} |{:>10} | {:}'.format(k,v['dims'],v['size'][0],v['size'][1],v['value'],v['units']))

	print('\nSDF File size:')
	print(' - Size of data: {:}'.format(sizeof_fmt(sum([ sdfDict[k].data_length for k in contentKeys ]))))
	print(' - Size on disk: {:}'.format(sizeof_fmt(os.path.getsize(args.sdfFile))))

if __name__ == "__main__":
	main()
