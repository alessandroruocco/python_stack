from . import shared

blockName = 'dist_fn'
blockDef = {
	'mandatory'    :False,
	'allowMultiple':True,
	'parameters'   :[
		{ 'name'        :'name',
		  'dType'       :str,
		  'mandatory'   :True },
		{ 'name'        :'ndims',
		  'dType'       :int,
		  'mandatory'   :True },
		{ 'name'        :'dumpmask',
		  'dType'       :str,
		  'mandatory'   :False,
		  'allowedVals' :shared.dumpMasks },
		{ 'name'        :'direction1',
		  'dType'       :str,
		  'mandatory'   :True },
		{ 'name'        :'direction2',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'direction3',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'range1',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'range2',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'range3',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'restrict_x',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'restrict_y',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'restrict_z',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'restrict_px',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'restrict_py',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'restrict_pz',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'resolution1',
		  'dType'       :int,
		  'mandatory'   :False },
		{ 'name'        :'resolution2',
		  'dType'       :int,
		  'mandatory'   :False },
		{ 'name'        :'resolution3',
		  'dType'       :int,
		  'mandatory'   :False },
		{ 'name'        :'include_species',
		  'dType'       :str,
		  'mandatory'   :False }
	]
}
