from . import shared

blockName = 'probe'
blockDef = {
	'mandatory'    :False,
	'allowMultiple':True,
	'parameters'   :[
		{ 'name'        :'name',
		  'dType'       :str,
		  'mandatory'   :True },
		{ 'name'        :'point',
		  'dType'       :str,
		  'mandatory'   :True },
		{ 'name'        :'normal',
		  'dType'       :str,
		  'mandatory'   :True },
		{ 'name'        :'include_species',
		  'dType'       :str,
		  'mandatory'   :True },
		{ 'name'        :'ek_min',
		  'dType'       :float,
		  'mandatory'   :False },
		{ 'name'        :'ek_max',
		  'dType'       :float,
		  'mandatory'   :False },
		{ 'name'        :'dumpmask',
		  'dType'       :str,
		  'mandatory'   :True,
		  'allowedVals' :shared.dumpMasks }
	]
}

