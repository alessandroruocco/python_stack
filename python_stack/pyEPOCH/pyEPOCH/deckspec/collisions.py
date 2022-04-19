blockName = 'collisions'
blockDef = {
	'mandatory'    :False,
	'allowMultiple':False,
	'parameters'   :[
		{ 'name'        :'use_collisions',
		  'dType'       :bool,
		  'mandatory'   :True },
		{ 'name'        :'coulomb_log',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'collide',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'collisional_ionisation',
		  'dType'       :bool,
		  'mandatory'   :False }
	]
}
