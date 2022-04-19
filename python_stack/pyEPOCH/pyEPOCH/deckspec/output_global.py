# TODO: Define this
blockName = 'output_global'
blockDef = {
	'mandatory'    :False,
	'allowMultiple':False,
	'parameters'   :[
		{ 'name'        :'force_first_to_be_restartable',
		  'dType'       :bool,
		  'mandatory'   :False },
		{ 'name'        :'force_last_to_be_restartable',
		  'dType'       :bool,
		  'mandatory'   :False },
		{ 'name'        :'dump_first',
		  'dType'       :bool,
		  'mandatory'   :False },
		{ 'name'        :'dump_last',
		  'dType'       :bool,
		  'mandatory'   :False },
		{ 'name'        :'time_start',
		  'dType'       :float,
		  'mandatory'   :False },
		{ 'name'        :'time_stop',
		  'dType'       :float,
		  'mandatory'   :False },
		{ 'name'        :'nstep_start',
		  'dType'       :int,
		  'mandatory'   :False },
		{ 'name'        :'nstep_stop',
		  'dType'       :int,
		  'mandatory'   :False },
		{ 'name'        :'sdf_buffer_size',
		  'dType'       :int,
		  'mandatory'   :False },
		{ 'name'        :'filesystem',
		  'dType'       :str,
		  'mandatory'   :False },
		{ 'name'        :'use_offset_grid',
		  'dType'       :bool,
		  'mandatory'   :False },
		{ 'name'        :'dump_first_after_restart',
		  'dType'       :bool,
		  'mandatory'   :False }
	]
}