from . import shared

blockName = 'boundaries'
blockDef = { 
	'mandatory'    :True,
	'allowMultiple':False,
	'parameters'   :[
		{ 'name'        :'bc_x_min',
		  'dType'       :str,
		  'mandatory'   :False,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_x_max',
		  'dType'       :str,
		  'mandatory'   :False,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_y_min',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :2,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_y_max',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :2,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_z_min',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :3,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_z_max',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :3,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_x_min_field',
		  'dType'       :str,
		  'mandatory'   :False,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_x_max_field',
		  'dType'       :str,
		  'mandatory'   :False,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_y_min_field',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :2,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_y_max_field',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :2,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_z_min_field',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :3,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_z_max_field',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :3,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_x_min_particle',
		  'dType'       :str,
		  'mandatory'   :False,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_x_max_particle',
		  'dType'       :str,
		  'mandatory'   :False,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_y_min_particle',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :2,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_y_max_particle',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :2,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_z_min_particle',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :3,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'bc_z_max_particle',
		  'dType'       :str,
		  'mandatory'   :False,
		  'minD'        :3,
		  'allowedVals' :shared.boundaries },
		{ 'name'        :'cpml_thickness',
		  'dType'       :int,
		  'mandatory'   :False,
		  'verification':lambda x: x>0 },
		{ 'name'        :'cpml_kappa_max',
		  'dType'       :float,
		  'mandatory'   :False },
		{ 'name'        :'cpml_a_max',
		  'dType'       :float,
		  'mandatory'   :False },
		{ 'name'        :'cpml_sigma_max',
		  'dType'       :float,
		  'mandatory'   :False }
	]
}
