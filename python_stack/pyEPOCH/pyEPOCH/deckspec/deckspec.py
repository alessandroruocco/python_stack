from . import control
from . import boundaries
from . import species
from . import laser
from . import fields
from . import window
from . import output
from . import output_global
from . import dist_fn
from . import collisions
from . import probe
from . import qed
from . import subset

blockDefinitions = {
	control.blockName       : control.blockDef,
	boundaries.blockName    : boundaries.blockDef,
	species.blockName       : species.blockDef,
	laser.blockName         : laser.blockDef,
	fields.blockName        : fields.blockDef,
	window.blockName        : window.blockDef,
	output.blockName        : output.blockDef,
	output_global.blockName : output_global.blockDef,
	dist_fn.blockName       : dist_fn.blockDef,
	collisions.blockName    : collisions.blockDef,
	probe.blockName         : probe.blockDef,
	qed.blockName           : qed.blockDef,
	subset.blockName        : subset.blockDef
}
