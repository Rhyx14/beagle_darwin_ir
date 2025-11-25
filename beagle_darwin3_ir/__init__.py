from .label import Label
from .RREG import RREG
from .pseudo_inst import jc,jmp,jnc,str_expr
from .if_else import If
from .preprocess import preprocess
from .LSLS_params import LSLS_Param
from .macro import Let
__all__=[
    'Label',
    'RREG',
    'jc','jmp','jnc','str_expr',
    'If',
    'preprocess',
    'LSLS_Param',
    'Let'
]

from .misc import bind32
__all__.extend([
    'bind32',
])

# monkey patch
from .monkey_patch.dump_neurons import dump_input_neuron,dump_pop,dump_output_neuron
__all__.extend([
    'dump_input_neuron',
    'dump_pop',
    'dump_output_neuron',
])

from .monkey_patch.linear import connect_linear
from .monkey_patch.one2one import connect_one2one
__all__.extend([
    'connect_linear',
    'connect_one2one'
])