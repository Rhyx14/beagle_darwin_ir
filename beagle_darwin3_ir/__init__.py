from .pseduo_inst.LabledInstruction import Label
from .pseduo_inst.RREG import RREG
from .pseduo_inst import jc,jmp,jnc,str_expr
from .pseduo_inst import If
from .preprocess import preprocess
from .LSLS_params import LSLS_Param
from .pseduo_inst.macro import Let
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
from .monkey_patch.dump.dump_neurons import dump_input_neuron,dump_pop,dump_output_neuron
from .monkey_patch.core_usage_info import CoreUsageInfo
from .monkey_patch.clear_compiler_state import clear_compiler_state
__all__.extend([
    'dump_input_neuron',
    'dump_pop',
    'dump_output_neuron',
    'CoreUsageInfo',
    'clear_compiler_state'
])

from .monkey_patch import add_conv_connection,connect_one2one,connect_linear
__all__.extend([
    'add_conv_connection',
    'connect_linear',
    'connect_one2one'
])
