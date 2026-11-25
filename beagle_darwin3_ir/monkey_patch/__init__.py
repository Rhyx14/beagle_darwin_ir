from darwin3_deployment.core_config import CoreConfig
from loguru import logger
# 添加16位支持
CoreConfig._INFERENCE_PARAMETER_MAP = {f"PAR{i}": (8, i * 8) for i in range(6)} | {"PAR01":(16,0),"PAR23":(16,16),"PAR45":(16,32)}
logger.info('Patch inference parameter map.')

from .connection.fast_conv import add_conv_connection 
from .connection.linear import connect_linear
from .connection.uneven_one2one import connect_one2one
from .connection.shared_one2one import connect_shared_one2one
from .connection.conv_variants import add_group_conv, add_transposed_conv, get_nc_channel
__all__=[
    'add_conv_connection',
    'connect_linear',
    'connect_one2one',
    'connect_shared_one2one',
    'add_group_conv',
    'add_transposed_conv',
    'get_nc_channel'
]

from .dump.axon_serialize import axon_serialize
from .dump.dump_input import dump_input_neuron
from .dump.dump_neurons import dump_pop
from .dump.dump_output import dump_output_neuron