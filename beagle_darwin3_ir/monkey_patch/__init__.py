from darwin3_deployment.core_config import CoreConfig
from loguru import logger
# 添加16位支持
CoreConfig._INFERENCE_PARAMETER_MAP = {f"PAR{i}": (8, i * 8) for i in range(6)} | {"PAR01":(16,0),"PAR23":(16,16),"PAR45":(16,32)}
logger.info('Patch inference parameter map.')

from .connection.fast_conv import add_conv_connection 
from .connection.linear import connect_linear
from .connection.one2one import connect_one2one
__all__=[
    'add_conv_connection',
    'connect_linear',
    'connect_one2one'
]

from .dump.axon_serialize import axon_serialize
