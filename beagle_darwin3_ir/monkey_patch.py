from darwin3_deployment.core_config import CoreConfig
from loguru import logger
# 添加16位支持
CoreConfig._INFERENCE_PARAMETER_MAP = {f"PAR{i}": (8, i * 8) for i in range(6)} | {"PAR01":(16,0),"PAR23":(16,16),"PAR45":(16,32)}
logger.info('Patching compiler done')
 
