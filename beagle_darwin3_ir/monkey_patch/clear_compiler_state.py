import darwin3_deployment.codegen as _codegen
from darwin3_deployment.ir.net_population import PhysicalPopulation


def clear_compiler_state() -> None:
    """
    Clear any compiler state.
    This function is intended to be called before recompiling.
    """
    # Clear used coordinates set
    PhysicalPopulation._used_coords.clear()
    # codegen 的超距标记是进程级粘滞开关：一旦置位，check_distance 会静默跳过所有连接、
    # skip_if_flagged 会静默跳过所有 dump，且 darwin3_deployment 内部不复位。
    # 不清掉的话，第二次 compile 会产出空连接，并在 dump_pop 解包 None 时丢出误导性的 TypeError。
    _codegen._distance_exceeded_flag=False
    _codegen._skipped_functions.clear()
