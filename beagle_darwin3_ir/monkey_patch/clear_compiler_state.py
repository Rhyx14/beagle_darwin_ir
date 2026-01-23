from darwin3_deployment.ir.net_population import PhysicalPopulation
def clear_compiler_state() -> None:
    """
    Clear any compiler state.
    This function is intended to be called before recompiling.
    """
    # Clear used coordinates set
    PhysicalPopulation._used_coords.clear()