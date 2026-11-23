import os,json
from typing import Optional,Union,Sequence
from darwin3_deployment.codegen import skip_if_flagged,DelayRecord,PhysicalPopulation

@skip_if_flagged
def dump_output_neuron(output_pops: Union[PhysicalPopulation, Sequence[PhysicalPopulation]], pop_name: str,
                       output_dir: str | None = None, file_name: str = "output_neuron"):
    
    if output_dir is None: return # fake compile模式下不用dump output neuron

    if isinstance(output_pops, PhysicalPopulation):
        output_pops = [output_pops]

    output_neuron = {}  # 划分后coord, 局部id -> 每层的逻辑id
    start = 0
    for output_pop in output_pops:
        # output_pop.remove_empty_axon()
        for axon_id in range(output_pop.size):
            output_neuron[f"{output_pop.coord[0]}, {output_pop.coord[1]}, {axon_id}"] = start + axon_id
        start += output_pop.size
    file_name = file_name + "_" + pop_name + ".json"
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(output_neuron, f)