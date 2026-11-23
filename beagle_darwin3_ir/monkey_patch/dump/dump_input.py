import os,json
from typing import Optional,Union,Sequence
from darwin3_deployment.codegen import skip_if_flagged,DelayRecord,PhysicalPopulation

@skip_if_flagged
def dump_input_neuron(input_pops: Union[PhysicalPopulation, Sequence[PhysicalPopulation]],
                      output_dir: str | None = None, file_name: str = "input_neuron.json"):
    """将输入神经元信息导出为JSON文件

    Args:
        input_pops: 所有输入神经元群
        output_dir: 输出文件的目录，默认为当前目录
        file_name: 输出文件的名称，默认为 "input_neuron.json"
    """
    if output_dir is None: return # fake compile 模式下不用 dump input
    if isinstance(input_pops, PhysicalPopulation):
        input_pops = [input_pops]

    input_neuron = {}  # 前神经元序号: [0, 后神经元序号, [[x, y, 树突id], ...]]
    start_id = 0
    for input_pop in input_pops:
        if input_pop.pop_position != "input":
            raise ValueError(f"Population on core {input_pop.coord} is not an input population.")
        anaxonic_neurons = []
        for pre_neu_id, axon in input_pop._axons.items():
            if not axon:
                anaxonic_neurons.append(pre_neu_id)
                input_neuron[pre_neu_id + start_id] = []
                continue
            pre_neu_id += start_id
            post_neu_id = max(axon[0], 0)
            for axon_id in axon[1:]:
                post_pop, den_id = input_pop._axon_out[axon_id]
                if pre_neu_id not in input_neuron:
                    input_neuron[pre_neu_id] = [0, post_neu_id, []]
                input_neuron[pre_neu_id][2].append([post_pop.coord[0], post_pop.coord[1], den_id])
        start_id += input_pop.size
        if anaxonic_neurons:
            print(f"[INFO] Input population on virtual core {input_pop.coord} contains anaxonic neurons: {anaxonic_neurons}.")
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(input_neuron, f)
