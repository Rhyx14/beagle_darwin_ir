import os,json
import numpy as np
from pathlib import Path
from typing import Optional,Union,Sequence
from darwin3_deployment.core_config import CoreConfig
from loguru import logger
# 添加16位支持
CoreConfig._INFERENCE_PARAMETER_MAP = {f"PAR{i}": (8, i * 8) for i in range(6)} | {"PAR01":(16,0),"PAR23":(16,16),"PAR45":(16,32)}
logger.info('Patching compiler done')
 
from darwin3_deployment.codegen import skip_if_flagged,DelayRecord,PhysicalPopulation,PopsDataConfig
@skip_if_flagged
def dump_pop(pop_list: Union[Sequence[PhysicalPopulation], PhysicalPopulation],
             pop_name: str, output_dir: Optional[str | Path] = None, need_description=True,
             need_show_usage=True) -> list[tuple]:
    """序列化PhysicalPopulation

    Args:
        pop_list (Sequence[PhysicalPopulation] | PhysicalPopulation): 由同一神经元群切分后得到的若干神经元群
        pop_name: pop名
        output_dir (str | None): 输出目录
        need_description: 是否需要输出描述信息
        need_show_usage: 是否需要输出节点使用率信息
    
    Return: 编译错误列表
    """
    delay_record = DelayRecord()
    pop_info = PopsDataConfig().get_pop_data(pop_name)
    original_dir = os.getcwd()
    if output_dir is not None:
        os.chdir(output_dir)

    pop_list = [pop_list] if isinstance(pop_list, PhysicalPopulation) else pop_list

    core_config = pop_info["core_config"]
    conf, regs, inst, inf_state, inf_param, vt, learning_param = core_config.get_core_config_value(pop_info)
    delay = regs[core_config.delay_reg[1]] if core_config.delay_reg else None

    total_pop_size = sum(pop.size for pop in pop_list)

    def _validate_array_size(array: Union[np.ndarray, int, None], name: str) -> Optional[np.ndarray]:
        if array is None:
            return None
        if isinstance(array, int):
            array = np.full(total_pop_size, array, dtype=np.int64)
        elif isinstance(array, np.ndarray) and array.size != total_pop_size:
            raise ValueError(f"{name} size ({array.size}) does not match "
                             f"the number of neurons ({total_pop_size}).")
        return array

    inf_state = _validate_array_size(inf_state, "Inference state")
    inf_param = _validate_array_size(inf_param, "Inference parameter")
    vt = _validate_array_size(vt, "Vt")
    learning_param = _validate_array_size(learning_param, "Learning parameter")

    inf_start = 0

    id_start = 0  # 输出json文件: id->coord+id
    pops_mapping = {}

    error_list=[]
    for pop in pop_list:
        pop.remove_empty_axon()  # 重排神经元（有轴突、无轴突）
        conf[0] = (pop.size - 1) << 4  # 神经元个数寄存器[15:4] 神经元个数-1
        conf[1] = (pop.num_dendrites - 1) << 16  # 树突个数寄存器[30:16] 树突终止编号
        conf[8] |= pop.weight_qa_serialize()
        if delay is not None:
            high_reg, low_reg = core_config.delay_reg
            scaled_delay = pop.size * delay
            regs[high_reg] = (scaled_delay >> 15) & 0x7fff
            regs[low_reg] = scaled_delay & 0x7fff
            delay_record.set_delay(pop.coord, [high_reg, regs[high_reg], low_reg, regs[low_reg]])

        x, y = pop.coord

        for i in range(pop.size):
            pops_mapping[id_start+i] = (x, y, i)
        id_start += pop.size

        # ========================= 描述文件 =========================
        if need_description:
            desc_path = os.path.join('.', 'desc')
            if not os.path.exists(desc_path):
                os.makedirs(desc_path)
            desc_file_path = os.path.join(desc_path, f"{x}-{y}-desc.md")
            with open(desc_file_path, "w") as f:
                f.write("# Axon:\n")
                axon_desc, axon_out_desc = pop.axon_description()
                f.write(axon_desc + "\n\n")
                f.write(axon_out_desc + "\n\n")
                f.write("# Dendrite:\n")
                dendrite_desc, neu_id_desc, weight_desc = pop.dendrite_description()
                f.write(dendrite_desc + "\n\n")
                f.write(neu_id_desc + "\n\n")
                f.write(weight_desc + "\n\n")

        # ========================= 树突存储区 =========================
        with open(f"{x}-{y}-de.txt", "w") as f:
            dendrite_bin = pop.dendrite_serialize()
            stack_point = pop.dendrite_memory_size
            if inf_state is None:
                stack_point -= 4096
            if inf_param is None:
                stack_point -= 4096
            data_size = len(dendrite_bin)
            link_size_minus_one = (data_size - 1) & 0xffff
            conf[29] = (stack_point << 16) | link_size_minus_one
            if inf_state is not None:
                data_size += pop.size
            elif inf_param is not None:
                data_size += 4096 + pop.size
            dendrite_memory_usage = data_size / pop.dendrite_memory_size * 100
            for bin in dendrite_bin:
                f.write(f"{bin:012x}\n")

        # ========================= 轴突存储区 =========================
        with open(f"{x}-{y}-ax.txt" , "w") as f:
            axon_bin = pop.axon_serialize()
            axon_memory_usage = len(axon_bin) / pop.axon_memory_size * 100
            for axon_id, bin in sorted(axon_bin.items()):
                f.write(f"{bin:07x}\n")

        # ========================= 状态/参数 =========================
        if inf_state is not None:
            with open(f"{x}-{y}-is.txt", "w") as f:
                num_neurons = pop.size
                for i in range(num_neurons):
                    f.write(f"{int(inf_state[inf_start + num_neurons - 1 - i]):012x}\n")
        if inf_param is not None:
            with open(f"{x}-{y}-ip.txt", "w") as f:
                num_neurons = pop.size
                for i in range(num_neurons):
                    f.write(f"{int(inf_param[inf_start + num_neurons - 1 - i]):012x}\n")
        if learning_param is not None:
            with open(f"{x}-{y}-lp.txt", "w") as f:
                num_neurons = pop.size
                for i in range(num_neurons):
                    f.write(f"{int(learning_param[i]):012x}\n")
        if vt is not None:
            with open(f"{x}-{y}-vt.txt", "w") as f:
                num_neurons = pop.size
                for i in range(num_neurons):
                    f.write(f"{int(vt[i]):04x}\n")
        inf_start += pop.size

        # ========================= 寄存器/配置 =========================
        with open(f"{x}-{y}-config.dwnc", "w") as f:
            f.write(f"# core ({x}, {y})\n")
            for i in range(len(conf)):
                f.write(f"0 write {x} {y} 0x{i:04x} 0x{conf[i]:08x}\n")
            for i in range(len(regs)):
                f.write(f"0 write {x} {y} 0x{i+0x800:04x} 0x{regs[i]:08x}\n")
            for i in range(len(inst)):
                f.write(f"0 write {x} {y} 0x{i+0x1000:04x} 0x{inst[i]:08x}\n")
            f.write(f"0 write_ram {x} {y} 0x08000 {x}-{y}-ax.txt\n")
            f.write(f"0 write_ram {x} {y} 0x10000 {x}-{y}-de.txt\n")
            if inf_state is not None:
                f.write(f"0 write_ram {x} {y} 0x1{pop.inf_state_start_addr-pop.size:04x} {x}-{y}-is.txt\n")
            if inf_param is not None:
                f.write(f"0 write_ram {x} {y} 0x1{pop.inf_param_start_addr-pop.size:04x} {x}-{y}-ip.txt\n")
            if learning_param is not None:
                f.write(f"0 write_ram {x} {y} 0x1{pop.learning_param_start_addr-pop.size:04x} {x}-{y}-lp.txt\n")
            if vt is not None:
                f.write(f"0 write_ram {x} {y} 0x02000 {x}-{y}-vt.txt\n")
        
        # ========================= 打印树突/轴突存储区使用情况 =========================
        is_error = (data_size > pop.dendrite_memory_size) or (len(axon_bin) > pop.axon_memory_size)
        error_list.append((pop.coord,data_size,pop.dendrite_memory_size,len(axon_bin),pop.axon_memory_size))
        core_usage = (f"Core{pop.coord}: "
                      f"Dendrite(size/depth) {data_size}/{pop.dendrite_memory_size}|{dendrite_memory_usage:.2f}%"
                      f" --- Axon(size/depth) {len(axon_bin)}/{pop.axon_memory_size}|{axon_memory_usage:.2f}%")
        if is_error:
            logger.error(core_usage)
        else:
            if need_show_usage or is_error:
                logger.info(core_usage)
    
    # ========================= 神经元群json文件 =========================
    file_name = pop_name + ".json"
    with open(file_name, "w") as f:
        json.dump(pops_mapping, f)

    os.chdir(original_dir)
    return error_list


@skip_if_flagged
def dump_input_neuron(input_pops: Union[PhysicalPopulation, Sequence[PhysicalPopulation]],
                      output_dir: str = ".", file_name: str = "input_neuron.json"):
    """将输入神经元信息导出为JSON文件

    Args:
        input_pops: 所有输入神经元群
        output_dir: 输出文件的目录，默认为当前目录
        file_name: 输出文件的名称，默认为 "input_neuron.json"
    """
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

@skip_if_flagged
def dump_output_neuron(output_pops: Union[PhysicalPopulation, Sequence[PhysicalPopulation]], pop_name: str,
                       output_dir: str = ".", file_name: str = "output_neuron"):
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