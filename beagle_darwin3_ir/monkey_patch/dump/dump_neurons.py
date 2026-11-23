import os,json
import numpy as np
from pathlib import Path
from typing import Optional,Union,Sequence
from loguru import logger
from ..core_usage_info import CoreUsageInfo
from collections import defaultdict
from io import StringIO
from darwin3_deployment.codegen import skip_if_flagged,DelayRecord,PhysicalPopulation

@skip_if_flagged
def dump_pop(pop_list: Union[Sequence[PhysicalPopulation], PhysicalPopulation],
             pop_name: str, pop_info: dict, output_dir:  Path = None, need_description=True,
             need_show_usage=True) -> list[tuple]:
    """序列化PhysicalPopulation

    Args:
        pop_list (Sequence[PhysicalPopulation] | PhysicalPopulation): 由同一神经元群切分后得到的若干神经元群
        pop_name: pop名
        output_dir (str | None): 输出目录
        need_description: 是否需要输出描述信息
        need_show_usage: 是否需要输出节点使用率信息
    
    Return: 编译错误列表, 核心占用列表
    """
    delay_record = DelayRecord()
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

    id_start = 0  # 输出json文件: id->coord+id, 子神经元索引相对于整体的偏移量
    pops_mapping = {}

    error_flag=False
    pops_usages=[]
    
    string_io_dict=defaultdict(StringIO) # {filename: filecontent}
    
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

        # ========================= 序列化树突 ============================
        dendrite_bin = pop.dendrite_serialize()
        stack_point = pop.dendrite_memory_size
        if inf_state is None:
            stack_point -= 4096
        if inf_param is None:
            stack_point -= 4096
        dendrite_data_size = len(dendrite_bin)
        link_size_minus_one = (dendrite_data_size - 1) & 0xffff
        conf[29] = (stack_point << 16) | link_size_minus_one
        if inf_state is not None:
            dendrite_data_size += pop.size
        elif inf_param is not None:
            dendrite_data_size += 4096 + pop.size


        if inf_state is not None and pop.inf_state_start_addr-pop.size < len(dendrite_bin):
            logger.error(f'Core{pop.coord}: IS area covers the dendrite memory!')
            error_flag=True

        if inf_param is not None and pop.inf_param_start_addr-pop.size < len(dendrite_bin):
            logger.error(f'Core{pop.coord}: IP area covers the dendrite memory!')
            error_flag=True

        # ========================= 序列化树突 ============================
        axon_bin = pop.axon_serialize()

        # ================= save the usage of 树突/轴突存储区 =============
        _core_usage_info=CoreUsageInfo(pop.coord,
                                        dendrite_used=dendrite_data_size,
                                        dendrite_size=pop.dendrite_memory_size,
                                        axon_used=len(axon_bin),
                                        axon_size=pop.axon_memory_size)
        if _core_usage_info.is_error():
            error_flag=True
            if need_show_usage:
                logger.error(_core_usage_info)
        if need_show_usage:
            logger.info(_core_usage_info)
        pops_usages.append(_core_usage_info)
        
        # ========================= 写入文件 ============================
        if output_dir is not None:
            # ======= 写入轴突树突存储区 =========
            f= StringIO()
            for _axon_id, _bin in sorted(axon_bin.items()):
                f.write(f"{_bin:07x}\n")
            string_io_dict[output_dir / f"{x}-{y}-ax.txt"]=f
            f= StringIO()
            for _bin in dendrite_bin:
                f.write(f"{_bin:012x}\n")
            string_io_dict[output_dir / f"{x}-{y}-de.txt"]=f

            # ======== 状态/参数 ================
            if inf_state is not None:
                f = StringIO()
                num_neurons = pop.size
                for i in range(num_neurons): # 逆序写入
                    f.write(f"{int(inf_state[id_start + num_neurons - 1 - i]):012x}\n")
                string_io_dict[output_dir / f"{x}-{y}-is.txt"]=f

            if inf_param is not None:
                f=StringIO()
                num_neurons = pop.size
                for i in range(num_neurons): # 逆序写入
                    f.write(f"{int(inf_param[id_start + num_neurons - 1 - i]):012x}\n")
                string_io_dict[output_dir / f"{x}-{y}-ip.txt"]=f

            if learning_param is not None:
                f=StringIO()
                num_neurons = pop.size
                for i in range(num_neurons):
                    f.write(f"{int(learning_param[id_start + num_neurons - 1 - i]):012x}\n")
                string_io_dict[output_dir / f"{x}-{y}-lp.txt"]=f

            if vt is not None:
                f=StringIO()
                num_neurons = pop.size
                for i in range(num_neurons):
                    f.write(f"{int(vt[id_start + i]):04x}\n")
                string_io_dict[output_dir / f"{x}-{y}-vt.txt"]=f

            # ========================= 寄存器/配置 =========================
            f=StringIO()
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
            string_io_dict[output_dir / f"{x}-{y}-config.dwnc"]=f

            # ========================= 描述文件 =========================
            if need_description:
                desc_path = output_dir / 'desc'
                if not desc_path.exists():
                    Path.mkdir(desc_path)

                desc_file_path = desc_path / f"{x}-{y}-desc.md"
                f=StringIO()
                f.write("# Axon:\n")
                axon_desc, axon_out_desc = pop.axon_description()
                f.write(axon_desc + "\n\n")
                f.write(axon_out_desc + "\n\n")
                f.write("# Dendrite:\n")
                dendrite_desc, neu_id_desc, weight_desc = pop.dendrite_description()
                f.write(dendrite_desc + "\n\n")
                f.write(neu_id_desc + "\n\n")
                f.write(weight_desc + "\n\n")
                string_io_dict[desc_file_path]=f

        # =========================  接着下一个神经元的偏移量  =========================
        id_start += pop.size

    # ========================= 神经元群json文件 =========================
    if output_dir is not None:
        with open(output_dir / (pop_name + ".json"), "w") as f:
            json.dump(pops_mapping, f)
        for _p,_sio in string_io_dict.items():
            _p.write_text(_sio.getvalue(),encoding='utf-8')
            _sio.close()

    return error_flag,pops_usages