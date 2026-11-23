from typing import Union,Optional,List
from darwin3_deployment.codegen import check_distance,Sequence,PhysicalPopulation
import numpy as np
def connect_linear(
        weight: np.ndarray, 
        pre_pops: List[PhysicalPopulation],
        post_pops: List[PhysicalPopulation],
        compress_weight=True):
    """添加标准连接(M to N)
    Args:
        weight: 权重向量 [N, M] # 注意顺序！
        pre_pops: 突触前神经元群列表
        post_pops: 突触后神经元群列表
    """
    assert weight.shape[1] == (_t:=sum(map(lambda x: x.size,pre_pops))), f"Expect {weight.shape[1]} pre-neurons, got {_t} instead"
    assert weight.shape[0] == (_t:=sum(map(lambda x: x.size,post_pops))), f"Expect {weight.shape[0]} post-neurons, got {_t} instead"

    weight_in_offset=0
    weight_out_offset=0
    for _pre in pre_pops:
        for _post in post_pops:
            add_pop_linear_connection(
                weight[weight_out_offset:weight_out_offset+_post.size,
                       weight_in_offset: weight_in_offset+ _pre.size],
                _pre,_post,
                compress_weight
            )
            weight_out_offset+=_post.size
        weight_in_offset+=_pre.size

@check_distance
def add_pop_linear_connection(weight: np.ndarray, pre_pop: PhysicalPopulation, post_pop: PhysicalPopulation, compress_weight):
    """添加标准连接(1 to N)

    Args:
        weight: 权重向量 [out, in]
        pre_pop: 突触前神经元群
        post_pop: 突触后神经元群
    Notes:
        - `weight` 和 `post_neu_ids` 均为向量，且长度应相同。
        - `post_neu_ids` 为 `None` 时，表示连接到突触后神经元群中的所有神经元。
    """
    weight_ = np.asarray(weight, dtype=int)
    if weight_.ndim != 2:
        raise ValueError("The weight of full connection must be a two-dimensional array.")
    _post_neu_ids = list(range(post_pop.size))

    # is_neu_index_available = pre_pop._validate_neu_index(np.zeros((pre_pop.size, 1)))
    # connection_type, neu_index = (4, 0) if is_neu_index_available else (0, -1)
    connection_type, neu_index =  (0, -1)

    in_pop_size,out_pop_size=weight.shape[1],weight.shape[0]
    
    assert in_pop_size <= pre_pop.size, f"Expect {in_pop_size} neurons in pre-neurongroup, got {out_pop_size} instead"
    assert out_pop_size <= out_pop_size, f"Expect {in_pop_size} neurons in pre-neurongroup, got {out_pop_size} instead"
    
    for _pre_neu_id in range(pre_pop.size):
        den_id = _add_linear_dendrite(post_pop,_post_neu_ids, weight_[:, _pre_neu_id].tolist(), type=connection_type, compress_weight=compress_weight)
        pre_pop.add_axon(_pre_neu_id, neu_index, post_pop, den_id)

def _add_linear_dendrite(
          post_pop:PhysicalPopulation, 
          post_neuron_id_list: list[int], 
          weight: Union[list[int], int],
          type: int, 
          compress_weight: bool) -> int:
        """添加树突数据

        Args:
            neuron_id: 突触后神经元序号
            weight: 权重
            share: 是否共享权重
            type: 连接类型
            length: 突触后神经元个数

        Returns:
            树突编号

        Notes:
            `share` 为 `True` 时，`weight` 应为一个整数；`share` 为 `False` 时，`weight` 应为一个整数列表
        """
        length = len(post_neuron_id_list)
        post_neuron_table_id = post_pop._register(post_pop._neuron_ids, post_pop._neuron_ids_registry, post_neuron_id_list)

        if compress_weight:
            waddr = post_pop._register(post_pop._weights, post_pop._weights_registry, weight)
        else:
            waddr=len(post_pop._weights) # waddr is always different
            post_pop._weights.append(weight)

        dendrite_table_entry = (post_neuron_table_id, type, length, False, waddr)
        den_id = post_pop._register(
                 post_pop._dendrites, post_pop._dendrites_registry, 
                 dendrite_table_entry)
        return den_id