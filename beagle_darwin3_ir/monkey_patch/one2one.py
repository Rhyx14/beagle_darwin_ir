from darwin3_deployment.codegen import check_distance,Sequence,_validate_neuron_ids,PhysicalPopulation
import numpy as np
from typing import List,Any,Union
def connect_one2one(
          weight: int | np.ndarray, 
          pre_pops: List[PhysicalPopulation],
          post_pops:List[PhysicalPopulation],
          conn_table: np.ndarray | None =None,
          compress: bool=True,
          omit_zero_weight=True):
    """
    connect physical populations one by one
    
    weight: connection weight

    pre_pops: pre-synaptic neuron populations

    post_pops: post-synaptic neuron populations

    shuffle_list: connections info, if is set to None, connect neurons in order,
    
        in formate [[pre,post],[pre,post],...[pre,post]].

        note that neuron ids go across all the pops in the list.

    compress: whether compress the weight storage.

    omit_zero_weight: omit the zero weight
    """     
    if conn_table is not None:
        num_conn=conn_table.shape[0]
        assert num_conn == (_t:=sum(map(lambda x: x.size,pre_pops))), f"Expect {num_conn} pre-neurons, got {_t} instead"
        assert num_conn == (_t:=sum(map(lambda x: x.size,post_pops))), f"Expect {num_conn} post-neurons, got {_t} instead"
    else:
        num_conn=sum(map(lambda x: x.size,pre_pops))
        assert sum(map(lambda x: x.size,pre_pops))==(_p:=sum(map(lambda x: x.size,post_pops))), f"Number of pre-neurons should equal to the post-neurons, got 'pre {num_conn} - post{_p}' instead"
    
    if isinstance(weight,np.ndarray):
        assert weight.shape==(num_conn,num_conn), f"got wrong weight shape {weight.shape}"
    else:
        weight=np.zeros(num_conn,dtype=np.int32)+weight
    
    # map from id to pop_id
    pre_id_pop_map=np.concatenate([np.ones(_pop.size,dtype=np.int32)*_i for _i,_pop in enumerate(pre_pops)])
    post_id_pop_map=np.concatenate([np.ones(_pop.size,dtype=np.int32)*_i for _i,_pop in enumerate(post_pops)])
    pre_id_pop_id_map=np.concatenate([np.arange(_pop.size,dtype=np.int32) for _pop in pre_pops])
    post_id_pop_id_map=np.concatenate([np.arange(_pop.size,dtype=np.int32) for _pop in post_pops])

    if conn_table is None:
        conn_table=np.arange(num_conn)[:,None].repeat(2,axis=1)

    for _weight,(_global_pre_id,_global_post_id) in zip(weight,conn_table):
        if omit_zero_weight and _weight==0: continue
        _add_one2one_connection(
            _weight,
            pre_pops[pre_id_pop_map[_global_pre_id]],post_pops[post_id_pop_map[_global_post_id]],
            pre_id_pop_id_map[_global_pre_id],post_id_pop_id_map[_global_post_id],
            compress
        )
    pass

@check_distance
def _add_one2one_connection(weight: int, pre_pop: PhysicalPopulation, post_pop: PhysicalPopulation,
                            pre_neuron_id: int, post_neuron_id:int,compress_weight):
    """添加连接(1 to 1)

    Args:
        weight: 权重向量
        pre_pop: 突触前神经元群
        post_pop: 突触后神经元群
        pre_neu_id: 突触前神经元编号
        post_neu_ids: 突触后神经元编号
        relay_coord:

    Notes:
        - `weight` 和 `post_neu_ids` 均为向量，且长度应相同。
        - `post_neu_ids` 为 `None` 时，表示连接到突触后神经元群中的所有神经元。
    """
    length=1
    
    post_neuron_table_id = post_pop._register(post_pop._neuron_ids, post_pop._neuron_ids_registry,[post_neuron_id])
    if compress_weight:
        waddr = post_pop._register(post_pop._weights, post_pop._weights_registry, (weight,))
    else:
        waddr=len(post_pop._weights) # waddr is always different
        post_pop._weights.append(weight)

    dendrite_table_entry = (post_neuron_table_id, 0, length, False, waddr)
    den_id = post_pop._register(
             post_pop._dendrites, post_pop._dendrites_registry, 
             dendrite_table_entry)
    
    pre_pop.add_axon(pre_neuron_id, -1, post_pop, den_id)
    return