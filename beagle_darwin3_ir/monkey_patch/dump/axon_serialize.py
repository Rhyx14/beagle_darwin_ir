from darwin3_deployment.ir import PhysicalPopulation
def axon_serialize(self:PhysicalPopulation):
    addr = self.size
    target_id_list_table={} # out_did_list: target_did_list index
    axon_binary = {} 
    axon_addrs = []
    for _neuron_id, _axon_ids in self._axons.items():
        if not _axon_ids: continue

        _nid=_axon_ids[0]
        if _nid < 0: _nid = 0 # 将设置为-1的neu_idx改为0

        _out_dids=tuple(_axon_ids[1:])
        
        if _out_dids in target_id_list_table:
            _axon_id = target_id_list_table[_out_dids]
        
        else:
            _axon_id = len(axon_addrs)
            target_id_list_table[_out_dids]=_axon_id

            axon_addrs.append(addr)
            num_addr = len(_out_dids)
            for _i in range(num_addr):  # 目标条目表
                pop, den_id = self._axon_out[_axon_ids[_i + 1]]
                axon_binary[addr + _i] = (
                        (pop.coord[0] < self.coord[0]) << 24
                        | (abs(pop.coord[0] - self.coord[0]) & 0xf) << 20
                        | (pop.coord[1] < self.coord[1]) << 19
                        | (abs(pop.coord[1] - self.coord[1]) & 0xf) << 15
                        | (den_id & 0x7fff)
                )
            addr += num_addr
            axon_binary[addr - 1] += 1 << 25
        axon_binary[_neuron_id] = (axon_addrs[_axon_id] << 12) + _nid

    return axon_binary

from loguru import logger
logger.info('Patch PhysicalPopulation.axon_serialize.')
PhysicalPopulation.axon_serialize = axon_serialize