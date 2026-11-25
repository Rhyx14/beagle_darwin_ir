"""分组卷积 / 转置卷积的 PhysicalPopulation 连接构建.

这些工具函数把一组 pre/post PhysicalPopulation 按组切分，再逐组调用
``add_connections`` 建立连接。约定：

- pre_pop  : list[PhysicalPopulation]，每个 pop 的 shape 为 (in_ch, H, W)
- post_pop : list[PhysicalPopulation]，每个 pop 的 shape 为 (out_ch, H', W')
- weight   : (out_ch, in_ch, kh, kw)，分组时必须能被 groups 整除，
             并且每个 pop 的通道数要能按组均分
"""
from darwin3_deployment.codegen import PhysicalPopulation
from darwin3_deployment.codegen import add_connections


def get_nc_channel(pops:list[PhysicalPopulation]):
    """统计一组 pop 的总通道数，pop 的 shape 必须是 (C,H,W)."""
    ch=0
    for _pop in pops:
        assert len(_pop.shape)==3
        ch+=_pop.shape[0]
    return ch


def _divide_group_conv_pop(pre_pop:list[PhysicalPopulation],post_pop:list[PhysicalPopulation],group):
    """按 group 均分 pre/post pop 列表.

    返回 (divided_in_pops, divided_out_pops, in_ch_span, out_ch_span)，
    其中 in_ch_span / out_ch_span 是每组的输入/输出通道数。
    """
    in_ch=get_nc_channel(pre_pop)
    out_ch=get_nc_channel(post_pop)
    in_ch_span=in_ch//group
    out_ch_span=out_ch//group
    assert in_ch_span >= in_ch / len(pre_pop)
    assert out_ch_span >= out_ch /len(post_pop)

    in_pop_span=in_ch_span // pre_pop[0].shape[0]
    out_pop_span=out_ch_span//post_pop[0].shape[0]

    divided_in_pops=  [pre_pop[_i*in_pop_span:(_i+1)*in_pop_span] for _i in range(group)]
    divided_out_pops=[post_pop[_i*out_pop_span:(_i+1)*out_pop_span] for _i in range(group)]
    # make sure any phy_pop does not contains different groups
    assert len(divided_in_pops)*in_pop_span == len(pre_pop)
    assert len(divided_out_pops)*out_pop_span == len(post_pop)
    return divided_in_pops,divided_out_pops,in_ch_span,out_ch_span


def add_group_conv(pre_pops:list[PhysicalPopulation],post_pops:list[PhysicalPopulation],weight,stride,padding,groups):
    """分组卷积连接.

    weight 形状 (out_ch, in_ch, kh, kw)，按输出通道切成 groups 份
    （每份 out_ch_span 个输出通道）。
    """
    divided_in_pops,divided_out_pops,in_ch_span,out_ch_span=_divide_group_conv_pop(pre_pops,post_pops,groups)
    for _i,(_in,_out) in enumerate(zip(divided_in_pops,divided_out_pops)):
        add_connections(
            'conv',
            weight[_i*out_ch_span:(_i+1)*out_ch_span,...],
            _in,_out,
            stride=stride,
            padding=padding
        )


def add_transposed_conv(pre_pops:list[PhysicalPopulation],post_pops:list[PhysicalPopulation],weight,stride=2,padding=1,output_padding=1,groups=1):
    """转置卷积连接.

    weight 形状 (in_ch, out_ch, kh, kw)（PyTorch ConvTranspose2d 的排布），
    按输入通道切成 groups 份（每份 in_ch_span 个通道）。
    """
    divided_in_pops,divided_out_pops,in_ch_span,out_ch_span=_divide_group_conv_pop(pre_pops,post_pops,groups)
    for _i,(_in,_out) in enumerate(zip(divided_in_pops,divided_out_pops)):
        add_connections(
            'trans_conv',
            weight[_i*in_ch_span:(_i+1)*in_ch_span,...],
            _in,_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
