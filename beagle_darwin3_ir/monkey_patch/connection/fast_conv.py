from darwin3_deployment.codegen import check_distance,Sequence,_validate_neuron_ids,PhysicalPopulation,_get_pair
import numpy as np
from collections import namedtuple
from functools import lru_cache
from typing import List,Any,Union

def _write_conv_dendrites(
    pre_pop,
    post_pop,
    pre_to_post_ids,
    pre_to_weights,
    pattern_keys,
    base_ids,
    spatial_size,
    in_channels,
    is_rel,
):
    """按 (边界模式, 输入通道) 共享树突，直接写入 PhysicalPopulation。

    这是 add_conv_connection 的私有实现细节，不属于公开 API。

    遍历顺序为 in_ch 外层、pre 内层，与 SDK 原 _add_m2n_connection 的
    pos 下标顺序（i = pre_spatial + in_ch * spatial_size）完全一致，
    因此 _dendrites / _axon_out 的注册顺序与去重后的 id 都保持不变。
    """
    axons = pre_pop._axons
    axon_out, axon_out_reg = pre_pop._axon_out, pre_pop._axon_out_registry
    neuron_ids, neuron_ids_reg = post_pop._neuron_ids, post_pop._neuron_ids_registry
    weights, weights_reg = post_pop._weights, post_pop._weights_registry
    dendrites, dendrites_reg = post_pop._dendrites, post_pop._dendrites_registry

    base_list = [int(v) for v in base_ids]   # 每个前神经元的绝对基准
    nid_cache = {}    # pattern_key -> neuron_table_id（仅 type=4 使用）
    out_cache = {}    # pattern_key -> [out_id per in_ch]

    for _in_ch in range(in_channels):
        for pre_spatial in range(spatial_size):
            key = pattern_keys[pre_spatial]
            if key is None:                      # padding-only 的前神经元
                continue

            if is_rel:
                # 相对寻址时，同一 (模式, in_ch) 的树突内容完全相同，可安全共享
                _outs = out_cache.get(key)
                if _outs is None:
                    _outs = [None] * in_channels
                    out_cache[key] = _outs
                _out_id = _outs[_in_ch]
            else:
                # 绝对寻址时每条树突的 neuron_id 各不相同，不能按模式合并
                _outs = None
                _out_id = None

            if _out_id is None:                   # 该 (模式, in_ch) 第一次出现
                ids = pre_to_post_ids[pre_spatial]
                base = base_list[pre_spatial]

                if is_rel:
                    nid = nid_cache.get(key)
                    if nid is None:
                        nid = post_pop._register(
                            neuron_ids, neuron_ids_reg, (ids - base).tolist())
                        nid_cache[key] = nid
                    conn_type = 4
                else:
                    nid = post_pop._register(
                        neuron_ids, neuron_ids_reg, ids.tolist())
                    conn_type = 0

                w = pre_to_weights[pre_spatial + _in_ch * spatial_size]
                wid = post_pop._register(weights, weights_reg, w.tolist())
                den_id = post_pop._register(
                    dendrites, dendrites_reg,
                    [nid, conn_type, ids.size, False, wid],
                )
                _out_id = pre_pop._register(
                    axon_out, axon_out_reg, (post_pop, den_id))
                if is_rel:
                    _outs[_in_ch] = _out_id

            ax = axons[pre_spatial + _in_ch * spatial_size]
            if not ax:
                ax.append(base_list[pre_spatial] if is_rel else -1)
            elif is_rel and ax[0] == -1:
                ax[0] = base_list[pre_spatial]
            ax.append(_out_id)


# ===================== 权重无关几何缓存 =====================
# add_conv_connection 中与权重数值无关的部分（网格、有效抽头、稳定排序、按前
# 神经元分组、权重下标）只取决于 (前/后形状, 核尺寸, stride, padding, dilation)。
# ResNet 的 3600 次调用只有 13 种签名，缓存后避免重复构造。
_ConvGeometry = namedtuple(
    "_ConvGeometry",
    [
        "pre_to_post_ids",  # object[spatial_size*in_channels]：后神经元 id（与 in_ch 无关）
        "weight_indices",   # object[spatial_size*in_channels]：kernel.ravel() 下标
        "pattern_keys",     # object[spatial_size]：边界模式 bytes（padding-only 为 None）
        "base_ids",         # int64[spatial_size]：每个前神经元的绝对基准 id
        "spatial_size",     # int：单通道空间大小 in_h * in_w
        "in_channels",      # int：输入通道数
    ],
)


@lru_cache(maxsize=30)
def _get_conv_geometry(pre_shape, post_shape, kernel_h, kernel_w, stride, padding, dilation):
    """构造与权重数值无关的卷积连接几何，结果只读，调用方不得原地修改。

    所有入参必须是可哈希的整数元组（stride/padding/dilation 为 (h, w) 二元组），
    由 add_conv_connection 归一化后传入，以便 lru_cache 得到稳定键。

    重量（kernel 数值）不在此处缓存：weight_indices 只是下标，每次调用再用
    当次的 kernel.ravel() 取值。
    """
    in_channels, in_h, in_w = (int(v) for v in pre_shape)
    out_channels, out_h, out_w = (int(v) for v in post_shape)
    kernel_h, kernel_w = int(kernel_h), int(kernel_w)
    sh, sw = (int(v) for v in stride)
    ph, pw = (int(v) for v in padding)
    dh, dw = (int(v) for v in dilation)

    # base 网格：每个输出位置对应的输入侧采样锚点（kernel 元素 (0,0) 的落点，可为负）
    base_h_grid, base_w_grid = np.meshgrid(
        np.arange(-ph, sh * (out_h - 1) - ph + 1, sh),
        np.arange(-pw, sw * (out_w - 1) - pw + 1, sw),
        indexing="ij",
    )
    # 卷积核内部的相对偏移
    kernel_h_grid, kernel_w_grid = np.meshgrid(np.arange(kernel_h), np.arange(kernel_w), indexing="ij")

    # 广播到 (out_h, out_w, kernel_h, kernel_w)
    base_h_grid = np.broadcast_to(base_h_grid[:, :, None, None], base_h_grid.shape + kernel_h_grid.shape)
    base_w_grid = np.broadcast_to(base_w_grid[:, :, None, None], base_w_grid.shape + kernel_h_grid.shape)

    # 每个 (输出位置, 核元素) 组合对应的输入特征图坐标
    in_h_grid = base_h_grid + kernel_h_grid[None, None, :, :] * dh
    in_w_grid = base_w_grid + kernel_w_grid[None, None, :, :] * dw

    # padding 越界的抽头直接剔除（等价 zero-padding）
    valid_mask = (in_h_grid >= 0) & (in_h_grid < in_h) & (in_w_grid >= 0) & (in_w_grid < in_w)
    valid_positions = np.argwhere(valid_mask)
    out_h_idx, out_w_idx, kh_idx, kw_idx = (
        valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2], valid_positions[:, 3]
    )

    # 高级索引：四个索引数组逐行配对，结果长度为有效抽头数。
    # base_*_pos 仍在输入坐标系（锚点，可为负）；in_*_pos 保证落在 [0, in_h/in_w)。
    base_h_pos = base_h_grid[out_h_idx, out_w_idx, kh_idx, kw_idx]
    base_w_pos = base_w_grid[out_h_idx, out_w_idx, kh_idx, kw_idx]
    in_h_pos = in_h_grid[out_h_idx, out_w_idx, kh_idx, kw_idx]
    in_w_pos = in_w_grid[out_h_idx, out_w_idx, kh_idx, kw_idx]

    # 单通道内的展平坐标
    pre_spatial_id = in_h_pos * in_w + in_w_pos
    # 输出侧：通过 (base + padding) // stride 反算输出索引
    post_spatial_id = (base_h_pos + ph) // sh * out_w + (base_w_pos + pw) // sw
    # 卷积核平面内的展平偏移（不含通道维）
    kernel_pos = kh_idx * kernel_w + kw_idx

    # 按前神经元分组：稳定排序保证组内顺序与 np.argwhere 遍历顺序一致
    order = np.argsort(pre_spatial_id, kind="stable")
    sorted_pre = pre_spatial_id[order]
    sorted_post = post_spatial_id[order]
    sorted_kernel_pos = kernel_pos[order]
    spatial_size = in_h * in_w
    offsets = np.searchsorted(sorted_pre, np.arange(spatial_size + 1))

    # 通道维度的展平步长
    out_channel_stride = np.arange(out_channels) * out_h * out_w
    kernel_out_ch_stride = np.arange(out_channels) * in_channels * kernel_h * kernel_w
    kernel_in_ch_stride = kernel_h * kernel_w

    pre_to_post_ids = np.empty(spatial_size * in_channels, dtype=object)
    weight_indices = np.empty(spatial_size * in_channels, dtype=object)
    pattern_keys = np.empty(spatial_size, dtype=object)   # 每个前神经元的边界模式
    base_ids = np.empty(spatial_size, dtype=np.int64)     # 每个前神经元的绝对基准 id

    for pre_spatial in range(spatial_size):
        start, end = offsets[pre_spatial], offsets[pre_spatial + 1]
        if start == end:
            continue

        post_seg = sorted_post[start:end]
        kpos_seg = sorted_kernel_pos[start:end]

        # 同一个空间坐标复制到所有输出通道；in_ch 维度只影响权重下标的通道偏移
        target_ids = (post_seg[:, None] + out_channel_stride).T.ravel()
        kernel_positions = (kpos_seg[:, None] + kernel_out_ch_stride).T.ravel()

        pattern_keys[pre_spatial] = kpos_seg.tobytes()
        base_ids[pre_spatial] = target_ids[0]

        for in_ch in range(in_channels):
            neuron_index = pre_spatial + in_ch * spatial_size
            pre_to_post_ids[neuron_index] = target_ids
            weight_indices[neuron_index] = kernel_positions + in_ch * kernel_in_ch_stride

    return _ConvGeometry(
        pre_to_post_ids, weight_indices, pattern_keys, base_ids, spatial_size, in_channels,
    )


@check_distance
def add_conv_connection(
    weight: np.ndarray,
    pre_pop: PhysicalPopulation,
    post_pop: PhysicalPopulation,
    stride: Union[int, tuple[int, int]] = 1,
    padding: Union[int, tuple[int, int]] = 0,
    dilation: Union[int, tuple[int, int]] = 1,
):
    """为两个神经元群添加卷积连接

    Args:
        weight: 卷积核权重矩阵，形状 (out_channels, in_channels, kernel_h, kernel_w)
        pre_pop: 突触前神经元群，形状 (in_channels, in_h, in_w)
        post_pop: 突触后神经元群，形状 (out_channels, out_h, out_w)
        stride: 卷积步长
        padding: 卷积填充
        dilation: 卷积膨胀率
    """
    kernel = np.asarray(weight, dtype=int)
    if kernel.ndim != 4:
        raise ValueError("The weight of convolution connection must be a four-dimensional array.")
    if len(pre_pop.shape) != 3:
        raise ValueError("The shape of the presynaptic population must be a "
                         "three-dimensional array (channels, height, width).")
    if len(post_pop.shape) != 3:
        raise ValueError("The shape of the postsynaptic population must be a "
                         "three-dimensional array (channels, height, width).")

    # 归一化签名：(前形状, 后形状, 核高, 核宽, stride, padding, dilation)。
    # 只有这些值影响与权重无关的几何，因此可直接作为缓存键。
    stride_t = tuple(int(v) for v in _get_pair(stride))
    padding_t = tuple(int(v) for v in _get_pair(padding))
    dilation_t = tuple(int(v) for v in _get_pair(dilation))
    pre_shape_t = tuple(int(v) for v in pre_pop.shape)
    post_shape_t = tuple(int(v) for v in post_pop.shape)
    kernel_h, kernel_w = (int(v) for v in kernel.shape[-2:])

    geom = _get_conv_geometry(
        pre_shape_t, post_shape_t, kernel_h, kernel_w,
        stride_t, padding_t, dilation_t,
    )

    # 外层浅拷贝：缓存对象保持只读，返回给调用方的数组可安全持有
    pre_to_post_ids = geom.pre_to_post_ids.copy()

    # 权重每次调用都不同：仅按缓存的下标从当次 kernel 取值
    kernel_flat = kernel.ravel()
    pre_to_weights = np.empty(geom.spatial_size * geom.in_channels, dtype=object)
    for neuron_index, weight_index in enumerate(geom.weight_indices):
        if weight_index is None:
            continue
        pre_to_weights[neuron_index] = kernel_flat[weight_index]

    # 与原始实现相同的全局 type 决策：所有前神经元都与自己的历史基准一致才用相对寻址
    is_rel = pre_pop._validate_neu_index(pre_to_post_ids)

    _write_conv_dendrites(
        pre_pop, post_pop,
        pre_to_post_ids, pre_to_weights,
        geom.pattern_keys, geom.base_ids,
        geom.spatial_size, geom.in_channels, is_rel,
    )
    return pre_to_post_ids, pre_to_weights


# ===================== monkey patch 安装 =====================
# 目的：让 darwin3_deployment 的调用方使用本文件的实现。
#
# 需要覆盖的绑定有三处：
#   1. codegen.add_conv_connection  —— 公开名；
#   2. codegen.add_to_dendrite_conv —— codegen.py:238 在 import 时生成的别名，
#      codegen.add_connections() 内部（codegen.py:499）调用的就是它；
#   3. darwin3_deployment.auto.build.add_to_dendrite_conv
#      —— build.py:13 用 `from darwin3_deployment.codegen import *` 又绑定了一份。
#      该模块若在本模块之后才导入，星号导入会拿到已打补丁的 codegen；
#      若它已经先导入，则必须同步覆盖它的模块级名字。
#
# 注意：
#   - 本函数已带 @check_distance，这里直接赋值，不能再套一次装饰器；
#   - 已经在别处用 `from ... import add_conv_connection` 提前绑定的引用无法追溯，
#     必须保证先 `import beagle_darwin3_ir` 再导入/调用 SDK。
import sys as _sys
import darwin3_deployment.codegen as _codegen

# 保留原函数引用，便于调试与对比
_ORIGINAL_ADD_CONV_CONNECTION = _codegen.add_conv_connection


def install_monkey_patch() -> None:
    """把本文件的 add_conv_connection 安装到 darwin3_deployment 中（幂等）。"""
    _codegen.add_conv_connection = add_conv_connection
    _codegen.add_to_dendrite_conv = add_conv_connection

    _build_module = _sys.modules.get("darwin3_deployment.auto.build")
    if _build_module is not None:
        if hasattr(_build_module, "add_to_dendrite_conv"):
            _build_module.add_to_dendrite_conv = add_conv_connection
        if hasattr(_build_module, "add_conv_connection"):
            _build_module.add_conv_connection = add_conv_connection

from loguru import logger
logger.info('Patch conv connection.')
install_monkey_patch()