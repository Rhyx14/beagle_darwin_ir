def argmax(lst):
    """
    返回列表中最大值的索引。
    
    参数:
        lst (list): 输入的列表。
        
    返回:
        int: 最大值的索引。
    """
    if not lst:
        raise ValueError("列表不能为空")
    
    max_value = lst[0]
    max_index = 0
    
    for i, value in enumerate(lst):
        if value > max_value:
            max_value = value
            max_index = i
    
    return max_index

def bind32(values, widths):
    """
    将values中的值（可能为有符号数）按指定位宽从右到左拼接成一个32位无符号数值.
     
    e.g., value[0]是高位, value[-1]是低位
    
    参数:
        values: 要拼接的值列表（可以是负数）
        width: 对应每个值的位宽列表 (非列表时候代表所有值的位宽)
        
    返回:
        拼接后的32位无符号数值
        
    示例:
        bind32([1, -2, 3], [8, 16, 8]) 
        将1(4位), -2(8位), 3(4位)拼接成 0b00000001_11111111_11111110_00000011 (32位)

    """
    result = 0
    written_bits = 0
    
    # 检查总位宽是否超过32位
    if not isinstance(widths,list):
        widths=[widths for _ in range(len(values))]
    if sum(widths) != 32:
        raise ValueError("Total width should equal to 32 bits")
    
    for _val, _w in zip(values, widths):
        # 处理有符号数
        if isinstance(_val, int) and _val < 0:
            # 计算指定位宽下的最小值
            min_val = -(1 << (_w - 1))
            # 计算指定位宽下的最大值
            max_val = (1 << (_w - 1)) - 1
            
            if _val < min_val:
                raise ValueError(f"Value {_val} underflows its specified signed width {_w} bits (min: {min_val})")
            # 虽然负数不会超过max_val，但为了完整性保留检查
            if _val > max_val:
                raise ValueError(f"Value {_val} overflows its specified signed width {_w} bits (max: {max_val})")
            
            # 将有符号数转换为无符号表示
            _val = _val & ((1 << _w) - 1) # -1 = 0x111, 1000, 
        else:
            # 无符号数检查
            max_unsigned = (1 << _w) - 1
            if _val < 0:
                raise ValueError(f"Value {_val} is negative but not treated as signed for width {_w} bits")
            if _val > max_unsigned:
                raise ValueError(f"Value {_val} exceeds its specified width {_w} bits (max: {max_unsigned})")
        
        # 将当前值左移已占用的位数，然后或到结果中
        written_bits += _w
        result |= (_val << (32-written_bits))
    
    return result & 0xFFFFFFFF  # 确保结果是32位无符号数

from PIL import Image
def draw_gif(frames: list[Image.Image], fps, scale, save_file):
    # 保存为GIF
    frames = list([
        _frame.resize((scale,scale),resample=Image.Resampling.NEAREST)
        for _frame in frames
    ])
    frames[0].save(str(save_file), 
               save_all=True, 
               append_images=frames[1:], 
               duration=1000//fps,  # 每帧持续时间(毫秒)
               loop=0)  # 0表示无限循环

if __name__=='__main__':
    rslt=bind32([1, -2, 3], [8, 16, 8])
    pass