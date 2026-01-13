from .IPseudoInstruction import IPseudoInstruction
from .str_expr import str_expr
from .macro import IMacro
def compile_instructions(inner_insts):
    rslt_list=[]
    for _inst in inner_insts:
        if isinstance(_inst,IPseudoInstruction): # 展开伪指令
            rslt_list.extend(_inst.compile())
        elif isinstance(_inst,str):
            rslt_list.extend(str_expr(_inst).compile())
        elif isinstance(_inst,IMacro):
            _inst.compile() # 执行宏指令
        else:
            rslt_list.append(_inst)
    return rslt_list