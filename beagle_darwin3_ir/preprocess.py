from .label import Label,LabledInstructionBase
def preprocess(nc_program:list, show=False,):
    processed_inst=compile_instructions(nc_program)

    label_addr_dict={}
    rslt_inst=[]
    for _pc,_inst in enumerate(processed_inst):
        _iter_inst=_inst
        while isinstance(_iter_inst,Label): # process recursive Label, e.g. Label(Lable(inst))
            label_addr_dict[_iter_inst.label_name]=_pc + _iter_inst.offset
            _iter_inst=_iter_inst.instruction
        rslt_inst.append(_iter_inst)
    # replace the label with addr
    for _i in range(len(rslt_inst)):
        if isinstance(rslt_inst[_i],LabledInstructionBase):
            rslt_inst[_i].process_label(label_addr_dict)
            rslt_inst[_i]=rslt_inst[_i].compile()[0] # labeled instruction only support 1 inst inside.
    
    if show:
        for _line,_inst in enumerate(rslt_inst): print(_line,': \t',str(_inst))
    return rslt_inst

from .pseudo_inst import IPseudoInstruction,str_expr
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