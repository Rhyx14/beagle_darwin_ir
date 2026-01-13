from .str_expr import IPseudoInstruction
from darwin3_deployment.ir import CMP
from .LabledInstruction import Label
from .RREG import RREG
from .compile_instructions import compile_instructions
from .tokenizer import tokenize
from .jmp_inst.jmp_inst import jc,jnc,jmp
class If(IPseudoInstruction):
    LABEL_ID=0
    @staticmethod
    def get_label_id():
        rslt=f'__label_{If.LABEL_ID}'
        If.LABEL_ID +=1
        return rslt

    def __init__(self,op:str,inline_insts:list=[],Else=[]) -> None:
        _ls=tokenize(op)
        self.rs=_ls[0]
        self.op=_ls[1]
        self.rt=_ls[2]
        
        self.inner_insts=inline_insts
        if len(self.inner_insts)==0:
            raise ValueError(f'If block should have some instructions!')
        self.else_insts=Else
    
    def compile(self) -> list:
        rslt=[]
        inverse_flag=False # whether using jc or jnc after cmp
        match self.op:
            case '!=': # 'rs != rt' equals to '!(rs == rt)'
                if self.rs == '0': raise ValueError(f"target reg can not be 0 with '{self.op}'")
                if self.rt == '0':
                    rslt.append(CMP(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rs],CMP.Func.EQ_0))
                else:
                    rslt.append(CMP(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rt],CMP.Func.EQ))
                inverse_flag=True
            case '==':
                if self.rt == '0':
                    rslt.append(CMP(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rs],CMP.Func.EQ_0))
                else:
                    rslt.append(CMP(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rt],CMP.Func.EQ))
            case '>=':
                if self.rt == '0':
                    rslt.append(CMP(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rs],CMP.Func.GE_0))
                else:
                    rslt.append(CMP(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rt],CMP.Func.GE))
            case '<': # 'rs < rt' equals to '!(rs >= rt)'
                if self.rt == '0': 
                    rslt.append(CMP(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rs],CMP.Func.GE_0))
                else:
                    rslt.append(CMP(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rt],CMP.Func.GE))
                inverse_flag=True
            case '>': # 'rs > rt' equals to '!(rt >= rs)'
                if self.rt == '0': raise ValueError(f"target reg can not be 0 with '{self.op}'")
                if self.rs == '0':
                    rslt.append(CMP(RREG.NAME_MAP[self.rt],RREG.NAME_MAP[self.rt],CMP.Func.GE_0))
                else:
                    rslt.append(CMP(RREG.NAME_MAP[self.rt],RREG.NAME_MAP[self.rs],CMP.Func.GE))
                inverse_flag=True
            case '<=':
                if self.rt == '0': raise ValueError(f"target reg can not be 0 with '{self.op}'")
                if self.rs == '0':
                    rslt.append(CMP(RREG.NAME_MAP[self.rt],RREG.NAME_MAP[self.rt],CMP.Func.GE_0))
                else:
                    rslt.append(CMP(RREG.NAME_MAP[self.rt],RREG.NAME_MAP[self.rs],CMP.Func.GE))
            case _:
                raise ValueError(f'erroneous operator {self.op}')

        if inverse_flag:
            rslt.append(jc(label_if:=If.get_label_id()))
        else:
            rslt.append(jnc(label_if:=If.get_label_id()))
        rslt.extend(compile_instructions(self.inner_insts))

        if len(self.else_insts)!=0:
            rslt.append(jmp(label_else:=If.get_label_id())) # skip the else-block after if 
            rslt[-1]=Label(label_if,rslt[-1],1) # jump here if not-if
            rslt.extend(compile_instructions(self.else_insts))
            rslt[-1]=Label(label_else,rslt[-1],1) # adding label
        else:
            rslt[-1]=Label(label_if,rslt[-1],1) # jump here if not-if
        return rslt
