from darwin3_deployment.ir import *
from .label import LabledInstructionBase
from .RREG import RREG
from .tokenizer import tokenize
class IPseudoInstruction():
    def compile(self) -> list:
        raise NotImplementedError
 
class jnc(LabledInstructionBase,IPseudoInstruction):
    '''
    jump if the condition is unmet
    '''
    def compile(self) -> list:
        if self.addr is not None: return [JC(1,1,self.addr)]
        else: return []

class jc(LabledInstructionBase,IPseudoInstruction):
    '''
    jump if the condition is met
    '''
    def compile(self) -> list:
        if self.addr is not None: return [JC(1,0,self.addr)]
        else: return []

class jmp(LabledInstructionBase,IPseudoInstruction):
    '''
    jump whatever the condition is
    '''
    def compile(self) -> list:
        if self.addr is not None: return [JC(0,0,self.addr)]
        else: return []

class str_expr(IPseudoInstruction):
    def __init__(self,expr):
        super().__init__()
        _ls=tokenize(expr)
        _ls=[_s for _s in _ls if _s!='']
        self.rs=_ls[0]
        self.op=_ls[1]
        self.rt=_ls[2]
    
    def compile(self):
        if self.rs not in RREG.NAME_MAP:
            raise ValueError(f'Left value should be a register, got "{self.rs}" instead.')
        match self.op:
            case '=':
                if self.rt in RREG.NAME_MAP:
                    return [MOV(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rt])]
                else:
                    if (imme:=int(self.rt)) ==0:
                        return [SUB(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rs])]
                    else:
                        return [SUB(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rs]),
                                ADDI(RREG.NAME_MAP[self.rs],imme)]
            case '+=':
                if self.rt in RREG.NAME_MAP:
                    return [ADD(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rt],1)]
                else:
                    return [ADDI(RREG.NAME_MAP[self.rs],int(self.rt))]
            case '-=':
                if self.rt in RREG.NAME_MAP:
                    return [SUB(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rt],1)]
                else:
                    return [ADDI(RREG.NAME_MAP[self.rs],-int(self.rt))]
            case '*=':
                return [MUL(RREG.NAME_MAP[self.rs],RREG.NAME_MAP[self.rt],MUL.QF.GLOBAL,0)]

            case '<<':
                return [SFTI(RREG.NAME_MAP[self.rs],SFTI.SF.LEFT,1,int(self.rt))]
            case '>>':
                return [SFTI(RREG.NAME_MAP[self.rs],SFTI.SF.RIGHT,1,int(self.rt))]
            case _:
                raise ValueError(f'Undefined operator "{self.op}"') 