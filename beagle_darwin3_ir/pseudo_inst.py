from darwin3_deployment.ir import JC,SUB,ADDI,MOV,ADD
from .label import LabledInstructionBase
from .RREG import RREG
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

class set_value(IPseudoInstruction):
    '''
    Assign value for a given register.
    '''
    def __init__(self,reg,imme) -> None:
        super().__init__()
        self.reg=reg
        self.imme=imme
    def compile(self) -> list:
        return [
            SUB(self.reg,self.reg),
            ADDI(self.reg,self.imme)
        ]

class expr(IPseudoInstruction):
    def __init__(self,expr):
        super().__init__()
        _ls=expr.split(' ')
        _ls=[_s for _s in _ls if _s!='']
        self.rs=_ls[0]
        self.op=_ls[1]
        self.rt=_ls[2]
    
    def compile(self):
        if self.rs not in RREG.R_NAME_MAP:
            raise ValueError(f'Left value should be a register, got "{self.rs}" instead.')
        if self.op=='=':
            if self.rt in RREG.R_NAME_MAP:
                return [MOV(RREG.R_NAME_MAP[self.rs],RREG.R_NAME_MAP[self.rt])]
            else:
                return [SUB(RREG.R_NAME_MAP[self.rs],RREG.R_NAME_MAP[self.rs]),
                        ADDI(RREG.R_NAME_MAP[self.rs],int(self.rt))]
        elif self.op=='+=':
            if self.rt in RREG.R_NAME_MAP:
                return [ADD(RREG.R_NAME_MAP[self.rs],RREG.R_NAME_MAP[self.rt],0)]
            else:
                return [ADDI(RREG.R_NAME_MAP[self.rs],int(self.rt))]
        elif self.op=='-=':
            if self.rt in RREG.R_NAME_MAP:
                return [SUB(RREG.R_NAME_MAP[self.rs],RREG.R_NAME_MAP[self.rt],0)]
            else:
                return [ADDI(RREG.R_NAME_MAP[self.rs],-int(self.rt))]
        else:
            raise ValueError(f'Undefined operator "{self.op}"')