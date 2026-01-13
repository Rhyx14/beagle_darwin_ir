from ..IPseudoInstruction import IPseudoInstruction
from ..LabledInstruction import LabledInstructionBase
from darwin3_deployment.ir import JC
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

