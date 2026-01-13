from darwin3_deployment.ir import MOV,ADD,ADDI,SUB,MUL,SFTI
from .RREG import RREG
from .tokenizer import tokenize
from .IPseudoInstruction import IPseudoInstruction
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