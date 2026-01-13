from .tokenizer import tokenize
from .RREG import RREG
class IMacro():
    '''
    宏指令, 仅在编译时候执行
    '''
    pass
class Let(IMacro):
    def __init__(self,op:str) -> None:
        self._op=op

    def compile(self) -> list:
        _ops=tokenize(self._op)
        assert _ops[1] == '=', 'Wrong formate, should be "<var> = Rx"'
        if _ops[0] in RREG.NAME_MAP:
            raise ValueError(f'Variant name "{_ops[0]}" has already defined')
        if _ops[2] not in RREG.NAME_MAP:
            raise ValueError(f'Register "{_ops[2]}" has not defined yet')
        RREG.NAME_MAP[_ops[0]]= RREG.NAME_MAP[_ops[2]]
        return 
