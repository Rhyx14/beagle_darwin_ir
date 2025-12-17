# beagle ir

为 Darwin3 芯片开发提供伪代码功能支持。
Provides pseudo-code functionality support for Darwin3 chip development.

为 Darwin3 Beagle 开发环境提供部分功能支持。
Offers functionality support for the Darwin3 Beagle development environment.

兼容 darwin3_development==0.6.3 SDK 版本。
Compatible with the darwin3_development==0.6.3 SDK version.

# ir支持 IR Support

使用前请用 preprocess 方法处理包含 Beagle IR 指令的列表。
Use the preprocess method to handle lists containing Beagle IR instructions before use.

```python
from beagle_darwin3_ir import *
# Before
coreConfig = CoreConfig(
        assembly_program=[
            # your darwin assembly code
        ],
        inference_state_settings={
            # state 
        },
    )

# After
coreConfig = CoreConfig(
        assembly_program=preprocess(show=True,nc_program=[
            # darwin assembly code with beagle ir
        ]),
        inference_state_settings={
            # state
        },
    )
```

## 支持操作 Supported Operations
```python
from beagle_darwin3_ir import *
coreConfig = CoreConfig(
        assembly_program=preprocess(show=True,nc_program=[
            # darwin assembly code with beagle ir
            
            # marco instructions
            Let('var name = r1'), # bind variable name with a specific register.

            # string expression
            # support operators: [+=,-=, >>,<<, =]
            'r1 += r2',
            'r1 += -1',
            'r1 = r2', 
            'r1 = 0',

            # `If-Else' block
            # the condition expression can be:
            #    r1 < r2, or [>=,<,<=,==,!=]. compare two registers
            #    r1 < 0, or [>=,<,<=]. compare register with 0
            IF(condition:str, [
                # assembly list
            ],Else[
                # assembly list
            ]),

            # `If' block
            IF(condition:str, [
                # assembly list
            ]),


        ]),
        inference_state_settings={
            # state
        },
    )
```

## 寄存器别名 Register Aliases

见 RREG.NAME_MAP 字典内容。
Please refer to the RREG.NAME_MAP dictionary for details.

注意这些寄存器均为 16 位，需要自行移位以获取真实值。
Note that these registers are all 16-bit; you need to shift them accordingly to obtain the actual value.
