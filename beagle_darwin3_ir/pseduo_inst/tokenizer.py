from typing import List, Tuple, Dict, Set, Iterator

class PythonTokenizer:
    """
    Python语法分词器（不使用正则表达式）
    支持Python的所有主要分词操作
    """
    
    # Python关键字
    KEYWORDS = {
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 
        'try', 'while', 'with', 'yield'
    }
    
    # 运算符（包含箭头运算符）
    OPERATORS = {
        '+', '-', '*', '**', '/', '//', '%', '@', '->',
        '<<', '>>', '&', '|', '^', '~',
        '<', '>', '<=', '>=', '==', '!=',
        '=', '+=', '-=', '*=', '/=', '//=', '%=', '@=',
        '&=', '|=', '^=', '>>=', '<<=', '**=',
    }
    
    # 分隔符
    DELIMITERS = {
        '(', ')', '[', ']', '{', '}', ',', ':', '.', ';'
    }
    
    # 空白字符
    WHITESPACE = {' ', '\t', '\n', '\r', '\f', '\v'}
    
    def __init__(self):
        # 构建字符分类查找表
        self.char_types = {}
        
        # 数字字符
        for c in '0123456789':
            self.char_types[c] = 'digit'
        
        # 字母和下划线
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_':
            self.char_types[c] = 'letter'
        
        # 运算符首字符
        for c in '+-*/%&|^~<>=!@':
            self.char_types[c] = 'operator'
        
        # 分隔符
        for c in '()[]{}:,.;':
            self.char_types[c] = 'delimiter'
        
        # 引号
        for c in '\'"':
            self.char_types[c] = 'quote'
        
        # 空白字符
        for c in self.WHITESPACE:
            self.char_types[c] = 'whitespace'
        
        # 反斜杠
        self.char_types['\\'] = 'backslash'
        
        # 注释字符
        self.char_types['#'] = 'comment'
    
    def get_char_type(self, char: str) -> str:
        """获取字符类型"""
        return self.char_types.get(char, 'unknown')
    
    def tokenize(self, code: str) -> List[Tuple[str, str]]:
        """
        将Python代码字符串分词为token列表
        
        Args:
            code: Python代码字符串
            
        Returns:
            List[Tuple[str, str]]: token列表，每个token是(类型, 值)的元组
        """
        tokens = []
        pos = 0
        length = len(code)
        
        while pos < length:
            current_char = code[pos]
            char_type = self.get_char_type(current_char)
            
            if char_type == 'whitespace':
                # 处理空白字符
                start = pos
                while pos < length and self.get_char_type(code[pos]) == 'whitespace':
                    pos += 1
                tokens.append(('WHITESPACE', code[start:pos]))
            
            elif char_type == 'digit':
                # 处理数字
                token_value, new_pos = self._parse_number(code, pos)
                if '.' in token_value or 'e' in token_value.lower():
                    tokens.append(('FLOAT', token_value))
                else:
                    tokens.append(('INTEGER', token_value))
                pos = new_pos
            
            elif char_type == 'letter':
                # 处理标识符或关键字
                token_value, new_pos = self._parse_identifier(code, pos)
                if token_value in self.KEYWORDS:
                    tokens.append(('KEYWORD', token_value))
                else:
                    tokens.append(('IDENTIFIER', token_value))
                pos = new_pos
            
            elif char_type == 'quote':
                # 处理字符串
                token_value, new_pos = self._parse_string(code, pos)
                tokens.append(('STRING', token_value))
                pos = new_pos
            
            elif char_type == 'comment':
                # 处理注释
                token_value, new_pos = self._parse_comment(code, pos)
                tokens.append(('COMMENT', token_value))
                pos = new_pos
            
            elif char_type == 'operator':
                # 处理运算符（包含箭头）
                token_value, new_pos = self._parse_operator(code, pos)
                tokens.append(('OPERATOR', token_value))
                pos = new_pos
            
            elif char_type == 'delimiter':
                # 处理分隔符
                tokens.append(('DELIMITER', current_char))
                pos += 1
            
            elif char_type == 'backslash':
                # 处理行继续符
                if pos + 1 < length and code[pos + 1] == '\n':
                    tokens.append(('LINE_CONTINUATION', '\\'))
                    pos += 2
                else:
                    tokens.append(('OPERATOR', '\\'))
                    pos += 1
            
            else:
                # 未知字符
                tokens.append(('UNKNOWN', current_char))
                pos += 1
        
        return tokens
    
    def _parse_number(self, code: str, start: int) -> Tuple[str, int]:
        """解析数字"""
        pos = start
        length = len(code)
        has_dot = False
        has_exponent = False
        
        while pos < length:
            char = code[pos]
            char_type = self.get_char_type(char)
            
            if char_type == 'digit':
                pos += 1
            elif char == '.' and not has_dot and not has_exponent:
                if pos + 1 < length and self.get_char_type(code[pos + 1]) == 'digit':
                    has_dot = True
                    pos += 1
                else:
                    break
            elif char.lower() == 'e' and not has_exponent:
                if pos + 1 < length and (code[pos + 1] in '+-' or 
                                       self.get_char_type(code[pos + 1]) == 'digit'):
                    has_exponent = True
                    pos += 1
                else:
                    break
            elif char in '+-' and has_exponent and pos > 0 and code[pos - 1].lower() == 'e':
                pos += 1
            else:
                break
        
        return code[start:pos], pos
    
    def _parse_identifier(self, code: str, start: int) -> Tuple[str, int]:
        """解析标识符"""
        pos = start
        length = len(code)
        
        while pos < length:
            char_type = self.get_char_type(code[pos])
            if char_type in ('letter', 'digit'):
                pos += 1
            else:
                break
        
        return code[start:pos], pos
    
    def _parse_string(self, code: str, start: int) -> Tuple[str, int]:
        """解析字符串"""
        quote_char = code[start]
        pos = start + 1
        length = len(code)
        
        # 检查是否是三重引号
        is_triple = (pos + 1 < length and 
                    code[pos] == quote_char and 
                    code[pos + 1] == quote_char)
        
        if is_triple:
            end_quote = quote_char * 3
            pos += 2
        else:
            end_quote = quote_char
        
        while pos < length:
            if code[pos] == '\\':
                # 转义字符，跳过下一个字符
                pos += 2
                continue
            
            if is_triple:
                if pos + 2 < length and code[pos:pos + 3] == end_quote:
                    pos += 3
                    break
            else:
                if code[pos] == end_quote:
                    pos += 1
                    break
            
            pos += 1
        
        return code[start:pos], pos
    
    def _parse_comment(self, code: str, start: int) -> Tuple[str, int]:
        """解析注释"""
        pos = start
        length = len(code)
        
        # 注释从 # 开始直到行尾
        while pos < length and code[pos] != '\n':
            pos += 1
        
        return code[start:pos], pos
    
    def _parse_operator(self, code: str, start: int) -> Tuple[str, int]:
        """解析运算符（统一处理所有运算符，包括箭头）"""
        pos = start
        length = len(code)
        current_op = code[pos]
        
        # 尝试匹配更长的运算符
        while pos + 1 < length:
            next_char = code[pos + 1]
            next_type = self.get_char_type(next_char)
            
            if next_type != 'operator':
                break
            
            potential_op = current_op + next_char
            
            # 检查是否是有效的运算符
            if any(op.startswith(potential_op) for op in self.OPERATORS):
                current_op = potential_op
                pos += 1
            else:
                break
        
        return current_op, start + len(current_op)
    
    def tokenize_generator(self, code: str) -> Iterator[Tuple[str, str]]:
        """
        生成器版本的tokenize方法
        
        Args:
            code: Python代码字符串
            
        Yields:
            Tuple[str, str]: token的(类型, 值)的元组
        """
        tokens = self.tokenize(code)
        for token in tokens:
            yield token
    
    def filter_whitespace(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        过滤掉空白字符token
        
        Args:
            tokens: 原始token列表
            
        Returns:
            List[Tuple[str, str]]: 过滤后的token列表
        """
        return [(token_type, value) for token_type, value in tokens 
                if token_type != 'WHITESPACE']
    
    def filter_comments(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        过滤掉注释token
        
        Args:
            tokens: 原始token列表
            
        Returns:
            List[Tuple[str, str]]: 过滤后的token列表
        """
        return [(token_type, value) for token_type, value in tokens 
                if token_type != 'COMMENT']
    
    def filter_whitespace_and_comments(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        过滤掉空白字符和注释token
        
        Args:
            tokens: 原始token列表
            
        Returns:
            List[Tuple[str, str]]: 过滤后的token列表
        """
        return [(token_type, value) for token_type, value in tokens 
                if token_type not in ('WHITESPACE', 'COMMENT')]
    
    def pretty_print(self, tokens: List[Tuple[str, str]]) -> None:
        """
        漂亮地打印token列表
        
        Args:
            tokens: token列表
        """
        print(f"{'Type':<20} {'Value':<30}")
        print("-" * 50)
        for token_type, token_value in tokens:
            # 处理特殊字符的显示
            display_value = (token_value
                           .replace('\n', '\\n')
                           .replace('\t', '\\t')
                           .replace('\r', '\\r'))
            if len(display_value) > 28:
                display_value = display_value[:25] + "..."
            print(f"{token_type:<20} {display_value:<30}")

def tokenize(str):
    tokenizer = PythonTokenizer()
    tokens = tokenizer.tokenize(str)
    return [_t[1] for _t in tokenizer.filter_whitespace_and_comments(tokens)]

# 使用示例
if __name__ == "__main__":
    # 示例Python代码（包含注释、箭头运算符等）
    sample_code = '''
# 这是一个单行注释
def factorial(n: int) -> int:
    """计算阶乘的递归函数 -
      这是一个文档字符串"""
    if n <= 1:
        return 1  # 基础情况
    else:
        # 递归调用
        return n * factorial(n - 1)

class MathUtils:
    # 类级别的常量
    PI = 3.14159
    
    @staticmethod
    def circle_area(radius: float) -> float:
        """计算圆的面积"""
        return MathUtils.PI * radius ** 2  # 面积公式

# 主程序入口
if __name__ == "__main__":
    # 测试代码
    num = 5
    result = factorial(num)
    print(f"{num}! = {result}")  # 输出结果
    
    # 计算圆面积
    area = MathUtils.circle_area(2.5)
    print(f"圆的面积: {area:.2f}")  # 格式化输出
'''

    # 创建分词器实例
    tokenizer = PythonTokenizer()
    
    # 分词
    tokens = tokenizer.tokenize(sample_code)
    
    # 打印完整结果
    print("Python分词结果 (包含所有token):")
    print("=" * 60)
    tokenizer.pretty_print(tokens)
    
    # 过滤空白字符后打印
    filtered_tokens = tokenizer.filter_whitespace(tokens)
    print("\n过滤空白字符后的结果:")
    print("=" * 60)
    tokenizer.pretty_print(filtered_tokens)
    
    # 过滤空白字符和注释后打印
    clean_tokens = tokenizer.filter_whitespace_and_comments(tokens)
    print("\n过滤空白字符和注释后的结果:")
    print("=" * 60)
    tokenizer.pretty_print(clean_tokens)
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"总token数: {len(tokens)}")
    print(f"过滤空白字符后: {len(filtered_tokens)}")
    print(f"过滤空白字符和注释后: {len(clean_tokens)}")
    
    # 详细统计
    token_counts = {}
    for token_type, _ in tokens:
        token_counts[token_type] = token_counts.get(token_type, 0) + 1
    
    print(f"\n详细统计:")
    for token_type, count in sorted(token_counts.items()):
        print(f"  {token_type}: {count}")
    
    # 特别显示注释内容
    print(f"\n注释内容:")
    for token_type, token_value in tokens:
        if token_type == 'COMMENT':
            print(f"  {token_value}")