import io

class VariableByteEncoder:
    @staticmethod
    def extract7bits(i, val):
        return (val >> (7 * i)) & ((1 << 7) - 1)
    
    @staticmethod
    def extract7bitsmaskless(i, val):
        result = (val >> (7 * i)) & 0xFF  # 确保结果在 0-255 范围内
        if result & 0x80:  # 如果最高位是1，表示负数，进行符号扩展
            result = result - 256  # 使用有符号位的右移运算符进行符号扩展
        return result
    
    @staticmethod
    def write(number, output_stream):
        if number < 0:
            number = ((-number) << 1) | 1  # 使用Zigzag编码处理负数
        else:
            number = number << 1  # 处理正数
        
        while number >= 128:
            output_stream.write(bytes([number & 0x7F | 0x80]))
            number >>= 7
        
        output_stream.write(bytes([number]))  # 对正数只设置最低位为0，不需要再设置最高位为1
    
    @staticmethod
    def read(input_stream):
        number = 0
        shift = 0
        
        while True:
            inp = input_stream.read(1)
            if not inp:
                break
            
            inp = inp[0]
            number |= (inp & 0x7F) << shift
            shift += 7
            
            if not inp & 0x80:
                break
        
        if number & 1:
            number = -(number >> 1)  # 还原负数
        else:
            number = number >> 1
        
        return number
    


def test_variable_byte_encoder():
    import random
    # 生成随机测试数据
    random_numbers = [random.randint(-1000000, 1000000) for _ in range(10)]

    for number in random_numbers:
        output_stream = io.BytesIO()
        VariableByteEncoder.write(number, output_stream)

        input_stream = io.BytesIO(output_stream.getvalue())
        decoded_number = VariableByteEncoder.read(input_stream)

        print(f"Original number: {number}")
        print(f"Decoded number: {decoded_number}")
        print(f"Is equal: {number == decoded_number}")
        print()

def test_encoding_length():
    numbers = [42, 128, -1024, 65535, 1000000]

    for number in numbers:
        output_stream = io.BytesIO()
        VariableByteEncoder.write(number, output_stream)
        encoded_bytes = output_stream.getvalue()

        print(f"Original number: {number}")
        print(f"Encoded bytes: {encoded_bytes}")
        print(f"Encoded length: {len(encoded_bytes)} bytes")
        print()

if __name__=='__main__':
    test_encoding_length()

