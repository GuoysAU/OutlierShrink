import io
import struct
from io import BytesIO


class FloatEncoder:
    @staticmethod
    def write(number, output_stream):
        int_bits = struct.pack('>f', number)
        output_stream.write(int_bits)
        return int_bits

    @staticmethod
    def read(input_stream):
        int_bits = input_stream.read(4)
        number = struct.unpack('>f', int_bits)[0]
        return number
    
if __name__ ==  '__main__':
    outStream = BytesIO()
    print(FloatEncoder.write(1,outStream ))#10,4bytes
    print(len(FloatEncoder.write(0,outStream )))#10,4bytes
