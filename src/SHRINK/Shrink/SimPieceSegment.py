import math

class SimPieceSegment:
    def __init__(self, init_timestamp, a_min, a_max, b):
        self.init_timestamp = init_timestamp
        self.a_min = a_min
        self.a_max = a_max
        self.a = (a_min + a_max) / 2
        self.b = b
        self.score = 1 #outlier score

    def round_to_least_decimal(self, aMin, aMax):
        a = (aMin+aMax)/2
        # Determine the number of decimal places in aMin and aMax
        decimal_places_aMin = len(str(aMin).split('.')[1].rstrip('0')) - len(str(aMin).split('.')[1].lstrip('0'))
        decimal_places_aMax = len(str(aMax).split('.')[1].rstrip('0')) - len(str(aMax).split('.')[1].lstrip('0'))

        # Find the minimum decimal places to round 'a'
        min_decimal_places = max(decimal_places_aMin, decimal_places_aMax)+2

        # Round 'a' to the minimum decimal places
        rounded_a = round(a, min_decimal_places)
        return rounded_a
    
    def count_matching_digits(self, num1, num2):
        if(math.isinf(num1) or math.isinf(num1) ):
            return -1
        leading_num1 = str(num1).split(".")[0]
        leading_num2 = str(num1).split(".")[0]

        # str_num1 = str(num1).split(".")[1]  # 获取小数点后的数字部分
        # str_num2 = str(num2).split(".")[1]  # 获取小数点后的数字部分
        str_num1 = "{:.16f}".format(num1).split(".")[1].rstrip("0")
        str_num2 = "{:.16f}".format(num2).split(".")[1].rstrip("0")

        count = 0
        for digit1, digit2 in zip(str_num1, str_num2):
            if digit1 == digit2:
                count += 1
            else:
                break

        leading_num1 = str(num1).split(".")[0]
        leading_num2 = str(num1).split(".")[0]
        if(leading_num1!=leading_num2):
            return 0

        return count

    def truncate_to_n_decimal_places(self, number, n):
        scaled_number = number * (10 ** n)  # 将浮点数放大n位
        truncated_number = int(scaled_number) / (10 ** n)  # 将放大后的数转换回原始大小并截断小数部分
        return truncated_number

    
    # If you need getter methods similar to Java, you can use properties in Python.
    @property
    def get_init_timestamp(self):
        return self.init_timestamp

    @property
    def get_a_min(self):
        return self.a_min

    @property
    def get_a_max(self):
        return self.a_max

    @property
    def get_a(self):
        if((self.a_max * self.a_min<0)):
            self.a =  0
            return self.a
        precision = self.count_matching_digits(self.a_max, self.a_min)
        if(precision==-1):
            self.a =  0
            return self.a
        if(precision<2):precision =2
        a =  self.truncate_to_n_decimal_places((self.a_max+self.a_min)/2, precision+2 )

        if (not(a>=self.a_min and a<=self.a_max)):
            a = self.round_to_least_decimal(self.a_max, self.a_min)

        if (not(a>=self.a_min and a<=self.a_max)):
            a =  self.truncate_to_n_decimal_places((self.a_max+self.a_min)/2, precision+3 )

        self.a = a
        if (not(a>=self.a_min and a<=self.a_max)):
            self.a = (self.a_max+self.a_min)/2

        leading_num1 = str(self.a_max).split(".")[0]
        leading_num2 = str(self.a_min).split(".")[0]
        if(leading_num1!=leading_num2):
            self.a = round((self.a_max+self.a_min)/2, 1)
            return self.a 

        if(abs(self.a)<1e-4):###***************这一点需要注意会不会引起问题***************
            self.a = 0

        #print("self.a_max = ", self.a_max, "  self.a_min = ", self.a_min, "  self.a = ", self.a)
    
        return self.a

    @property
    def get_b(self):
        return self.b
    
    def show(self):
        # print("[ ",self.get_init_timestamp, ", ", self.get_a_min, ", ", self.a_max, ", ", self.b, "]")
        print("[ ",self.get_init_timestamp, ", ", self.a, ", ", self.b, "]")


if __name__=='__main__':
    # Example usage of the class
    segment = SimPieceSegment(123456789, 0.1, 0.2, 5.0)
    print(segment.get_init_timestamp)
    print(segment.get_a_min)
    print(segment.get_a_max)
    print(segment.get_a)
    print(segment.get_b)
