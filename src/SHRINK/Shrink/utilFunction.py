import matplotlib.pyplot as plt
import numpy as np
import math
from decimal import Decimal


def plot_line_graph(x ,y, title="Two Line Graphs on One Plot", color="blue"):
    """
    绘制折线图。
    参数:
    x, y: 对应x轴和y轴
    """
    # 提取时间戳和值用于绘图

    # 绘制折线图（不包含点）
    plt.figure(figsize=(10, 6))
    #plt.plot(x, y)
    plt.plot(x, y, marker='o', color = color)
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.grid(True)

    # Annotate each point with its value
    for i, value in enumerate(y):
        plt.text(x[i], y[i], f"{value:.4f}", ha='center', va='bottom')

    plt.show()

def plot_simpleline( points=None, title="Two Line Graphs on One Plot", color="blue" ):
    """
    绘制折线图。
    参数:
    x, y: 对应x轴和y轴
    """
    # 提取时间戳和值用于绘图

    timestamps = [point.timestamp for point in points]
    values = [point.value for point in points]

    # 绘制折线图（不包含点）
    plt.figure(figsize=(10, 6))
    #plt.plot(x, y)
    plt.plot(timestamps, values, color=color)
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.grid(False)
    plt.tick_params(axis='x', which='both', labelbottom=False)


    plt.show()


def plot_two_line_graphs(points1, points2,start=0, end=100, title="Two Line Graphs on One Plot"):
    """
    在一幅图上绘制两个折线图。
    参数:
    points1 (list of Point): 第一个折线图的Point对象列表
    points2 (list of Point): 第二个折线图的Point对象列表
    """
    # 提取第一个折线图的时间戳和值
    timestamps1 = [point.timestamp for point in points1]
    values1 = [point.value for point in points1]

    # 提取第二个折线图的时间戳和值
    timestamps2 = [point.timestamp for point in points2]
    values2 = [point.value for point in points2]

    # 绘制两个折线图
    plt.figure(figsize=(14, 10))
    plt.plot(timestamps1[:end], values1[:end], label='Original data', color='red')
    plt.plot(timestamps2[:end], values2[:end], label='Approximated data', color='green')
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

        # Annotate each point with its value
    """  
    for i, value in enumerate(values2):
        plt.text(timestamps2[i], values2[i], f"{value:.1f}", ha='center', va='bottom')  
    """
    plt.show()


def piecewise_constant_approximation(data, num_segments=1):
    # Calculate the length of each segment
    segment_length = len(data) // num_segments

    # Pre-allocate the array for performance
    approximated_data = np.zeros_like(data)
    # Compute the approximation
    for i in range(num_segments):
        start_index = i * segment_length
        # Handle the last segment which might have different length
        end_index = len(data) if i == num_segments - 1 else start_index + segment_length
        # Compute the mean of the segment
        segment_mean = np.mean(data[start_index:end_index])
        # Assign the mean to the segment's data points
        approximated_data[start_index:end_index] = segment_mean

    return approximated_data

def generated_list(low=-1, high=1, size=10):
    # Generate a list of floats with a mean of 0 and a uniform distribution between -2 and 2
    generated_list = np.random.uniform(low, high, size)

    # Adjust the list to have a mean of 0
    adjusted_list = generated_list - np.mean(generated_list)

    # Ensure that the maximum and minimum values are within the specified range
    adjusted_list = np.clip(adjusted_list, low, high)

    return adjusted_list

def float_to_int(float_array, errorThre=1):
    # Convert the input to a numpy array if it's not already one
    float_array = np.array(float_array)
    # Multiply by 20, round, and then divide by 20 to ensure the error is within 0.05
    int_array = np.round(float_array / errorThre).astype(int)
    return int_array

def int_to_float(int_array, errorThre=1):
    # Convert back to float
    float_array = int_array * errorThre
    return float_array

def CR_When_int(int_array):
    # Original size in bits (16 bits per int)
    original_size = len(int_array) * 32

    # Calculate the size after compression (sum of the bit lengths of each int's binary representation)
    # For the number 0, we count 1 bit.
    compressed_size = sum(len(bin(abs(i)).lstrip('-0b')) if i != 0 else 1 for i in int_array)

    # Calculate the compression ratio
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0

    return compression_ratio

def totalbits_When_int(int_array):
    # Calculate the total number of bits after conversion (account for the number 0 as 1 bit)
    total_compressed_bits = sum(len(bin(abs(i)).lstrip('-0b')) if i != 0 else 1 for i in int_array)
    return total_compressed_bits

def QuantizeValues(datas, error):
    maxval  = max(datas)
    minval = min(datas)
    return [round((v-minval)/(maxval-minval)/error) for v in datas], maxval, minval

def DeQuantizeValues(QuanDatas, max, min, error):
    return [v*(max-min)*error+min for v in QuanDatas]

def ResQuantize( x, epsilon):
    res = [round((v/epsilon)) for v in x]
    return res

def deResQuantize(x_quant, epsilon):
    return [v*epsilon for v in x_quant]


def highPrecisionAdd(x,y):
    return float(Decimal(str(x )) +  Decimal(str(y)))

def highPrecisionsubtract(x,y):
    return float(Decimal(str(x )) -  Decimal(str(y)))

def getPrecision(num):
    return  len(str(num).split(".")[1])


if __name__ ==  '__main__':
    """    
        generatedList = generated_list(-1.145, 1.145, 20)
        print("Original List = ", generatedList)
        intList = float_to_int(generatedList, errorThre=0.05)
        print("Transformed int list = ", float_to_int(intList))
        approxiList = int_to_float( intList , errorThre=0.05)
        print("Float list = ", approxiList)
        print("error is = ",generatedList-approxiList )
    """
    """
        generatedList = generated_list(-1.145, 1.145, 200000)
        intList = float_to_int(generatedList, errorThre=0.05)

        print("Compression ration after converting float to int = ", CR_When_int(intList))
    """
    #print(totalbits_When_int([0,1,2]))

    generatedList = generated_list(-1.145, 1.145, 5)
    print(generatedList)
    QuanDatas, maxval, minval = QuantizeValues(generatedList, 0.05)
    print(QuanDatas)
    deQuandata = DeQuantizeValues(QuanDatas, maxval, minval, 0.05)
    print(deQuandata)


