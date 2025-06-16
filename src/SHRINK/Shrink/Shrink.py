import math
import io
import os
import sys
import csv
import numpy as np
from io import BytesIO
from typing import List
from decimal import Decimal
import time
from Shrink.FloatEncoder import FloatEncoder
from Shrink.UIntEncoder import UIntEncoder
from Shrink.VariableByteEncoder import VariableByteEncoder
from Shrink.Point import Point
from Shrink.SimPieceSegment import SimPieceSegment
from Shrink.utilFunction import *
from Shrink.lof import LOF
from sklearn.neighbors import LocalOutlierFactor
sys.path.append('/home/guoyou/Lossless')
import QuanTRC





class Shrink:
    def __init__(self, points=None, epsilon=None, window = None, bytes=None, variable_byte=False, zstd=False):
        """
        初始化，内含压缩和解压

        Args:
            points: List[Point]
            epsilon:  ts.range * epsilonPct(0.05)
            bytes: bytes=binary

        Returns:
            
        """
        if points is not None: # Handle the case where points is a list of Points
            start_time = time.time()

            if not points:
                raise ValueError("No points provided")
            self.alpha = 0.01  ### use L=0.01* length instead
            self.epsilon = epsilon


            self.lastTimeStamp = points[-1].timestamp
            self.values = [point.value for point in points]
            self.max, self.min = max(self.values), min(self.values)
            self.length = len(points)
            self.lengthofSegments = None
            self.window = window
            # self.window =  len(points)
            self.buflength = int(self.alpha * self.length * self.epsilon) 
            if(window!=None):
                if(self.buflength<self.window):
                    self.buflength=self.window
                elif(self.buflength>0.01*self.length):
                    self.buflength = int(0.01*self.length)
            
            # self.buflength = int(self.length)
            # print(f"self.buflength = {self.buflength}, window = {self.window}")
            # print(f"self.buflength = {self.buflength}, window = {self.window}, self.epsilon = {self.epsilon}")


            self.segments = self.mergePerB(self.compress(points))
            self.points = points[:]

            end_time = time.time()
            self.baseTime = int((end_time - start_time) * 1000)
        elif bytes is not None: # Handle the case where bytes is a byte array
            self.readByteArray(bytes, variable_byte, zstd)
        else:
            raise ValueError("Either points or bytes must be provided")

        
    def getResiduals(self):
        self.segments.sort(key=lambda segment: segment.get_init_timestamp)
        residuals = []
        expectedVals = []
        ExpectedPoints = []
        idx = 0
        currentTimeStamp = self.segments[0].get_init_timestamp

        for i in range(len(self.segments) - 1):
            while currentTimeStamp < self.segments[i + 1].get_init_timestamp:
                expectedValue = highPrecisionAdd( self.segments[i].get_a * (currentTimeStamp - self.segments[i].get_init_timestamp), self.segments[i].get_b )
                expectedVals.append(expectedValue)
                residualVal = highPrecisionsubtract(self.values[idx], expectedValue)
                residuals.append(residualVal)
                ExpectedPoints.append(Point(currentTimeStamp, expectedValue))
                currentTimeStamp += 1
                idx += 1

        while currentTimeStamp <= self.lastTimeStamp:
                expectedValue = highPrecisionAdd( self.segments[-1].get_a * (currentTimeStamp - self.segments[-1].get_init_timestamp), self.segments[-1].get_b )
                expectedVals.append(expectedValue)
                residuals.append(highPrecisionsubtract(self.values[idx], expectedValue))
                ExpectedPoints.append(Point(currentTimeStamp, expectedValue))
                currentTimeStamp += 1
                idx += 1    

        csv_file_path = "/home/guoyou/ExtractSemantic/residuals/resdiauls.csv"
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for item in residuals:
                csv_writer.writerow([item])

        return residuals

    
    def residualEncode(self, residuals, epsilon):
        start_time = time.time()
        if(epsilon!=0):
            QuantiresdiaulsVals = [round((v/epsilon)) for v in residuals]
            # QuantiresdiaulsVals = ResQuantize(residuals, epsilon)
        else:
            # multiplier = 10 ** self.multiplier  # 乘以的倍数
            # QuantiresdiaulsVals = (np.array(residuals) * multiplier).astype(np.int32)
            QuantiresdiaulsVals = residuals[:]
        end_time = time.time()
        residualTime = int((end_time - start_time) * 1000)

        # InFilePath = '/home/guoyou/ExtractSemantic/residuals/QuantiresdiaulsVal.csv'
        outputPath = '/home/guoyou/ExtractSemantic/residuals'
        InFilePath = "/home/guoyou/ExtractSemantic/residuals/QuantiresdiaulsVal.csv"
        with open(InFilePath, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for item in QuantiresdiaulsVals:
                csv_writer.writerow([item])

        start_time = time.time()
        QuanTRC.compress(InFilePath,outputPath)
        end_time = time.time()
        residualTime += int((end_time - start_time) * 1000)
        self.residualTime  = residualTime
        residualSize = os.path.getsize('/home/guoyou/ExtractSemantic/residuals/codes.rc')

        return residualSize
    
    def residualDecode(self, outputPath, epsilon):
        start_time = time.time()
        QuanTRC.decompress(outputPath+ '/codes'+'.rc', outputPath+"/Neworginal"+'.csv')
        Dequant_val = []
        with open( outputPath+"/Neworginal"+'.csv', mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)     
            for row in csv_reader:
                Dequant_val.append(float(row[0]))

        if(epsilon!=0):
            Dequant_val = deResQuantize(Dequant_val, epsilon)
        
        end_time = time.time()
        decompResTime = int((end_time - start_time) * 1000)

        csv_file_path = "/home/guoyou/ExtractSemantic/residuals/DeQuanresdiauls.csv"
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for item in Dequant_val:
                csv_writer.writerow([item])

        return Dequant_val,decompResTime 
        
        
    def AdaptiveMerge(self, points=None, epsilon=None, bytes=None, variable_byte=False, zstd=False):
        """
        动态查看compression ratio变化

        Args:
            points: 时序数据的list，内包含point
            epsilon： error threhold, ts.range * epsilonPct
            bytes：

        Returns:
            list，内包含各个compression ratio
        """
        if not points:
            raise ValueError("No points provided")
        
        self.epsilon = epsilon
        self.lastTimeStamp = points[-1].timestamp

        self.segments = None
        Allsegments = []
        CR = []

        currentIdx, preIdx = 0, 0
        partitionPoint = 90000
        
        while(currentIdx < len(points)):
            if(currentIdx-preIdx>=partitionPoint):
                binary = self.toByteArray(variableByte=False, zstd=False)
                compressedSize = len(binary)
                CR.append(((currentIdx-0) * (4 + 4)) / compressedSize)
                #print(f"Compression Ratio: {((currentIdx-0) * (4 + 4)) / compressedSize:.3f}")
                Allsegments = self.mergePerB(Allsegments)
                self.segments = Allsegments[:]
                preIdx = currentIdx
            segments = []        
            currentIdx = self.createSegment(currentIdx, points, segments)
            Allsegments.extend(segments)
            self.segments = Allsegments[:]
        
        Allsegments = self.mergePerB(Allsegments)
        self.segments = Allsegments[:]
        binary = self.toByteArray(variableByte=False, zstd=False)
        compressedSize = len(binary)

        print(f"Compression Ratio: {(len(points)  * (4 + 4)) / compressedSize:.3f}")
        CR.append((len(points) * (4 + 4)) / compressedSize)

        return CR
    
    """    
    def ResQuantize(self, x, epsilon):
        return [round((v/epsilon)) for v in x]

    def deResQuantize(self, x_quant, epsilon):
        return [v*epsilon for v in x_quant]
    """
        

    def dynamicEpsilon(self, startIdx, points):
        # if(buflength <100):
        #     buflength = 100
        # elif(buflength>0.1*self.length):
        #     buflength = int(  0.01*self.length)
        # buflength = 1000
        # buflength = 300 # Dataset MGAB 
        # buflength = self.length # Dataset SEd 
        # buflength = 100 # Dataset Genesis
        # buflength = 500 # Dataset SensorScope
        # buflength = int(self.alpha * self.length * self.epsilon) 
        # if(buflength < self.WindowSize):
        #     buflength = self.WindowSize

        if(self.buflength >= len(points)):
            return self.epsilon, len(points)-1
        buf = []
        for i in range(startIdx, startIdx + self.buflength):
            if(i>=len(points)):
                break
            buf.append(points[i].value)

        local_max = max(buf)
        local_min = min(buf)
        C = round(math.exp((2/3 - (local_max - local_min)/ (self.max-self.min))), 3) # truncate to avoid zero
        localepsilon = round(self.epsilon * C, 3)
        # print("buflength = ", buflength, "localepsilon  = ", localepsilon)

        return localepsilon, startIdx + self.buflength

    
    def quantization(self, value, localEpsilon):
        res = round(value / localEpsilon ) * localEpsilon
        return res
    
    def createSegment(self, startIdx, points, segments, localEpsilon):
        initTimestamp = points[startIdx].timestamp
        # b = self.quantization(points[startIdx].value, localEpsilon)
        b = self.quantization(points[startIdx].value, self.epsilon) ### for outlier detection
        # if(b>=10):
        #     b=int(b)
        # elif (b>=1):
        #     b=round(b, 1)
        # else: # (b>=0.01)
        #     b=round(b, 2)
        # b=round(b, 1)
        b=round(b, 2)
        #b = round(b, getPrecision(self.epsilon))
        #print("localEpsilon = ", localEpsilon, "  b = ", b, "v = ", points[startIdx].value)
        
        if startIdx + 1 == len(points): # Case1: only 1 ponit
            segments.append(SimPieceSegment(initTimestamp, -math.inf, math.inf, b))
            segment = SimPieceSegment(initTimestamp, -math.inf, math.inf, b)
            # print(f"{startIdx + 1}:")
            return startIdx + 1, segment
        
        aMax = ((points[startIdx + 1].value + localEpsilon) - b) / (points[startIdx + 1].timestamp - initTimestamp)
        aMin = ((points[startIdx + 1].value - localEpsilon) - b) / (points[startIdx + 1].timestamp - initTimestamp)
        if startIdx + 2 == len(points): # Case2: only 2 ponits
            segments.append(SimPieceSegment(initTimestamp, aMin, aMax, b))
            segment = SimPieceSegment(initTimestamp, aMin, aMax, b)
            # print(f"{startIdx + 2}:")
            return startIdx + 2, segment

        
        for idx in range(startIdx + 2, len(points)): # Case3: more than 2 ponits
            upValue = points[idx].value + localEpsilon
            downValue = points[idx].value - localEpsilon

            up_lim = aMax * (points[idx].timestamp - initTimestamp) + b
            down_lim = aMin * (points[idx].timestamp - initTimestamp) + b

            if (downValue > up_lim or upValue < down_lim):
                #***print("segment: ",initTimestamp, idx-1, (aMin + aMax) / 2, b)
                #print("amax = ", aMax, ",  amin = ", aMin)
                segments.append(SimPieceSegment(initTimestamp, aMin , aMax, b))
                segment = SimPieceSegment(initTimestamp, aMin, aMax, b)
                # print(f"{idx}:")
                return idx, segment

            

            if upValue < up_lim:
                aMax = max((upValue - b) / (points[idx].timestamp - initTimestamp), aMin)
            if downValue > down_lim:
                aMin = min((downValue - b) / (points[idx].timestamp - initTimestamp), aMax)

        segment = SimPieceSegment(initTimestamp, aMin, aMax, b)
        segments.append(segment)

        # print(f"{idx}:")
        # for seg in segments:
        #     print("segment: ",seg.init_timestamp,  seg.a_min,  seg.a_max, seg.b)

        return len(points), segment

    

            
    def compress(self, points):
        segments = []
        currentIdx = 0
        newIdx = -1
        localEpsilon = self.epsilon

        while(currentIdx < len(points)):
            if(currentIdx>newIdx):
                localEpsilon, newIdx  = self.dynamicEpsilon(currentIdx, points)
            # print(f"From {currentIdx} to ", end=" ")
            currentIdx, segment = self.createSegment(currentIdx, points, segments, localEpsilon)
        # print(f"\t buflength = {self.buflength}")

            
        return segments
    


        
    def mergePerB(self, segments):
        aMinTemp = float('-inf')
        aMaxTemp = float('inf')
        b = float('nan')
        timestamps = []
        mergedSegments = []
        self.lengthofSegments = 0

        segments.sort(key=lambda segment: (segment.get_b, segment.get_a))
        
        for i in range(len(segments)):
            if b != segments[i].get_b:
                if len(timestamps) == 1:
                    mergedSegments.append(SimPieceSegment(timestamps[0], aMinTemp, aMaxTemp, b))
                    self.lengthofSegments += 1
                else:
                    for timestamp in timestamps:
                        mergedSegments.append(SimPieceSegment(timestamp, aMinTemp, aMaxTemp, b))
                        self.lengthofSegments += 1

                
                timestamps.clear()
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = segments[i].get_a_min
                aMaxTemp = segments[i].get_a_max
                b = segments[i].get_b
                continue
            
            if segments[i].get_a_min <= aMaxTemp and segments[i].get_a_max >= aMinTemp:
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = max(aMinTemp, segments[i].get_a_min)
                aMaxTemp = min(aMaxTemp, segments[i].get_a_max)
            else:
                if len(timestamps) == 1:
                    mergedSegments.append(segments[i - 1])
                    self.lengthofSegments += 1
                else:
                    for timestamp in timestamps:
                        mergedSegments.append(SimPieceSegment(timestamp, aMinTemp, aMaxTemp, b))
                        self.lengthofSegments += 1
                
                timestamps.clear()
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = segments[i].get_a_min
                aMaxTemp = segments[i].get_a_max
        
        if timestamps:
            if len(timestamps) == 1:
                mergedSegments.append(SimPieceSegment(timestamps[0], aMinTemp, aMaxTemp, b))
                self.lengthofSegments += 1

            else:
                for timestamp in timestamps:
                    mergedSegments.append(SimPieceSegment(timestamp, aMinTemp, aMaxTemp, b))
                    self.lengthofSegments += 1

        # print("************************************************")
        # for i, s in enumerate(segments):
        #     if (i<100):
        #         print(f"{s.get_init_timestamp} \t {s.get_a} \t {s.get_b}")
        # print("************************************************")
        return mergedSegments
    

    def decompress(self):
        start_time = time.time()
        
        # Pre-calculate the initial timestamps and other values
        init_timestamps = [segment.get_init_timestamp for segment in self.segments]
        a_values = [segment.a for segment in self.segments]
        b_values = [segment.get_b for segment in self.segments]
        points = []

        # Loop over segments, avoiding method calls within the loop
        for i in range(len(self.segments) - 1):
            # Calculate the range of timestamps for the current segment
            timestamps = range(init_timestamps[i], init_timestamps[i + 1])
            # Use a list comprehension to generate points
            points += [Point(ts, a_values[i] * (ts - init_timestamps[i]) + b_values[i]) for ts in timestamps]

        # Handle the last segment
        last_segment_timestamps = range(init_timestamps[-1], self.lastTimeStamp + 1)
        points += [Point(ts, a_values[-1] * (ts - init_timestamps[-1]) + b_values[-1]) for ts in last_segment_timestamps]
        
        # with open("/home/guoyou/ExtractSemantic/Base/BasePoints.csv", 'w', encoding='utf-8') as file:
        #     for p in points:
        #         file.write(f"{p.timestamp}, {p.value}\n")

        end_time = time.time()
        decompBaseTime = int((end_time - start_time) * 1000)

        return points, decompBaseTime
    



        
    def toByteArrayPerBSegments(self, segments: List[SimPieceSegment], variableByte: bool, outStream: io.BytesIO) -> None:
        # Initialize a dictionary to organize segments by 'b' value
        input = {}
        resArr = []

        for segment in segments:
            a = segment.get_a
            #print("amax = ",segment.a_max, ",  amin = ", segment.a_min, ",  a = ", a)
            b = round(segment.get_b / self.epsilon)
            t = segment.get_init_timestamp
            
            if b not in input:
                input[b] = {}
            
            if a not in input[b]:
                input[b][a] = []
            
            input[b][a].append(t)
        
        # Write the size of the dictionary
        VariableByteEncoder.write(len(input), outStream)
        resArr.append(len(input))###****需要删除****###
        
        if not input.items():
            return
        
        previousB = min(input.keys())
        VariableByteEncoder.write(previousB, outStream)
        resArr.append(previousB)###****需要删除****###
        
        for b, aSegments in input.items():
            VariableByteEncoder.write(b - previousB, outStream)
            resArr.append(b - previousB)###****需要删除****###
            previousB = b
            VariableByteEncoder.write(len(aSegments), outStream)
            resArr.append(len(aSegments))###****需要删除****###

            
            for a, timestamps in aSegments.items():
                # Custom method to encode the float 'a' value
                FloatEncoder.write(float(a), outStream)
                resArr.append(float(a))###****需要删除****###
                len(aSegments)
                
                if variableByte:
                    print("variableByte为True了，出现错误\n")
                    timestamps.sort()
                
                VariableByteEncoder.write(len(timestamps), outStream)
                resArr.append(len(timestamps))###****需要删除****###

                
                previousTS = 0
                
                for timestamp in timestamps:
                    if variableByte:
                        print("variableByte为True了,出现错误\n")
                        VariableByteEncoder.write(timestamp - previousTS, outStream)
                        resArr.append(timestamp - previousTS)###****需要删除****###
                    else:
                        # Custom method to write 'timestamp' as an unsigned int
                        UIntEncoder.write(timestamp, outStream)
                        resArr.append(timestamp)###****需要删除****###

                        
                    
                    previousTS = timestamp
        # np_array = np.array(resArr, dtype=np.float32)
        # np.save('/home/guoyou/ExtractSemantic/Base/'+" Base_Watch_accelerometer.npy", np_array)
        filename = 'Base.csv'

        with open('/home/guoyou/ExtractSemantic/Base/'+filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(resArr)
        
    def toByteArray(self, variableByte: bool, zstd: bool) -> bytes:
        outStream = BytesIO()
        bytes = None

        FloatEncoder.write(float(self.epsilon), outStream)

        self.toByteArrayPerBSegments(self.segments, variableByte, outStream)

        if variableByte:
            VariableByteEncoder.write(int(self.lastTimeStamp), outStream)
        else:
            UIntEncoder.write(self.lastTimeStamp, outStream)

        if zstd:
            bytes = zstd.compress(outStream.getvalue())
        else:
            bytes = outStream.getvalue()

        outStream.close()
        return bytes
    
    def saveByte(self, byts, filename):
        
        path = '/home/guoyou/ExtractSemantic/Base/'+filename[:-7]+"_Base.bin"
        with open(path, 'wb') as file:
            file.write(byts)

        baseSize = os.path.getsize(path)
        return baseSize

        
    def readMergedPerBSegments(self, variableByte, inStream):
        segments = []
        numB = VariableByteEncoder.read(inStream)

        if numB == 0:
            return segments

        previousB = VariableByteEncoder.read(inStream)

        for _ in range(numB):
            b = VariableByteEncoder.read(inStream) + previousB
            previousB = b
            numA = VariableByteEncoder.read(inStream)

            for _ in range(numA):
                a = FloatEncoder.read(inStream)
                numTimestamps = VariableByteEncoder.read(inStream)

                for _ in range(numTimestamps):
                    if variableByte:
                        timestamp += VariableByteEncoder.read(inStream)
                    else:
                        timestamp = UIntEncoder.read(inStream)
                    segments.append(SimPieceSegment(timestamp, a, a, b * self.epsilon))

        return segments

    def readByteArray(self, input, variableByte, zstd):
        if zstd:
            binary = zstd.decompress(input)
        else:
            binary = input

        inStream = BytesIO(binary)

        self.epsilon = FloatEncoder.read(inStream)
        self.segments = self.readMergedPerBSegments(variableByte, inStream)

        if variableByte:
            print("variableByte为True了，出现错误\n")
            self.lastTimeStamp = VariableByteEncoder.read(inStream)
        else:
            self.lastTimeStamp = UIntEncoder.read(inStream)

        inStream.close()




    