import numpy as np
import numpy as np
import csv
from Shrink import *
from Shrink.Point import Point
from Shrink.TimeSeries import TimeSeries

class TimeSeriesReader:
    @staticmethod
    def getTimeSeries(csv_file):
        ts = []
        max_val = float("-inf")
        min_val = float("inf")

        try:
            with open(csv_file,'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    timestamp = int(row[0])
                    value = float(row[1])
                    ts.append(Point(timestamp, value))
                    max_val = max(max_val, value)
                    min_val = min(min_val, value)
        except Exception as e:
            print(e)
        ts = TimeSeries(ts, max_val - min_val)
        # ts.size = len(ts)*(4+4)
        # ts.size = os.path.getsize(csv_file)###去掉这个为默认npy的大小，本方法使用csv

        return ts