class TimeSeries:
    def __init__(self, data, range):
        self.data = data #Points
        self.range = range
        self.size = len(data) * (4 + 4)

    def length(self):
        return len(self.data)
