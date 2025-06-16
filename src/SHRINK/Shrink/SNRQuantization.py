import numpy as np
import math
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema


class TimeSeriesEncoder:
    def __init__(self, default_snr=30, acf_base=3, acf_nlags=400, min_window=3, max_window=300, default_window=100):
        self.default_snr = default_snr
        self.acf_base = acf_base
        self.acf_nlags = acf_nlags
        self.min_window = min_window
        self.max_window = max_window
        self.default_window = default_window

    def encode(self, data, snr=None, beta=None, estimate_window=False):
        """
        Parameters
        ----------
        data : np.ndarray
            One-dimensional time series array
        snr : float, optional
            Target signal-to-noise ratio
        beta : int, optional
            Quantization step exponent (if no SNR is specified)
        estimate_window : bool
            Whether to estimate a window size based on ACF

        Returns
        -------
        eps : float
            Quantization precision
        current_snr : float
            Final SNR after quantization
        quantized_data : np.ndarray
            Quantized version of the input data
        window_size : int, optional
            (Only if estimate_window=True) Estimated sliding window length
        """
        if snr is None and beta is None:
            snr = self.default_snr

        if snr is not None:
            eps, current_snr, quantized_data = self.quantize(data, snr)
        else:
            eps = 2 ** beta
            quantized_data = np.round(data / eps) * eps
            current_snr = self.compute_snr(data, quantized_data)

        if estimate_window:
            # print(f"Data shape = {data.shape}")
            window = self.estimate_window_size(data)
            return eps, current_snr, quantized_data, window
        else:
            return eps, current_snr, quantized_data

    def estimate_window_size(self, data):
        """
        Estimate an appropriate sliding window size from autocorrelation.
        """
        # if len(data.shape) > 1:
        #     return self.default_window
        # data = data[:min(20000, len(data))]

        # base = self.acf_base
        # auto_corr = acf(data, nlags=self.acf_nlags, fft=True)[base:]
        # local_max = argrelextrema(auto_corr, np.greater)[0]

        # try:
        #     max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        #     lag = local_max[max_local_max]
        #     if lag < self.min_window or lag > self.max_window:
        #         return self.default_window
        #     return lag + base
        # except:
        #     return self.default_window

        if len(data.shape)>1:
            return 0
        data = data[:min(20000, len(data))]
        
        base = 3
        auto_corr = acf(data, nlags=400, fft=True)[base:]
        
        
        local_max = argrelextrema(auto_corr, np.greater)[0]
        try:
            max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
            if local_max[max_local_max]<3 or local_max[max_local_max]>300:
                return 100
            return local_max[max_local_max]+base
        except:
            return 100
            

    def quantize(self, a, target_snr):
        beta = self.init_beta(a)
        pre_snr = float('inf')
        iteration, max_iterations = 0, 50

        while iteration < max_iterations:
            iteration += 1
            count, current_snr = self.quantize_with_beta(a, beta)
            if current_snr < target_snr:
                if pre_snr != float('inf'):
                    beta -= 1
                    _, current_snr = self.quantize_with_beta(a, beta)
                    eps = 2 ** beta
                    break
                else:
                    beta -= 2
                    _, current_snr = self.quantize_with_beta(a, beta)
                    eps = 2 ** beta
                    break
            beta += 1
            pre_snr = current_snr

        return eps, current_snr, np.round(a / eps) * eps

    def init_beta(self, a):
        power = np.mean(a ** 2) * 1e-6
        return int(math.floor(0.5 * math.log2(power)) + 1)

    def quantize_with_beta(self, a, beta):
        eps = 2 ** beta
        signal_power = np.sum(a ** 2)
        noise_power = np.sum((a - np.round(a / eps) * eps) ** 2)
        snr = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        return len(a), snr

    def compute_snr(self, signal, quantized):
        signal_power = np.sum(signal ** 2)
        noise_power = np.sum((signal - quantized) ** 2)
        return 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')




















# import numpy as np
# import math
# from statsmodels.tsa.stattools import acf
# from scipy.signal import argrelextrema


# class TimeSeriesEncoder:
#     def __init__(self, default_snr=30, acf_base=3, acf_nlags=400, min_window=3, max_window=300, default_window=100):
#         self.default_snr = default_snr
#         self.acf_base = acf_base
#         self.acf_nlags = acf_nlags
#         self.min_window = min_window
#         self.max_window = max_window
#         self.default_window = default_window

#     def encode(self, data, snr=None, beta=None, estimate_window=False):
#         """
#         Parameters
#         ----------
#         data : np.ndarray
#             One-dimensional time series array
#         snr : float, optional
#             Target signal-to-noise ratio
#         beta : int, optional
#             Quantization step exponent (if no SNR is specified)
#         estimate_window : bool
#             Whether to estimate a window size based on ACF

#         Returns
#         -------
#         eps : float
#             Quantization precision
#         current_snr : float
#             Final SNR after quantization
#         quantized_data : np.ndarray
#             Quantized version of the input data
#         window_size : int, optional
#             (Only if estimate_window=True) Estimated sliding window length
#         """
#         if snr is None and beta is None:
#             snr = self.default_snr

#         if snr is not None:
#             eps, current_snr, quantized_data = self.quantize(data, snr)
#         else:
#             eps = 2 ** beta
#             quantized_data = np.round(data / eps) * eps
#             current_snr = self.compute_snr(data, quantized_data)

#         # if eps >= data.max() - data.min():
#         #     eps = 0.001 * (data.max() - data.min())

#         if estimate_window:
#             # print(f"Data shape = {data.shape}")
#             window = self.estimate_window_size(data)
#             return eps, current_snr, quantized_data, window
#         else:
#             return eps, current_snr, quantized_data

#     def estimate_window_size(self, data):
#         """
#         Estimate an appropriate sliding window size from autocorrelation.
#         """
#         # if len(data.shape) > 1:
#         #     return self.default_window
#         # data = data[:min(20000, len(data))]

#         # base = self.acf_base
#         # auto_corr = acf(data, nlags=self.acf_nlags, fft=True)[base:]
#         # local_max = argrelextrema(auto_corr, np.greater)[0]

#         # try:
#         #     max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
#         #     lag = local_max[max_local_max]
#         #     if lag < self.min_window or lag > self.max_window:
#         #         return self.default_window
#         #     return lag + base
#         # except:
#         #     return self.default_window

#         if len(data.shape)>1:
#             return 0
#         data = data[:min(20000, len(data))]
        
#         base = 3
#         auto_corr = acf(data, nlags=400, fft=True)[base:]
        
        
#         local_max = argrelextrema(auto_corr, np.greater)[0]
#         try:
#             max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
#             if local_max[max_local_max]<3 or local_max[max_local_max]>300:
#                 return 100
#             return local_max[max_local_max]+base
#         except:
#             return 100
            

#     def quantize(self, a, target_snr):
#         beta = self.init_beta(a)
#         pre_snr = float('inf')
#         iteration, max_iterations = 0, 50

#         while iteration < max_iterations:
#             iteration += 1
#             count, current_snr = self.quantize_with_beta(a, beta)
#             if current_snr < target_snr:
#                 beta -= 1
#                 _, current_snr = self.quantize_with_beta(a, beta)
#                 eps = 2 ** beta
#                 break
#             beta += 1
#             pre_snr = current_snr

#         return eps, current_snr, np.round(a / eps) * eps

#     def init_beta(self, a):
#         power = np.mean(a ** 2) * 1e-6
#         return int(math.floor(0.5 * math.log2(power)) + 1)

#     def quantize_with_beta(self, a, beta):
#         eps = 2 ** beta
#         signal_power = np.sum(a ** 2)
#         noise_power = np.sum((a - np.round(a / eps) * eps) ** 2)
#         snr = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
#         return len(a), snr

#     def compute_snr(self, signal, quantized):
#         signal_power = np.sum(signal ** 2)
#         noise_power = np.sum((signal - quantized) ** 2)
#         return 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    






    






# import numpy as np
# import pandas as pd
# import math

# def Encoding(data, beta=None, snr=None):
#     if snr is None:
#         eps = 2 ** beta
#         quantized_data = np.round(data / eps) * eps
#     else:
#         eps, current_snr, quantized_data = quantize(data, snr)

#     return eps, current_snr, quantized_data

# def quantize(a, target_snr):
#     beta = init_beta(a)
#     # print(f"init_beta: {beta}")
#     pre_snr = float('inf')
#     iteration, max_iterations = 0, 50  # 设置最大迭代次数，防止死循环
#     while iteration<max_iterations:
#         iteration += 1
#         count, current_snr = quantize_with_beta(a, beta)
#         # print(f"current_snr = {current_snr}")
#         if current_snr < target_snr:
#             if(pre_snr!=float('inf')):
#                 beta -= 1
#                 _, current_snr = quantize_with_beta(a, beta)
#                 eps = 2 ** beta
#                 break
#             else:
#                 beta -= 2
#                 _, current_snr = quantize_with_beta(a, beta)
#                 eps = 2 ** beta
#                 break

#         beta += 1
#         pre_snr = current_snr

#     return eps,current_snr,np.round(a / eps) * eps

# def init_beta(a):
#     power = np.mean(a ** 2) * 1e-6
#     return int(math.floor(0.5 * math.log2(power)) + 1)


# def quantize_with_beta(a, beta):
#     eps = 2 ** beta
#     signal_power = np.sum(a ** 2)
#     noise_power = 0
#     for value in a:
#         quantized_value = round(value / eps) * eps
#         error = value - quantized_value
#         noise_power += error ** 2
#     snr = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
#     # print(f"beta: {beta}, noise_power: {noise_power}, snr: {noise_power}, ", end="")
#     return len(a), snr

# if __name__ =="__main__":

#     path =  "/home/guoyou/OutlierDetection/TSB-UAD/data/MGAB/"
#     filenames = ["1.test.out"]
#     data = pd.read_csv(path+filenames[0], header=None)
#     values = data[0].values
#     print(values)
#     target_snr, beta = 25, 4 
#     eps, current_snr,quantized_data = Encoding(values, beta=beta, snr=target_snr)
#     print(f"\ntarget_snr = {target_snr}, eps = {eps}, current_snr = {current_snr}")


















# import numpy as np 
# import pandas as pd
# import math

# def Encoding(data, beta=None, snr=None):
#     if snr is None:
#         eps = 2 ** beta
#         quantized_data = np.round(data / eps) * eps
#     else:
#         eps, current_snr, quantized_data = quantize(data, snr)

#     return eps, current_snr, quantized_data

# def quantize(a, target_snr):
#     beta = init_beta(a)
#     while True:
#         count, current_snr = quantize_with_beta(a, beta)
#         if current_snr < target_snr:
#             break
#         beta += 1
#     beta -= 1
#     _, current_snr = quantize_with_beta(a, beta)
#     eps = 2 ** beta
#     return eps,current_snr,np.round(a / eps) * eps

# def init_beta(a):
#     power = np.mean(a ** 2) * 1e-6
#     return int(math.floor(0.5 * math.log2(power)) + 1)

# def quantize_with_beta(a, beta):
#     eps = 2 ** beta
#     signal_power = np.sum(a ** 2)
#     noise_power = 0
#     for value in a:
#         quantized_value = round(value / eps) * eps
#         error = value - quantized_value
#         noise_power += error ** 2
#     snr = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
#     return len(a), snr

# # if __name__ =="__main__":
# #     import pandas as pd

# #     path =  "/home/guoyou/OutlierDetection/TSB-UAD/data/MGAB/"
# #     filenames = ["1.test.out"]
# #     data = pd.read_csv(path+filenames[0], header=None)
# #     values = data[0].values
    
# #     encoder = TimeSeriesEncoder(default_snr=25, max_window=3000)
# #     # print(f"Inital SNR = {snr}")
# #     epsilon, snr, _ , window = encoder.encode(data, snr=25, estimate_window=False)

# #     print(f"\ntarget_snr = {25}, eps = {epsilon}, current_snr = {snr}")

# if __name__ =="__main__":

#     path = "/home/guoyou/OutlierDetection/TSB-UAD/data/MGAB/"
#     filenames = ["1.test.out"]

#     data = pd.read_csv(path+filenames[0], header=None)
#     values = data[1].values
#     beta = 4 
#     target_snr = 25
#     eps, current_snr,quantized_data = Encoding(values, beta=beta, snr=target_snr)
#     print(f"target_snr = {target_snr}, eps = {eps}, current_snr = {current_snr}")
#     # print(eps)
#     # print(current_snr)
#     # print(data)
#     # print(quantized_data)