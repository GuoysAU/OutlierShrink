import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.utils.visualisation import plotFig
from sklearn.preprocessing import MinMaxScaler
# from TSB_UAD.models.series2graph import Series2Graph
# from TSB_UAD.models.norma import NORMA
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.matrix_profile import MatrixProfile
# from TSB_UAD.models.pca import PCA
# from TSB_UAD.models.poly import POLY
# from TSB_UAD.models.ocsvm import OCSVM
# from TSB_UAD.models.lstm import lstm
# from TSB_UAD.models.AE import AE_MLP2
# from TSB_UAD.models.cnn import cnn
# from TSB_UAD.models.damp import DAMP
from TSB_UAD.models.sand import SAND
from TSB_UAD.vus.metrics import get_metrics
from typing import List
import gzip
import sys
import os
import time
import csv
import unittest
import sys
from datetime import datetime, timedelta
sys.path.append('/home/guoyou')
sys.path.append('/home/guoyou/SHRINK')
from SHRINK.Shrink.TimeSeriesReader import TimeSeriesReader
from SHRINK.Shrink.Shrink import Shrink
from SHRINK.Shrink.Point import Point
from SHRINK.Shrink.utilFunction import *
from SHRINK.Shrink.SNRQuantization import *
from SHRINK.Shrink.Transform import Transform
from SHRINK.Shrink.Transform import DeTransform
import QuanTRC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误信息
from series2graph import Series2Graph
from norma import *
from typing import Optional
import re


def compute_entropy_norm(values, num_bins=64):
    """
    Compute normalised Shannon entropy for time series data.
    """
    hist, _ = np.histogram(values, bins=num_bins, density=True)
    p = hist / np.sum(hist)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    entropy_max = np.log2(num_bins)
    return entropy / entropy_max if entropy_max > 0 else 0

def estimate_snr_from_file(filepath, value_col=0, label_col=1, snr_min=25, snr_max=35, num_bins=64):
    """
    Estimate SNR from a CSV file with value and label columns.
    """
    df = pd.read_csv(filepath, header=None)
    values = df.iloc[:, value_col].values
    entropy_norm = compute_entropy_norm(values, num_bins)
    snr = snr_min + (snr_max - snr_min) * entropy_norm
    return int(snr)



# 1. Util function
def compress(path: str, filename: str, snr: int) -> pd.DataFrame:
    """
    Compresses a time series file using the SHRINK algorithm and extracts representative data points.
    
    Parameters
    ----------
    path : str
        The directory path containing the input file.
    filename : str
        The name of the input CSV file to compress.
    snr : float, optional
        Signal-to-noise ratio used to control compression precision (default is 25.0).
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with representative data points, including 'index' and 'value' columns.
    """
    
    # Step 1: Preprocess input data
    raw_df = pd.read_csv(os.path.join(path, filename), header=None).dropna()
    processed_df = raw_df[[0]].copy()
    processed_df.insert(0, 'index', range(len(processed_df)))
    
    modified_path = os.path.join(path, "Compress", f"Modified_{filename[:-4]}.csv")
    processed_df.to_csv(modified_path, index=False, header=False)

    # data_for_encoding = pd.read_csv(modified_path, header=None)

    df = raw_df.to_numpy()
    max_length = df.shape[0]
    data = df[:max_length,0].astype(float)

    encoder = TimeSeriesEncoder(default_snr=25, max_window=3000)
    epsilon, snr, _ , window = encoder.encode(data, snr=snr, estimate_window=True)

    # print(f"\n epsilon = {epsilon}, snr={snr}")
    # print(f"window = {window}")


    # epsilon, current_snr, _ = Encoding(data_for_encoding[1].values, snr=snr)
    # print(f"\t snr = {snr},  epsilon = { epsilon}, window = {window}")

    # Step 2: Compress with SHRINK
    ts = TimeSeriesReader.getTimeSeries(modified_path)
    ts.size = os.path.getsize(os.path.join(path, filename))

    shrink = Shrink(points=ts.data, epsilon=epsilon, window = window)
    compressed_binary = shrink.toByteArray(variableByte=False, zstd=False)
    base_size = shrink.saveByte(compressed_binary, filename)

    # Step 3: Extract representative points for downstream tasks
    representatives = Transform(shrink)
    indices, values = DeTransform(representatives)
    
    compressed_df = pd.DataFrame({
        'index': indices,
        'value': values
    })

    # Step 4: (Optional) Print compression performance for debugging
    # print(f"{filename}: {ts.size / 1024 / 1024:.2f} MB")
    # print("Base Size:", base_size / 1024 / 1024, "MB")
    # print("Compression Ratio:", base_size / ts.size)

    return compressed_df


def PaperFig(start, end, df_temp, df_comp):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 17,
    })

    values_temp = df_temp[0].iloc[start:end]
    labels_temp = df_temp[1].iloc[start:end]
    full_index = values_temp.index

    comp_filtered = df_comp[(df_comp['index'] >= start) & (df_comp['index'] < end)]
    comp_index = comp_filtered['index']
    comp_values = comp_filtered['value']

    # Set seaborn style
    sns.set_style("whitegrid")
    palette = sns.color_palette('muted', 4)
    colour_normal = palette[0]    # Soft blue
    colour_outlier = palette[3]   # Soft red/orange
    colour_compress = palette[2]  # Soft green

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- Top: Original Data ---
    ax = axes[0]
    labels_for_temp = labels_temp.values

    normal_idx = full_index[labels_for_temp == 0]
    normal_vals = values_temp[labels_for_temp == 0]
    ax.scatter(normal_idx, normal_vals, color=colour_normal, s=10, label="Time series")

    outlier_idx = full_index[labels_for_temp == 1]
    outlier_vals = values_temp[labels_for_temp == 1]
    ax.scatter(outlier_idx, outlier_vals, color=colour_outlier, s=10, marker='s', label="Outliers")

    ax.set_ylabel("Original Value", fontsize=18)
    ax.legend()
    ax.grid(False)

    # Add (a) text
    ax.text(0.01, 0.95, r'$(a)$', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

    # --- Bottom: Compressed Data ---
    ax = axes[1]
    labels_for_comp = df_temp[1].iloc[comp_index].values

    normal_idx = comp_index[labels_for_comp == 0]
    normal_vals = comp_values[labels_for_comp == 0]
    ax.scatter(normal_idx, normal_vals, color=colour_normal, s=10, label="Time series")

    outlier_idx = comp_index[labels_for_comp == 1]
    outlier_vals = comp_values[labels_for_comp == 1]
    ax.scatter(outlier_idx, outlier_vals, color=colour_outlier, s=10, marker='s', label="Outliers")

    # ax.set_xlabel("Index", fontsize=18)
    ax.set_ylabel("Compressed Value", fontsize=18)
    ax.legend()
    ax.grid(False)

    # Add (b) text
    ax.text(0.01, 0.95, r'$(b)$', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

    # --- Set the same y-axis limits ---
    all_values = np.concatenate([values_temp.values, comp_values.values])
    ymin, ymax = all_values.min(), all_values.max()
    margin = (ymax - ymin) * 0.05  # 5% margin
    for ax in axes:
        ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    pdf_filename = "/home/guoyou/OrigVsComp.pdf"
    plt.savefig(pdf_filename, format='pdf')
    plt.show()


def showFig(start, end, df_temp, df_comp,compress=False):
    values_temp = df_temp[0].iloc[start:end]
    labels_temp = df_temp[1].iloc[start:end]

    comp_filtered = df_comp[(df_comp['index'] >= start) & (df_comp['index'] < end)]
    comp_index = comp_filtered['index']
    comp_values = comp_filtered['value']

    import seaborn as sns
    palette = sns.color_palette()

    plt.figure(figsize=(12, 8))
    # plt.scatter(values_temp[labels_temp == 0].index, values_temp[labels_temp == 0], color=palette[0], label='Normal (Original)', s=5)
    # plt.scatter(values_temp[labels_temp == 1].index, values_temp[labels_temp == 1], color=palette[2], label='Outlier (Original)', s=10)
    plt.scatter(values_temp[labels_temp == 0].index, values_temp[labels_temp == 0], color=palette[0], s=5)
    plt.scatter(values_temp[labels_temp == 1].index, values_temp[labels_temp == 1], color=palette[3], s=10 )
    if(compress):
        plt.scatter(comp_index, comp_values, color='green', label='Sampled Data (df_comp)', s=10, marker='x')

    # plt.title(f"Original Data with Tranformed Data Highlighted (Index {start}-{end})")
    plt.grid(False)
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    plt.legend()
    # pdf_filename = "/home/guoyou/DaphnetDataset.pdf"
    # plt.savefig(pdf_filename, format='pdf')
    plt.show()

# def showFig(start, end, df_temp, df_comp, compress=False):
#     # Set global font sizes
#     plt.rcParams.update({
#         'font.size': 18,            # Base font size
#         'axes.titlesize': 20,       # Title
#         'axes.labelsize': 18,       # X and Y labels
#         'xtick.labelsize': 16,      # Tick labels
#         'ytick.labelsize': 16,
#         'legend.fontsize': 17,      # Legend
#     })

#     import seaborn as sns
#     values_temp = df_temp[0].iloc[start:end]
#     labels_temp = df_temp[1].iloc[start:end]
#     full_index = values_temp.index

#     comp_filtered = df_comp[(df_comp['index'] >= start) & (df_comp['index'] < end)]
#     comp_index = comp_filtered['index']
#     comp_values = comp_filtered['value']

#     # Use seaborn 'muted' palette
#     sns.set_style("whitegrid")
#     palette = sns.color_palette('muted', 4)
#     colour_normal = palette[0]    # e.g. muted blue
#     colour_outlier = palette[1]   # e.g. muted red/orange
#     colour_compress = palette[2]  # e.g. muted green

#     plt.figure(figsize=(12, 8))

#     # Plot all values as a line
#     plt.plot(full_index, values_temp.values, color=colour_normal, label='Time Series', linewidth=1)

#     # Outlier logic
#     outlier_mask = (labels_temp == 1).values
#     outlier_indices = np.where(outlier_mask)[0]

#     if len(outlier_indices) > 0:
#         split_points = np.where(np.diff(outlier_indices) != 1)[0] + 1
#         segments = np.split(outlier_indices, split_points)

#         for segment in segments:
#             seg_idx = values_temp.iloc[segment].index
#             seg_vals = values_temp.iloc[segment].values
#             if len(segment) == 1:
#                 plt.scatter(seg_idx, seg_vals, color='red', label='Outliers', s=10)
#             else:
#                 plt.plot(seg_idx, seg_vals, color='red', linewidth=1, label='Outliers')

#     # Compressed line
#     if compress:
#         plt.scatter(comp_index, comp_values, color='green', label='Sampled Data (df_comp)', s=5, marker='x')
#         # plt.plot(comp_index, comp_values, color='olive', linestyle='--', linewidth=3, label='Sampled Data (df_comp)')

#     # plt.title(f"Original Data with Transformed Data Highlighted (Index {start}-{end})")
#     # plt.title("IOPS", fontsize=25)
#     # plt.xlabel("Index")
#     # plt.ylabel("Value")

#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())

#     plt.tight_layout()
#     plt.grid(False)
#     plt.show()

def showFig2(start, end, df_temp, df_comp, compress=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 17,
    })

    values_temp = df_temp[0].iloc[start:end]
    labels_temp = df_temp[1].iloc[start:end]
    full_index = values_temp.index

    comp_filtered = df_comp[(df_comp['index'] >= start) & (df_comp['index'] < end)]
    comp_index = comp_filtered['index']
    comp_values = comp_filtered['value']

    # Seaborn muted palette
    sns.set_style("whitegrid")
    palette = sns.color_palette('muted', 4)
    colour_normal = palette[0]
    colour_outlier = palette[1]
    colour_compress = palette[2]

    plt.figure(figsize=(12, 8))

    if not compress:
        # --- Plot original full data ---
        plt.plot(full_index, values_temp.values, color=colour_normal, label='Time Series', linewidth=1)

        # Plot outliers from df_temp
        outlier_mask = (labels_temp == 1).values
        outlier_indices = np.where(outlier_mask)[0]

        if len(outlier_indices) > 0:
            split_points = np.where(np.diff(outlier_indices) != 1)[0] + 1
            segments = np.split(outlier_indices, split_points)

            first_outlier = True
            for segment in segments:
                seg_idx = values_temp.iloc[segment].index
                seg_vals = values_temp.iloc[segment].values
                if len(segment) == 1:
                    plt.scatter(seg_idx, seg_vals, color=colour_outlier, s=20,
                                label='Outliers' if first_outlier else "")
                else:
                    plt.plot(seg_idx, seg_vals, color=colour_outlier, linewidth=1,
                             label='Outliers' if first_outlier else "")
                first_outlier = False

        # Also plot compressed points if needed
        if len(comp_index) > 0:
            plt.scatter(comp_index, comp_values, color=colour_compress, label='Sampled Data (df_comp)', s=20, marker='x')

    else:
        # --- Plot compressed sampled points only ---
        # Check labels from df_temp for these points
        labels_for_comp = df_temp[1].iloc[comp_index].values

        normal_idx = comp_index[labels_for_comp == 0]
        normal_vals = comp_values[labels_for_comp == 0]

        outlier_idx = comp_index[labels_for_comp == 1]
        outlier_vals = comp_values[labels_for_comp == 1]

        if len(normal_idx) > 0:
            plt.scatter(normal_idx, normal_vals, color=colour_compress, label='Normal (Compressed)', s=20, marker='o')

        if len(outlier_idx) > 0:
            plt.scatter(outlier_idx, outlier_vals, color=colour_outlier, label='Outlier (Compressed)', s=40, marker='s')

    # Final styling
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.grid(False)
    plt.show()



def Filenames2List(folder_path):
    import os
    import re
    def natural_sort_key(file_name):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_name)]
    file_list = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    file_list.sort(key=natural_sort_key)
    return file_list


# 2. AD MODELs for compression
def IF(df, df_comp, X_data, label_selected, slidingWindow_selected, contamination=0.1, random_state=None):
    start_time = time.time()
    clf = IForest(n_jobs=1, contamination=contamination, random_state=random_state)
    x = X_data
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow_selected-1)/2) + list(score) + [score[-1]]*((slidingWindow_selected-1)//2))
    # results = get_metrics(score, label_selected, metric="all", slidingWindow=slidingWindow_selected)
    #********************************************************************************#
    original_size = df.shape[0]
    mapped_label = np.zeros(original_size, dtype=int)
    mapped_score = np.zeros(original_size, dtype=float)
    for i, index in enumerate(df_comp['index'].values):
        mapped_label[index] = label_selected[i]  # 映射 label
        mapped_score[index] = score[i]  # 映射 score
    #*************************************************************************************#
    # Safety condition: skip evaluation if no anomalies or flat score
    if np.sum(label_selected) == 0 or np.std(score) == 0:
        return (0.5, 0.0, 0.0, execution_time)  # Default: AUC=0.5 (random), AP/F=0.0
    results = get_metrics(mapped_score, mapped_label, metric="all", slidingWindow=slidingWindow_selected)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

def lof(df, df_comp, data_selected, label_selected, slidingWindow_selected,  contamination=0.1, random_state=None):
    start_time = time.time()
    clf = LOF(n_neighbors=20, n_jobs=1, contamination=contamination)
    x = data_selected
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 5)
    # execution_time = round(end_time - start_time, 4)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow_selected-1)/2) + list(score) + [score[-1]]*((slidingWindow_selected-1)//2))
    #********************************************************************************#
    original_size = df.shape[0]
    mapped_label = np.zeros(original_size, dtype=int)
    mapped_score = np.zeros(original_size, dtype=float)
    for i, index in enumerate(df_comp['index'].values):
        mapped_label[index] = label_selected[i]  # 映射 label
        mapped_score[index] = score[i]  # 映射 score
    #*************************************************************************************#
    if np.sum(label_selected) == 0 or np.std(score) == 0:
        return (0.5, 0.0, 0.0, execution_time)  # Default: AUC=0.5 (random), AP/F=0.0
    results = get_metrics(mapped_score, mapped_label, metric="all", slidingWindow=slidingWindow_selected)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

# def sand(df, df_comp, data_selected, label_selected, slidingWindow_selected, contamination=0.1):
#     # print(data_selected.shape)
#     # data_selected = data_selected.reshape(-1, 1)
#     # print(data_selected.shape)
#     start_time = time.time()
#     clf = SAND(pattern_length=slidingWindow_selected,subsequence_length=4*(slidingWindow_selected))
#     x = data_selected
#     clf.fit(x,overlaping_rate=int(1.5*slidingWindow_selected))
#     end_time = time.time()
#     execution_time = round(end_time - start_time, 2)
#     score = clf.decision_scores_
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

#     #********************************************************************************#
#     original_size = df.shape[0]
#     mapped_label = np.zeros(original_size, dtype=int)
#     mapped_score = np.zeros(original_size, dtype=float)

#     for i, index in enumerate(df_comp['index'].values):
#         mapped_label[index] = label_selected[i]  # 映射 label
#         mapped_score[index] = score[i]  # 映射 score
#     #*************************************************************************************#

#     results = get_metrics(mapped_score, mapped_label, metric="all", slidingWindow=slidingWindow_selected)
#     return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)


def sand(df, df_comp, data_selected, label_selected, slidingWindow_selected, contamination=0.1, random_state=None):
    # print(data_selected.shape)
    # data_selected = data_selected.reshape(-1, 1)
    # print(data_selected.shape)
    start_time = time.time()
    clf = SAND(pattern_length=slidingWindow_selected, subsequence_length=4*(slidingWindow_selected), random_state=random_state)
    x = data_selected
    clf.fit(x,overlaping_rate=int(1.5*slidingWindow_selected))
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    #********************************************************************************#
    original_size = df.shape[0]
    mapped_label = np.zeros(original_size, dtype=int)
    mapped_score = np.zeros(original_size, dtype=float)

    for i, index in enumerate(df_comp['index'].values):
        mapped_label[index] = label_selected[i]  # 映射 label
        mapped_score[index] = score[i]  # 映射 score
    #*************************************************************************************#
    if np.sum(label_selected) == 0 or np.std(score) == 0:
        return (0.5, 0.0, 0.0, execution_time)  # Default: AUC=0.5 (random), AP/F=0.0
    results = get_metrics(mapped_score, mapped_label, metric="all", slidingWindow=slidingWindow_selected)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

def damp(df, df_comp, X_data, label_selected, slidingWindow_selected, contamination=0.1):
    start_time = time.time()
    clf = DAMP(m = slidingWindow_selected,sp_index=slidingWindow_selected+1)
    x = X_data
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    print("shape of x and score: ", x.shape, score.shape)

    #********************************************************************************#
    original_size = df.shape[0]
    mapped_label = np.zeros(original_size, dtype=int)
    mapped_score = np.zeros(original_size, dtype=float)

    for i, index in enumerate(df_comp['index'].values):
        mapped_label[index] = label_selected[i]  # 映射 label
        mapped_score[index] = score[i]  # 映射 score
    #*************************************************************************************#

    results = get_metrics(mapped_score, mapped_label, metric="all", slidingWindow=slidingWindow_selected)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

def matrixProfile(df, df_comp, data_selected, label_selected, slidingWindow_selected, contamination):
    start_time = time.time()
    clf = MatrixProfile(window = slidingWindow_selected)
    x = data_selected
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow_selected-1)/2) + list(score) + [score[-1]]*((slidingWindow_selected-1)//2))
    #********************************************************************************#
    original_size = df.shape[0]
    mapped_label = np.zeros(original_size, dtype=int)
    mapped_score = np.zeros(original_size, dtype=float)
    for i, index in enumerate(df_comp['index'].values):
        mapped_label[index] = label_selected[i]  # 映射 label
        mapped_score[index] = score[i]  # 映射 score
    #*************************************************************************************#

    results = get_metrics(mapped_score, mapped_label, metric="all", slidingWindow=slidingWindow_selected)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

def Ser2Graph(df, df_comp, data_selected, label_selected, slidingWindow_selected, contamination):
    start_time = time.time()
    clf = Series2Graph(pattern_length=slidingWindow_selected)
    x = data_selected
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    query_length = slidingWindow_selected*2
    clf.score(query_length=query_length,dataset=data_selected)
    score = clf.decision_scores_
    score = np.array([score[0]]*math.ceil(query_length//2) + list(score) + [score[-1]]*(query_length//2))
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    #********************************************************************************#
    original_size = df.shape[0]
    mapped_label = np.zeros(original_size, dtype=int)
    mapped_score = np.zeros(original_size, dtype=float)
    for i, index in enumerate(df_comp['index'].values):
        mapped_label[index] = label_selected[i]  # 映射 label
        mapped_score[index] = score[i]  # 映射 score
    #*************************************************************************************#

    results = get_metrics(score, label_selected, metric="all", slidingWindow=slidingWindow_selected)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

# def AE(df, df_comp, data_selected, label_selected, slidingWindow_selected, contamination=0.1):
#     start_time = time.time()
#     ###***********压缩后的数据修改训练集比例***********
#     data_train = data_selected[:int(0.3*len(data_selected))]
#     data_test = data_selected
#     ###***********压缩后的数据修改训练集比例***********
#     clf = AE_MLP2(slidingWindow = slidingWindow_selected, epochs=100, verbose=0)
#     clf.fit(data_train, data_test)
#     end_time = time.time()
#     execution_time = round(end_time - start_time, 2)
#     score = clf.decision_scores_
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     results = get_metrics(score, label_selected, metric="all", slidingWindow=slidingWindow_selected)
#     return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)


# def sand(df, df_comp, data_selected, label_selected, slidingWindow_selected, contamination=0.1):
#     # print(data_selected.shape)
#     # data_selected = data_selected.reshape(-1, 1)
#     # print(data_selected.shape)
#     start_time = time.time()
#     clf = SAND(pattern_length=slidingWindow_selected,subsequence_length=4*(slidingWindow_selected))
#     x = data_selected
#     clf.fit(x,overlaping_rate=int(1.5*slidingWindow_selected))
#     end_time = time.time()
#     execution_time = round(end_time - start_time, 2)
#     score = clf.decision_scores_
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

#     #********************************************************************************#
#     original_size = df.shape[0]
#     mapped_label = np.zeros(original_size, dtype=int)
#     mapped_score = np.zeros(original_size, dtype=float)

#     for i, index in enumerate(df_comp['index'].values):
#         mapped_label[index] = label_selected[i]  # 映射 label
#         mapped_score[index] = score[i]  # 映射 score
#     #*************************************************************************************#

#     results = get_metrics(mapped_score, mapped_label, metric="all", slidingWindow=slidingWindow_selected)
#     return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)


# def IF(df, df_comp, X_data, label_selected, slidingWindow_selected, contamination=0.1):
#     start_time = time.time()
#     clf = IForest(n_jobs=1, contamination=contamination)
#     x = X_data
#     clf.fit(x)
#     end_time = time.time()
#     execution_time = round(end_time - start_time, 2)
#     score = clf.decision_scores_
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     score = np.array([score[0]]*math.ceil((slidingWindow_selected-1)/2) + list(score) + [score[-1]]*((slidingWindow_selected-1)//2))
#     # results = get_metrics(score, label_selected, metric="all", slidingWindow=slidingWindow_selected)
#     #********************************************************************************#
#     original_size = df.shape[0]
#     mapped_label = np.zeros(original_size, dtype=int)
#     mapped_score = np.zeros(original_size, dtype=float)
#     for i, index in enumerate(df_comp['index'].values):
#         mapped_label[index] = label_selected[i]  # 映射 label
#         mapped_score[index] = score[i]  # 映射 score
#     #*************************************************************************************#
#     results = get_metrics(mapped_score, mapped_label, metric="all", slidingWindow=slidingWindow_selected)
#     return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

# def lof(df, df_comp, data_selected, label_selected, slidingWindow_selected, contamination):
#     start_time = time.time()
#     clf = LOF(n_neighbors=20, n_jobs=1, contamination=contamination)
#     x = data_selected
#     clf.fit(x)
#     end_time = time.time()
#     execution_time = round(end_time - start_time, 2)
#     score = clf.decision_scores_
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     score = np.array([score[0]]*math.ceil((slidingWindow_selected-1)/2) + list(score) + [score[-1]]*((slidingWindow_selected-1)//2))
#     #********************************************************************************#
#     original_size = df.shape[0]
#     mapped_label = np.zeros(original_size, dtype=int)
#     mapped_score = np.zeros(original_size, dtype=float)
#     for i, index in enumerate(df_comp['index'].values):
#         mapped_label[index] = label_selected[i]  # 映射 label
#         mapped_score[index] = score[i]  # 映射 score
#     #*************************************************************************************#

#     results = get_metrics(mapped_score, mapped_label, metric="all", slidingWindow=slidingWindow_selected)
#     return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)


# 3. AD MODELs for Orginal
def IF_o(df, X_data, label, slidingWindow, contamination=0.1, random_state = None):
    clf = IForest(n_jobs=1, contamination=contamination, random_state = random_state) 
    x = X_data
    start_time = time.time()
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

def lof_o(df, X_data, label, slidingWindow, contamination):
    # #****************************************#
    # slidingWindow  = 1
    # #****************************************#
    start_time = time.time()
    clf = LOF(n_neighbors=20, n_jobs=1, contamination=contamination)
    x = X_data
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

def sand_o(df, data, label, slidingWindow, contamination=0.1):
    start_time = time.time()
    # clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
    # x = data
    # # print(f"shape of x: {x.shape}")
    # clf.fit(x,overlaping_rate=int(1.5*slidingWindow))

    clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
    x = data
    clf.fit(x,overlaping_rate=int(1.5*slidingWindow))
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)



def damp_o(df, data, label, slidingWindow, contamination=0.1):
    start_time = time.time()
    clf = DAMP(m = slidingWindow,sp_index=slidingWindow+1)
    x = data
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    # print("shape of x and score: ", x.shape, score.shape)


    results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

def Ser2Graph_o(df, data, label, slidingWindow, contamination=0.1):
    start_time = time.time()
    clf = Series2Graph(pattern_length=slidingWindow)
    x = data
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    query_length = slidingWindow*2
    clf.score(query_length=query_length,dataset=data)
    score = clf.decision_scores_
    score = np.array([score[0]]*math.ceil(query_length//2) + list(score) + [score[-1]]*(query_length//2))
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)


def matrixProfile_o(df, data, label, slidingWindow, contamination=0.1):
    start_time = time.time()
    clf = MatrixProfile(window = slidingWindow)
    x = data
    clf.fit(x)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    # clf.score(query_length=2*slidingWindow,dataset=x)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)

def AE_o(df, data, label, slidingWindow, contamination=0.1):
    start_time = time.time()
    ###***********压缩后的数据修改训练集比例***********
    data_train = data[:int(0.1*len(data))]
    data_test = data
    ###***********压缩后的数据修改训练集比例***********
    clf = AE_MLP2(slidingWindow = slidingWindow, epochs=100, verbose=0)
    clf.fit(data_train, data_test)
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
    return (results['AUC_ROC'], results['AUC_PR'], results['F'], execution_time)


