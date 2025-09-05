# Towards Highly Efficient Semantic Communication via Semantic-aware Compression on Time Series Data


## Overview
Welcome to the repository accompanying our paper "Towards Highly Efficient Semantic Communication via Semantic-aware Compression on Time Series Data". We propose a novel lightweight method to support semantic communication through direct analytics on time series data compressed without decompression via the SHRINK compression. By incorporating semantic quantization and transformations, our method extends SHRINK to align with the principles of semantic communication, enabling task-oriented analytics directly on compressed data. 


## Dataset files
- See Datasets folder


## Prerequisites
- Python version 3.9.18
- Create a new Python virtual environment
- Install the dependencies via pip install -r requirements.txt
- Windows, Linux or macOS system


## Usage
To use the model, download the data, change directory accordingly to this project code, execute the command:
- `python main.py` (for testing outlier detection results for uncompressed as well as compressed data)
- `python AnomalyD_Comp.py` (for testing outlier detection results for compressed data)
- `python AnomalyD_Orig.py` (for testing outlier detection results for compressed data)
- `Experiment.ipynb` (an example showing call detection methods as well as the detection results)


## Summary of Major Experimental Findings.
In terms of semantic edge analytics, we have the following observations for outlier detection task:

- `IForest`:
 demonstrates consistently strong performance in both detection accuracy and runtime across datasets. Using semantics, IForest-Sem achieves the highest ROC AUC, PR AUC, and F1 scores on nearly all datasets, while further reducing runtime with an average speedup of 5$\times$  speedup on average. Our method makes it particularly well-suited for lightweight outlier detection at edge. 

- `LOF`:
known for detecting nuanced local density anomalies, suffers from high computational cost and low accuracy on large datasets. Interestingly, using semantic, LOF-Sem often achieves the shortest runtime and significantly improved detection performance across datasets, demonstrating how our method facilitates computationally intensive models to perform efficiently. %However, IForest remains the most consistently balanced in terms of both speed and accuracy. 

- `SAND`：
designed for streaming and sequence-based outlier detection, conceptually aligns with semantic communication and edge analytics. However, its high computational overhead due to overlapping subsequence clustering limits its suitability for real-time or low-power edge devices. While semantics data significantly reduces SAND’s runtime, it remains less efficient than the other two counterparts, and may require further optimization to meet strict edge deployment constraints.
