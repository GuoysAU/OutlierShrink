# Towards Highly Efficient Semantic Communication via Semantic-aware Compression on Time Series Data


## Overview
Welcome to the repository accompanying our paper "Towards Highly Efficient Semantic Communication via Semantic-aware Compression on Time Series Data". We propose a novel lightweight method to support semantic communication through direct analytics on time series data compressed without decompression via the SHRINK compression. By incorporating semantic quantization and transformations, our method extends SHRINK to align with the principles of semantic communication, enabling task-oriented analytics directly on compressed data. 


## Dataset files
- See Datasets folder


## Prerequisites
- Python version 3.9.18
- All required packages are installed (by command `pip install -r requirements.txt`)
- Windows, Linux or macOS system


## Usage
To use the model, download the data, change directory to this project code, execute the command:
- `python main.py` (for testing outlier detection results for uncompressed as well as compressed data)
- `python AnomalyD_Comp.py` (for testing outlier detection results for compressed data)
- `python industry_demo.py` (for testing outlier detection results for compressed data)
