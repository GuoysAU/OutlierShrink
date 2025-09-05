# Highly Efficient Semantic Communication via Semantic-aware Compression


## Dataset files
Download from the following Google Drive link:

https://drive.google.com/drive/folders/1nKLL751nPLweo3aI_8Dp09RUpkEXd2YX?usp=drive_link


# Datasets details

To evaluate the effectiveness of the proposed method, we conduct a comparative study on synthetic and public datasets, encompassing **193 datasets**.

## Synthetic Datasets
Following our previous work [[1]](#1), the **UcNe collection** includes nine public datasets without ground truth:
- 5 datasets from the UCR time series repository  
- 4 datasets from the National Ecological Observatory Network (NEON)

We inject synthetic anomalies following standard practices [[2],[3]]:
- **Point outliers**: randomly inject 0.1% of time points by adding/subtracting four times the standard deviation.  
- **Sequence outliers**: inject non-overlapping segments (length 10–30 time steps), accounting for ~1.0% of total points.  

This procedure emulates sensor faults and abrupt behavioral changes commonly encountered in IoT systems.  

## Public Datasets with Ground Truth

- **SED** [[4]](#4): Simulated spinning disks in NASA’s Rotor Dynamics Lab, quasi-periodic with subtle deviations.  
- **Genesis** [[5]](#5): Industrial pick-and-place with pneumatic actuators, capturing air pressure/control anomalies.  
- **ECG** [[6]](#6): Ambulatory ECG recording, anomalies = premature ventricular contractions.  
- **MGAB** [[7]](#7): Mackey-Glass equations, chaotic nonlinear series for distinguishing noise vs. anomalies.  
- **SensorScope** [[8]](#8): Multi-modal environmental data (temperature, humidity, solar radiation) from wireless sensors.  
- **MITDB** [[9]](#9): MIT-BIH Arrhythmia Database, 48 recordings, heterogeneous arrhythmias/noise.  
- **Daphnet** [[10]](#10): Tri-axial accelerometer data from Parkinson’s patients; anomalies = freezing-of-gait.  
- **IOPS** [[11]](#11): Server-level performance metrics from cloud infrastructure (AIOps challenge).  

---

## References
1. Sun et al., SHRINK: Data Compression by Semantic Extraction and Residuals Encoding, 2024.  
2. Blázquez-García et al., A review on outlier/anomaly detection in time series data, 2021.  
3. Wenig et al., TimeEval, 2022.  
4. Abdul et al., SED dataset, 2012.  
5. Von et al., Genesis dataset, 2018.  
6. Moody et al., MIT-BIH ECG, 2001.  
7. Thill et al., MGAB dataset, 2020.  
8. Yao et al., SensorScope, 2010.  
9. Moody et al., MITDB dataset, 1992.  
10. Bachlin et al., Daphnet dataset, 2009.  
11. Ren et al., IOPS dataset, 2019.  

