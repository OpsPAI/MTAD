## MTAD: Tools and Benchmark for Multivariate Time Series Anomaly Detection

This repository is a **M**ultivariate **T**ime Series **A**nomaly **D**etection toolkit named ***MTAD*** with a comprehensive benchmarking protocol and contains state-of-the-art methods with a unified and easy-to-use interface.

Multivariate time series are a group of inherently correlated time series. For example, in the area of manufacturing industry and Information Technology (IT) systems, an entity (e.g., a physical machine or software service) is generally equipped with a monitoring mechanism to ensure its security or reliability.

In contrast to anomaly detection on single time series, extensive recent studies indicate that dependency hidden in MTS is of great importance for accurate anomaly detection, namely, the anomaly detector should consider the MTS as a whole. To this end, state-of-the-art methods have resort to deep learning-based methods to capture the dependency for more accurate anomaly detection.

The ultimate goal of MTSAnomaly is to unify current various evaluation protocols to reveal the best performance of models proposed recently.



### Our evaluation protocol

**Threshold Selection:**

- EVT-based method
- Seaching for the optimal one

**Metrics:**

- Precision, Recall, F1-score: *how accruate a model is?*
  - with **or** without point adjustment
- Delay: *how timely can a model report an anomaly?*
- Efficiency: *how fast can a model be trained and perform anomaly detection?*



### Requirement

> cd ./requirements
>
> pip install -r \<requirement file\>

### Run our benchmark

The following is an example to run benchmark for LSTM, whose configuration files are stored in the `./benchmark_config/`folder.

```
cd benchmark
python lstm_benchmark.py
```

### Models integrated in this tool

**General Machine Learning-based Models**

| Model   | Paper reference                                              |
| :------ | :----------------------------------------------------------- |
| PCA     | **[2003]** Shyu M L, Chen S C, Sarinnapakorn K, et al. A novel anomaly detection scheme based on principal component classifier |
| iForest | **[ICDM'2008]** Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou: Isolation Forest |
| LODA    | **[Machine Learning'2016]** Tomás Pevný. Loda**:** Lightweight online detector of anomalies |

**Deep Learning-based Models**

| Model       | Paper reference                                              |
| :---------- | :----------------------------------------------------------- |
| AE          | **[AAAI'2019]** Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, Luca Benini. Anomaly Detection Using Autoencoders in High Performance Computing Systems |
| LSTM        | **[KDD'2018]** Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, Tom Söderström. Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding |
| LSTM-VAE    | **[Arxiv'2017]** A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder |
| DAGMM       | **[ICLR'2018]** Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Dae-ki Cho, Haifeng Chen. Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection |
| MSCRED      | **[AAAI'19]** Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, Nitesh V. Chawla. A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data. |
| OmniAnomaly | **[KDD'2019]** Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, Dan Pei. Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network |
| MTAD-GAT | **[ICDM'2020]** Multivariate Time-series Anomaly Detection via Graph Attention Networks |
| USAD | **[KDD'2020]** USAD: UnSupervised Anomaly Detection on Multivariate Time Series. |
| InterFusion | **[KDD'2021]** Zhihan Li, Youjian Zhao, Jiaqi Han, Ya Su, Rui Jiao, Xidao Wen, Dan Pei. Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding |
| TranAD | **[VLDB'2021]** TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data |
| RANSynCoders | **[KDD'2021]** Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization |
| AnomalyTransformer | **[ICLR'2022]** Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy |
| GANF | **[ICLR'2022]** Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series |



### Datasets 

The following datasets are kindly released by different institutions or schools. Raw datasets could be downloaded or applied from the link right behind the dataset names. The processed datasets can be found [here](https://drive.google.com/drive/folders/1NEGyB4y8CvUB8TX2Wh83Eas_QHtufGPR?usp=sharing)⬇️ (SMD, SMAP, and MSL).

- Server Machine Datase (**SMD**) [Download raw datasets⬇️](https://github.com/NetManAIOps/OmniAnomaly.git)

  > Collected from a large Internet company containing a 5-week-long monitoring KPIs of 28 machines. The meaning for each KPI could be found [here](https://github.com/NetManAIOps/OmniAnomaly/issues/22).

- Soil Moisture Active Passive satellite (**SMAP**) and Mars Science Laboratory rovel (**MSL**) [Download raw datasets⬇️](link)

  > They are collected from the running spacecraft and contain a set of telemetry anomalies corresponding to actual spacecraft issues involving various subsystems and channel types.

- Secure Water Treatment (**WADI**) [Apply here\*](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

  >WADI is collected from a real-world industrial water treatment plant, which contains 11-day-long multivariate KPIs. Particularly, the system is in a normal state in the first seven days and is under attack in the following four days.

- Water Distribution (**SWAT**) [Apply here\*](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

  > An extended dataset of SWAT. 14-day-long operation KPIs are collected when the system is running normally and 2-day-long KPIs are obtained when the system is in attack scenarios.

\* WADI and SWAT datasets were released by iTrust, which should be individually applied. One can request the raw datasets and preprocess them with our preprocessing scripts.

