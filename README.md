# FENERBAHCE UNIVERSITY GROUP12// Anomaly-Detection-and-Machine-Learning-Based-Classification-on-CTU-13-Botnet-Traffic


**Course:** Data Mining for Cybersecurity  
**University:** Fenerbahce University  
**Group:** Group 12  

**Students:**
- Nisa Aksoy – 230304047 
- Nilsu Bülbül – 230304055  
 

---

## Project Overview

This project focuses on analyzing network traffic data from the **CTU-13 Botnet Dataset**, a widely used benchmark dataset in cybersecurity research.  
The main goal is to detect and classify malicious botnet traffic using data mining and machine learning techniques.

This study is conducted as part of the **Data Mining for Cybersecurity** course (Phase 2).

---

## Dataset

- **Dataset Name:** CTU-13 Botnet Dataset  
- **Source:** Stratosphere IPS – Czech Technical University  
- **Description:**  
  The CTU-13 dataset contains real botnet traffic mixed with normal and background network traffic.  
  Each network flow is labeled, enabling supervised learning approaches for botnet detection.

---

## Methodology

The project consists of the following steps:

### 1. Data Pre-processing
- Loading raw network flow data
- Handling missing values
- Feature selection
- Stratified sampling to preserve class imbalance

### 2. Exploratory Data Analysis (EDA)
- Class distribution analysis
- Statistical summary of numeric features
- Visualizations including histograms, boxplots, and correlation heatmaps

### 3. Data Mining Techniques
The following classification algorithms are applied:
- **Logistic Regression**
- **Random Forest Classifier**

Target classes:
- **0:** Normal traffic  
- **1:** Botnet traffic  

---

## Results

- Logistic Regression is used as a baseline classifier.
- Random Forest significantly improves detection of botnet traffic.
- Model performance is evaluated using:
  - Confusion Matrix
  - Precision
  - Recall
  - F1-score
  - Accuracy
-Random Forest outperformed Logistic Regression in detecting minority (botnet) class due to its ability to handle non-linear feature interactions and class imbalance.

---

## Project Structure
CTU_13-Project/
│
├── data/
│ └── raw/ # Raw CTU-13 dataset files
│
├── src/
│ └── load_data.py # Main script (preprocessing, EDA, modeling)
│
├── figures/ # Generated plots and visualizations
│
├── report/ # Project report (PDF)
│
└── README.md


---

## How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/nilsaki/Anomaly-Detection-and-Machine-Learning-Based-Classification-on-CTU-13-Botnet-Traffic.git


2.Install required libraries and run main script:
pip install -r requirements.txt
python src/load_data.py



## Requirements / Environment
-Python 3.10+ (tested on 3.12)

## Data
-Raw CTU-13 file should be placed in: data/raw/
-A small sample is provided for demo: data/sample/ctu13_sample_10k.csv

## Outputs
-Figures are saved under: figures/

---

## References

García, S., Grill, M., Stiborek, J., & Zunino, A. (2014).
An empirical comparison of botnet detection methods.

CTU-13 Dataset: https://www.stratosphereips.org/datasets-ctu13

