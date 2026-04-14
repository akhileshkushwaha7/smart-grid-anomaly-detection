# 🚨 LSTM Autoencoder for Time Series Anomaly Detection

## 📌 Overview

This project implements an **unsupervised anomaly detection system** using an **LSTM Autoencoder** on time-series data.

The model is trained **only on normal data** and detects anomalies based on **reconstruction error**.

---

## 🎯 Objective

- Learn normal temporal patterns
- Detect anomalies without labels
- Simulate real-world attacks
- Build full ML pipeline

---

## 📊 Dataset

- Numenta Anomaly Benchmark (NAB)
- File: nyc_taxi.csv

---

## 🧠 Methodology

### Data Preprocessing
- Convert timestamps
- Sort data
- Normalize values

### Sequence Creation
- Sliding window sequences

### Model
- LSTM Autoencoder (Encoder + Decoder)

### Training
- Train only on normal data

### Testing
- Inject anomalies
- Compute reconstruction error

---

## 🔁 Workflow

Raw Data → Normalize → Sequence → Train → Test → Error → Detect

---

## 📈 Outputs

- Training Loss Graph
- Reconstruction Error Plot
- Detected Anomalies
> **Note:** Find the plots in results/plots.
---

## ⚙️ Installation

pip install numpy pandas matplotlib scikit-learn tensorflow

---

## ▶️ Run

python main.py

---

## 🧠 Key Idea

If reconstruction error is high → anomaly

---

## 👨‍💻 Author

Akhilesh Singh Kushwaha
