# ⚛️🤖🧠  HQ-FHLRE: Hybrid-Quantum Federated Hydrogen Leak Recognition Engine 🖥️🌐

This repository contains experiments from my Industrial Ph.D. project in Computational Intelligence at UniNa (scholarship funded by ENEA), focused on anomaly detection in hydrogen transport networks.

We simulate a hydrogen pipeline system using **Simscape**, collecting multivariate time series from **four pressure sensors**. The dataset includes:
- A **normal operating scenario**, in which the pressure stabilizes after an initial transient.
- **Three anomalous scenarios**: local restrictions, valve closures, and compressor failures.

Anomalies manifest as subtle, asynchronous perturbations across the sensor time series—making this a challenging **multivariate anomaly detection** task.



**QFADH** (Quantum Federated Anomaly Detection for Hydrogen) is the experimental framework in which we apply:
1. **Local training** of anomaly detection models on each simulated scenario.
2. **Federated learning**, where models trained locally share only parameters—not data—preserving privacy and enabling **decentralized intelligence**.

### ✨ Models used (deployed both locally and in federated setting):
- **LSTM Autoencoder** – classical deep learning baseline.
- **QLSTM** – LSTM augmented with hybrid quantum-classical gates.
- **QTLSTM** – Classical LSTM trained using parameters generated from quantum circuits (Quantum Train approach).

---

This pipeline allows us to compare classical and quantum-enhanced architectures, first in an isolated (local) setting, then within a federated learning framework, providing insight into:
- How quantum models perform on sparse, multivariate anomalies.
- How federated learning impacts generalization and robustness.

<div align="center">

  <img src="./enea/quantum lst.png" alt="Quantum LSTM" width="45%" style="display:inline-block;">
  <img src="./enea/trasporto-idrogeno.jpg" alt="Hydrogen Network Station" width="45% style="display:inline-block;">
  <br><br>
  
  <img src="./enea/federated-gif.gif" alt="Federated Learning Animation" width="60%">

</div>

---

## 📁 Project Structure

```QFAD/
QFAD/
├── enea/
│   ├── __pycache__/
│   ├── anomaly_detection_classifiers/
│   │   ├── __init__.py
│   │   ├── anomaly_detection_classifier.py
│   ├── data_extraction/
│   │   └── selected_features.xlsx
│   ├── jupyter/
│   │   └── anomaly_detection_enea.ipynb
│   ├── time_series_plot/
│   │   ├── time_series_0.png
│   │   ├── time_series_1.png
│   │   ├── time_series_2.png
│   │   └── time_series_3.png
│   ├── utils/
│   │   ├── __init__.py
│   │   └── utils_data.py
│   ├── anomaly_detection_classifier.cpython-310.pyc
│   ├── utils_data.cpython-310.pyc
│   ├── anomaly_detection_hydrogen_network_pressure_sensors.py
├── federated-gif.gif
├── quantum lst.png
├── trasporto-idrogeno.jpg
├── output.txt
└── README.md
``` 
---

## 🧠 Models Overview

- **LSTM Autoencoder**: Classical baseline.
- **QLSTM**: LSTM with hybrid classical-quantum architecture.
- **QTLSTM**: Quantum train protocols for Classical LSTM.
---

## 🚀 Getting Started

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```



