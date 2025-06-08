# ⚛️🤖🧠  HQ-FHLRE: Hybrid-Quantum Federated Hydrogen Leak Recognition Engine 🖥️🌐

This repository contains the core research activities of my Industrial Ph.D. project in Computational Intelligence at the University of Naples Federico II (UniNa), funded by ENEA. This work represents the **primary objective** of my doctoral scholarship and is focused on the development of intelligent systems for anomaly analysis in hydrogen transport networks.

The research is structured in three main stages:

1. **Anomaly Detection** – identifying abnormal patterns in multivariate time series from pressure sensors across the hydrogen network.
2. **Anomaly Classification** – categorizing the type of anomaly (e.g., valve closure, compressor failure, local restriction).
3. **Anomaly Localization** – determining the precise location of the fault using **Deep Reinforcement Learning (DRL)**.

Each stage is initially tackled using **classical machine learning and deep learning techniques**, and then extended to the **quantum domain** using hybrid approaches. In particular:
- **Hybrid Quantum-Classical Neural Networks** are applied for detection and classification.
- **Variational Quantum Deep Reinforcement Learning (VQDRL)** strategies are explored for the localization task, leveraging quantum circuits to model complex decision-making policies.

The hydrogen transport system is simulated using **Simscape**, and the dataset includes both normal operation and multiple anomalous scenarios. Federated learning is also employed to ensure **data privacy** and **decentralized intelligence**, reflecting real-world industrial constraints.

This repository reflects the full methodological evolution of the project—from classical to quantum intelligence and stands as the **central deliverable** of my Ph.D. research.

We simulate a hydrogen pipeline system using **Simscape**, collecting multivariate time series from **four pressure sensors**. The dataset includes:
- A **normal operating scenario**, in which the pressure stabilizes after an initial transient.
- **Two anomalous scenarios**: local restrictions (valve closures) and compressor failures.

Anomalies manifest as subtle, asynchronous perturbations across the sensor time series—making this a challenging **multivariate anomaly detection** task.

**HQ-FHLRE** (Hybrid-Quantum Federated Hydrogen Leak Recognition Engine) is the experimental framework in which we apply:
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
  
  <img src="./enea_h2net_v2/hydrogen-station.jpg" alt="Hydrogen Network Station" width="45%">
  <img src="./enea_h2net_v2/quantum lst.png" alt="Quantum LSTM" width="45%">
  <br><br>
  
  <img src="./enea_h2net_v2/federated-gif.gif" alt="Federated Learning Animation" width="60%">

</div>

## 🧪 Dataset & Experimental Versions

The project includes two distinct simulation phases, each represented by a separate folder:

- **`enea_h2net_v1/`**  
  This folder contains early-stage experiments based on a simplified hydrogen network model simulated in **Simscape**. At this stage, anomaly detection was performed using traditional machine learning techniques such as **Isolation Forest**, **One-Class SVM**, and basic **statistical thresholding**.

- **`enea_h2net_v2/`**  
  As the initial model proved limited in complexity and realism, we developed a more detailed and representative simulation of the hydrogen transport network. In this phase, we introduced advanced deep learning architectures such as **LSTM**, **QLSTM**, and **QTLSTM** applied both in standalone settings and within a **federated learning** framework to better capture temporal dependencies and asynchronous multivariate anomalies.

This evolution reflects the methodological progression of the project: from classical ML on simplified systems to hybrid quantum-classical deep learning in more realistic, complex environments.

---

## 📁 Project Structure

```QFAD/
QFAD/
├── enea_h2net_v1/
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
├── enea_h2net_v2
    ├── ...
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



