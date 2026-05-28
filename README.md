# 🏭🚨🧯🛠️  H2-SimNet: A Preliminary Simulation-Based Testbed for Anomaly Detection in Hydrogen Transport Networks 🧠🤖💻⚛️

This repository provides the simulation framework and supervised baseline models developed in support of the paper “H2-SimNet: A Preliminary Simulation-Based Testbed for Anomaly Detection in Hydrogen Transport Networks”, which addresses the technical and safety challenges of transporting hydrogen through existing gas pipeline infrastructure.

## 🌍 Motivation

Hydrogen is a zero-emission energy carrier with the potential to decarbonize critical sectors such as transportation, power generation, and heavy industry. However, adapting existing **natural gas networks** for hydrogen or hydrogen blends introduces significant challenges:

- Hydrogen molecules are **smaller and lighter** than methane, leading to increased permeability and higher risk of **leaks**, especially at **joints, valves, and welds**.
- **Hydrogen embrittlement** weakens metallic components, increasing the likelihood of failure.
- Conventional sensors and instruments (e.g., flow meters, pressure gauges) calibrated for natural gas may provide **inaccurate readings** with hydrogen blends.
- **Compression and pressure reduction stations** may not handle hydrogen’s distinct thermodynamic behavior effectively.

These limitations necessitate new **monitoring, simulation, and anomaly detection systems** to ensure safe and efficient hydrogen transport.

![Hydrogen Station](enea_h2net_v2/hydrogen-station.jpg)

## 🧠 The Digital Twin Approach

We simulate a hydrogen pipeline system using **MATLAB Simscape**, generating multivariate time series from pressure sensors and mass flow rate sensors. This Digital Twin enables:

- Accurate modeling of **transient and steady-state dynamics**.
- Injection of various **anomalous scenarios**: leaks, compressor failures, delayed responses.
- Evaluation of **supervised anomaly detection models** on **noisy data filtered with a moving average**, reflecting realistic measurement conditions.

## 🎯 Optimal Sensor Placement (OSP)

H2-SimNet integrates a complete **Optimal Sensor Placement (OSP)** framework for hydrogen transport networks, designed as part of a unified **Digital Twin architecture** that jointly addresses sensing, simulation, and anomaly detection.

The objective of the OSP module is to identify the optimal configuration of pressure sensors across the network, maximizing the system’s ability to detect and characterize anomalous conditions (such as leaks, compressor faults, and transient disturbances), while minimizing sensor redundancy and overall deployment cost.

### 🧠 Data-driven modeling via neural surrogate

The sensor placement problem is formulated in a fully data-driven manner through a **neural surrogate model** that learns to estimate the impact of different sensor configurations on the performance of the anomaly detection system.

- Each configuration is evaluated in terms of its ability to improve state reconstruction and anomaly detection within the Digital Twin.
- The informative contribution of individual sensors is learned directly from data.
- Pairwise interactions explicitly capture redundancy and complementarity effects between sensors.

This establishes a direct link between sensor deployment strategies and the actual performance of the monitoring and detection pipeline.

### ⚙️ QUBO formulation

The problem is expressed as a **:contentReference[oaicite:0]{index=0}** model, where:

- binary variables represent whether a sensor is activated or not;
- linear terms encode individual sensor informativeness;
- quadratic terms capture interactions between sensors (redundancy or synergy);
- a soft penalty term controls the total number of deployed sensors.

The resulting energy function jointly balances detection performance and sparsity of the sensor configuration.

### ⚛️ Quantum and hybrid optimization via QAOA

The QUBO formulation is mapped into the framework of the **:contentReference[oaicite:1]{index=1}**, which solves the optimization problem through a variational quantum circuit composed of:

- alternating cost and mixer layers;
- variational parameters optimized within a hybrid classical–quantum loop.

Circuit parameters are optimized using a global evolutionary strategy followed by local refinement via memetic optimization, ensuring effective exploration of highly non-convex parameter landscapes.

### 🔬 Role within H2-SimNet and the Digital Twin

Within H2-SimNet, the OSP module plays a fundamental role in enabling a coherent Digital Twin workflow by:

- optimizing sensor deployment across hydrogen transport networks;
- improving the quality of system state observability;
- enhancing the performance of anomaly detection models;
- reducing measurement redundancy and sensing costs;
- enabling co-design of physical infrastructure and intelligent monitoring layers.


## 🗂️ Version History

- **TDADH2 v1** — Initial simplified version of the hydrogen transport network simulation.  
  Developed as a proof-of-concept.  

- **TDADH2 v2** — Enhanced and extended version developed for this work.  
  Includes a more detailed simulation model, additional anomaly scenarios, refined labeling, sensor noise simulation, and an expanded set of **supervised baseline models**.

## 📊 Dataset Features

The synthetic dataset includes:

- **Normal operation**: pressure stabilizes after a transient regime.
- **Anomalies**: compressor faults, local restrictions (valve closures), delayed pressure recovery.
- **Sensor noise**: Gaussian noise added to simulate measurement errors.
- **Labels**: for normal and anomalous timestamps.
- **Preprocessing**: moving average filter applied to noisy signals.

## 🧪 Baseline Experiments

As an initial benchmark, we evaluated several **supervised learning models** on the **noisy, filtered data**:

- **MLP (Multi-Layer Perceptron)**  
- **LSTM (Long Short-Term Memory network) -> Seq2Lab**  
- **CNN+LSTM (Convolutional + LSTM network)**
- 
For the unsupervised anomaly detection, the following autoencoder-based models were considered:

- **AE (Standard Autoencoder)** 
- **LSTM-AE (LSTM Autoencoder)** 
- **CNN-LSTM-AE (CNN + LSTM Autoencoder)** 

These models capture temporal patterns in the noisy sensor data and allow robust anomaly detection in hydrogen transport networks.

## 📁 Repository Structure
```bash
enea_h2net_v1/
│
├── anomaly_detection_classifiers/ # Classical baseline models (v1)
├── jupyter/ # Notebooks for visualization and analysis
├── time_series_plot/ # Time series plotting utilities
├── utils/ # Common utility functions
├── anomaly_detection_hydrogen_network_pressure_sensors.py
└── output.txt # Example output logs

enea_h2net_v2/
│
├── unsupervised_framework_experiments/
│   ├── main.py
│   ├── anomalies_detector.py
│   └── build_unsupervised_dataset.py
├── supervised_framework_experiments/
│   ├── main.py
│   ├── anomalies_detector.py
│   └── build_supervised_dataset.py
├── optimal_sensor_placement/
    ├── QUBO_opt_deploy_realsim.py
    ├── QUBO_opt_deploy_shots.py
    ├── mut_info_delta_loss_estimation.py
    ├── mut_info_delta_loss_pair_estimation.py
    └── result/
        ├──
        ├──
        ├──
        └──
├── anomaly_labeling_and_noise_simulation.ipynb
└── hydrogen-station.jpg # Schematic illustration
```

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/andysinx/H2-SimNet.git
cd H2-SimNet
pip install -r requirements.txt
```

Run the Jupyter notebooks or scripts in the respective folders to reproduce the experiments.

## 👨‍🔬 Project Context
This work is part of the Industrial Ph.D. program in Computational Intelligence at the University of Naples Federico II, funded by ENEA. It focuses on the development of advanced anomaly detection methods for critical energy infrastructures.


