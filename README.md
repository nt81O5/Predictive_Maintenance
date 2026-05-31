# CMAPSS Turbofan Engine Predictive Maintenance using LSTM

This repository contains a PyTorch-based Deep Learning solution using Long Short-Term Memory (LSTM) networks to predict the **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS dataset.

## Dataset
The repository utilizes the **C-MAPSS (Cognitive Microwave Aerospace Propulsion System Simulation) dataset** (FD001 sub-dataset):
* **`train_FD001.txt`**: Engine run-to-failure operational history under normal settings.
* **`test_FD001.txt`**: Engine operational history ending some time before failure.
* **`RUL_FD001.txt`**: The true Remaining Useful Life (RUL) values for the engines in the test set at their final recorded cycle.

## Model Architecture
The model uses a multi-layer Long Short-Term Memory (LSTM) network to capture temporal dependencies in the turbofan sensor time-series data, followed by a fully-connected layer to predict the RUL.

* **LSTM Inputs**: 21 sensor channels and 3 operational settings, normalized to `[-1, 1]` using `MinMaxScaler`.
* **Sequence Length**: 30 cycles (historical sliding window).
* **Optimizer**: Adam (`lr=0.001`).
* **Loss Function**: Mean Squared Error (MSE).

## Performance Summary
* **Test MAE**: ~18.83
* **Test RMSE**: ~26.87
* **Relative Accuracy**: ~75.75%
* **Prediction Accuracy (within ±10 cycles)**: ~46%

## Repository Structure
```bash
├── NASA_LSTMipynb.ipynb   # Jupyter Notebook containing the training and evaluation pipeline
├── train_FD001.txt        # Training dataset
├── test_FD001.txt         # Testing dataset
├── RUL_FD001.txt          # Ground truth RUL for test set
├── .gitignore             # Git ignore file for temp files, IDE artifacts, and Pycache
└── README.md              # Repository documentation
```

## Getting Started

### Prerequisites
Make sure you have python 3.8+ installed. You can install all dependencies via:
```bash
pip install pandas numpy matplotlib scikit-learn torch
```

### Running the Notebook
You can open and execute all cells in `NASA_LSTMipynb.ipynb` to train the LSTM model, predict RUL on the test dataset, and visualize performance metrics.
