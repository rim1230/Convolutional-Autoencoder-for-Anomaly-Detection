# Convolutional-Autoencoder-for-Anomaly-Detection

This repository provides a PyTorch implementation of autoencoders (both Convolutional and MLP-based) for anomaly detection on time series waveform data (e.g., from CSV files).

## Features

- CNN & MLP based autoencoders
- Reconstruction-based anomaly detection
- Simple training & evaluation via `main.py`
- CSV waveform input support

## Directory Structure
```
.
├── data/              # CSV waveform data
├── configs/           # json config files
├── src/               # Model & utility code
    └── main.py        # Training + evaluation entry point
├── outputs/           # Results, scores
└── README.md
```

## Usage

1. **Prepare Data**  
   Place your waveform CSV files in the `data/` directory.

2. **Run Training & Evaluation**
   ```bash
   python3 main.py --config configs/sample.json
   ```
- Reconstruction loss saved in `outputs/` directory.

## Requirements
- Python 3.8+
- PyTorch
- Numpy, torchinfo, tqdm

## License
MIT License
