# Real-Time Gesture Recognition with FPGA Acceleration

> A complete end-to-end system for real-time hand gesture recognition using dual IMU sensors, deployed on FPGA hardware for ultra-low latency inference.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/dataset-Kaggle-20BEFF.svg)](https://kaggle.com/your-dataset-link)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Dataset and Data Format](#dataset-and-data-format)
- [Usage Examples](#usage-examples)
- [Model Details](#model-details)
- [Hardware Deployment](#hardware-deployment)
- [Performance](#performance)
- [Citation](#citation)

## 🎯 Overview

This project implements a real-time gesture recognition system using data from two IMU (Inertial Measurement Unit) sensors - one on the wrist and one on the index finger. The system:

- Captures 6-axis IMU data (3-axis accelerometer + 3-axis gyroscope) from 2 sensors
- Extracts 84 hand-crafted features from 1-second windows
- Classifies gestures using a Multi-Layer Perceptron (MLP) neural network
- Achieves **X.X ms inference time** on FPGA hardware (Xilinx Ultra96/ZCU104)
- Recognizes **8 gesture classes** with **XX% accuracy**

### Supported Gestures
1. Slide Left
2. Slide Right
3. Slide Up
4. Slide Down
5. Wrist Turn Clockwise
6. Wrist Turn Anti-Clockwise
7. Grasp
8. None (idle/no gesture)

## ✨ Features

- **Complete Pipeline**: Data collection → Preprocessing → Training → Deployment
- **Hardware Accelerated**: FPGA implementation for real-time inference
- **Efficient Feature Extraction**: 84-dimensional feature vector optimized for embedded systems
- **Variance-Based Segmentation**: Automatic gesture detection from continuous IMU streams
- **Production Ready**: Includes C++ implementation and FPGA bitstream

## 🏗️ System Architecture

```
IMU Sensors → Data Collection → Segmentation → Feature Extraction → Classification
  (ESP32)        (MQTT)        (Variance)      (84 features)       (MLP on FPGA)
```

### Data Flow
1. **Raw IMU Data**: 12 channels (2 IMUs × 6 axes) sampled at ~50Hz
2. **Segmentation**: Variance-based detection identifies 1-second gesture windows
3. **Feature Extraction**: 84 statistical features (mean, std, RMS, max, median, correlations)
4. **Preprocessing**: StandardScaler normalization
5. **Inference**: MLP classification on FPGA

## 📁 Repository Structure

```
gesture-recognition-fpga/
├── README.md                          # This file
├── model_info.json                    # Model metadata and performance metrics
│
├── model/                             # Trained model artifacts
│   ├── weights_npy/                   # NumPy weight files (8 files)
│   │   ├── w0.npy, b0.npy            # Layer 1 weights & biases
│   │   ├── w1.npy, b1.npy            # Layer 2 weights & biases
│   │   ├── w2.npy, b2.npy            # Layer 3 weights & biases
│   │   └── w3.npy, b3.npy            # Output layer weights & biases
│   ├── mlp_weights.h                  # C header with weights
│   ├── mlp_test_data.h               # Test data for validation
│   └── scaler.pkl                     # StandardScaler for preprocessing
│
├── notebooks/                         # Jupyter notebooks
│   ├── train_model.ipynb             # Model training pipeline
│   └── evaluate_model.ipynb          # Model evaluation and testing
│
├── src/                               # Python source code
│   ├── preprocess.py                 # Feature extraction (84 features)
│   ├── segment_gestures.py           # Gesture segmentation from streams
│   └── extract_weights.py            # Convert model to C headers
│
└── hardware/                          # FPGA/embedded implementation
    ├── mlp_model.cpp                 # C++ inference implementation
    ├── mlp_model.h                   # C++ header
    ├── mlp_model_test.cpp            # C++ test harness
    ├── mlp_weights.h                 # Model weights (C header)
    ├── mlp_test_data.h               # Test data (C header)
    └── bitstream/                    # FPGA bitstream files
        ├── mlp.bit                   # FPGA bitstream
        ├── mlp.hwh                   # Hardware handoff file
        └── mlp.xsa                   # Xilinx System Archive
```

## 🚀 Getting Started

### Prerequisites

**Python 3.8+** with the following packages:
```bash
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.1.0
tensorflow>=2.8.0       # For model training
matplotlib>=3.4.0       # For visualization
jupyter>=1.0.0          # For notebooks
```

**For FPGA deployment (optional):**
- Xilinx Vivado 2020.2+
- PYNQ-enabled board (Ultra96, ZCU104, etc.)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gesture-recognition-fpga.git
cd gesture-recognition-fpga
```

2. **Install Python dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas scikit-learn joblib tensorflow matplotlib jupyter
```

3. **Download the dataset**
- Dataset is available on Kaggle: [link-to-your-dataset]
- Extract CSV files to your working directory

## 📊 Dataset and Data Format

### Dataset
The training dataset is available on Kaggle: **[Link to your Kaggle dataset]**

**Dataset Statistics:**
- Total gesture windows: X,XXX
- Classes: 8 (7 gestures + null/none class)
- Sampling rate: ~50 Hz
- Window duration: 1 second (~50 samples per gesture)
- Subjects: X participants
- Train/Val/Test split: XX% / XX% / XX%

### Raw IMU Data Format

Each gesture sample is stored as a CSV file with the following structure:

**Columns (13 total):**
```
timestamp, 
Imu0_linear_accleration_x, Imu0_linear_accleration_y, Imu0_linear_accleration_z,
Imu0_angular_velocity_x, Imu0_angular_velocity_y, Imu0_angular_velocity_z,
Imu1_linear_accleration_x, Imu1_linear_accleration_y, Imu1_linear_accleration_z,
Imu1_angular_velocity_x, Imu1_angular_velocity_y, Imu1_angular_velocity_z
```

**Note:** The typo "accleration" (instead of "acceleration") is intentional and preserved for consistency with the data collection system.

**Column Details:**
- `timestamp`: Time in milliseconds (ESP32 clock)
- `Imu0_*`: Wrist IMU data
  - `linear_accleration_[x,y,z]`: Acceleration in m/s² (range: ±156.96)
  - `angular_velocity_[x,y,z]`: Gyroscope in deg/s (range: ±2000)
- `Imu1_*`: Index finger IMU data (same format as IMU0)

**Example CSV:**
```csv
timestamp,Imu0_linear_accleration_x,Imu0_linear_accleration_y,...
1234567,0.12,9.81,0.03,-1.5,2.3,0.8,0.15,9.78,0.05,-1.2,2.5,0.9
1234587,0.14,9.83,0.04,-1.6,2.4,0.7,0.17,9.80,0.06,-1.3,2.6,0.8
...
```

### Label Encoding

| Label | Class Name | Description |
|-------|------------|-------------|
| 0 | NONE | No gesture / idle state |
| 1 | SLIDE_LEFT | Hand sliding left |
| 2 | SLIDE_RIGHT | Hand sliding right |
| 3 | WRIST_TURN_CLOCKWISE | Wrist rotation clockwise |
| 4 | WRIST_TURN_ANTI_CLOCKWISE | Wrist rotation counter-clockwise |
| 5 | SLIDE_UP | Hand sliding up |
| 6 | SLIDE_DOWN | Hand sliding down |
| 7 | GRASP | Grasping motion |

## 💻 Usage Examples

### 1. Extract Features from Gesture Data

```python
from src.preprocess import features_from_12xN
import pandas as pd

# Load gesture CSV
df = pd.read_csv('gesture_sample.csv')

# Extract 84 features (automatically handles the DataFrame)
features = features_from_12xN(df, timing=True)
print(f"Extracted {len(features)} features")

# Access specific features
print(f"Wrist accel X mean: {features['Imu0_acc_x_mean']}")
print(f"Cross-sensor correlation: {features['corr_acc_W_F']}")
```

### 2. Segment Gestures from Continuous Data

Use this when you have a long recording and want to automatically detect and extract gesture windows:

```bash
# Basic usage with automatic threshold detection
python src/segment_gestures.py long_recording.csv --output-dir ./segmented/

# With visualization to tune parameters
python src/segment_gestures.py long_recording.csv \
    --output-dir ./segmented/ \
    --window-size 1.0 \
    --threshold auto \
    --visualize

# Custom threshold
python src/segment_gestures.py recording.csv \
    --output-dir ./out/ \
    --threshold 5000 \
    --min-gap 0.3
```

### 3. Full Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
import joblib
from src.preprocess import features_from_12xN

# Load gesture CSV
df = pd.read_csv('my_gesture.csv')

# Extract features
features_dict = features_from_12xN(df)

# Convert to array
feature_values = np.array(list(features_dict.values())).reshape(1, -1)

# Load scaler and normalize
scaler = joblib.load('model/scaler.pkl')
features_scaled = scaler.transform(feature_values)

# features_scaled is now ready for model inference (shape: 1×84)
print(f"Scaled features ready for inference: {features_scaled.shape}")
```

### 4. Train Your Own Model

Open and run `notebooks/train_model.ipynb` for the complete training pipeline:
- Load and preprocess dataset
- Train MLP classifier
- Evaluate performance with confusion matrix
- Export model weights to NumPy files
- Save StandardScaler

### 5. Test the Trained Model

Use `notebooks/evaluate_model.ipynb` to:
- Load trained model
- Test on validation/test set
- Generate performance metrics
- Visualize predictions

## 🧠 Model Details

### Architecture
```
Input (84) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(8, Softmax)
```

### Feature Engineering (84 features)
- **Per-IMU features (40 × 2 = 80)**:
  - Axis statistics (5 stats × 3 axes × 2 signals): mean, std, RMS, max, median
    - 3 accel axes (x, y, z)
    - 3 gyro axes (x, y, z)
  - Magnitude statistics (5 stats × 2 magnitudes): accel magnitude, gyro magnitude

- **Cross-IMU features (4)**:
  - Acceleration magnitude correlation (wrist ↔ finger)
  - Gyroscope magnitude correlation (wrist ↔ finger)
  - RMS ratio: finger accel / wrist accel
  - RMS ratio: finger gyro / wrist gyro

### Training Details
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Regularization**: Dropout (0.3)
- **Data Augmentation**: [Describe any augmentation used]

### Performance Metrics
| Metric | Value |
|--------|-------|
| Test Accuracy | XX.X% |
| Precision (macro avg) | XX.X% |
| Recall (macro avg) | XX.X% |
| F1-Score (macro avg) | XX.X% |
| Inference Time (FPGA) | X.X ms |
| Model Size | XX KB |

## ⚡ Hardware Deployment (FPGA)

The model is deployed on Xilinx FPGA for ultra-low latency inference. The repository includes:
- **C++ implementation** (`mlp_model.cpp/.h`) - Pure C++ inference without dependencies
- **Pre-compiled bitstream** (`hardware/bitstream/`) - Ready to deploy on PYNQ boards
- **Test harness** (`mlp_model_test.cpp`) - Validates C++ implementation against known outputs

### Quick Start on PYNQ Board

**Hardware tested:** Xilinx Ultra96-V2, ZCU104

1. **Copy files to board**
```bash
scp hardware/bitstream/* xilinx@192.168.3.1:/home/xilinx/
scp model/scaler.pkl xilinx@192.168.3.1:/home/xilinx/
```

2. **Run on FPGA**
```bash
ssh xilinx@192.168.3.1
cd /home/xilinx
source /usr/local/share/pynq-venv/bin/activate

# Load bitstream and run inference
python your_inference_script.py
```

### Test C++ Implementation Locally

```bash
cd hardware
g++ -O3 -o mlp_test mlp_model_test.cpp mlp_model.cpp
./mlp_test
```

This validates the C++ model against test data embedded in `mlp_test_data.h`.

### Regenerate C Headers from Model

If you retrain the model and want to regenerate the C header files:

```bash
cd src
python extract_weights.py
```

This reads the `.npy` files from `model/weights_npy/` and generates:
- `mlp_weights.h` - Model parameters
- `mlp_test_data.h` - Test vectors for validation

## 🔬 Reproducing Results

1. **Download dataset** from Kaggle
2. **Train model:** Open `notebooks/train_model.ipynb` and run all cells
3. **Evaluate:** Open `notebooks/evaluate_model.ipynb` for metrics and confusion matrix
4. **Deploy to FPGA:** Follow hardware deployment steps above

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

## 📝 Citation

If you use this work in your research or project, please cite:

```bibtex
@software{gesture_recognition_fpga_2025,
  author = {Your Name},
  title = {Real-Time Gesture Recognition with FPGA Acceleration},
  year = {2025},
  url = {https://github.com/yourusername/gesture-recognition-fpga},
  note = {Dataset: https://kaggle.com/your-dataset-link}
}
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Dataset collected as part of CG4002 Embedded Systems Design Project, NUS
- FPGA deployment using Xilinx Vivado HLS and PYNQ framework
- IMU sensors: MPU6050 (wrist and index finger)

## 📧 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/gesture-recognition-fpga/issues)

---

**Project Tags**: `gesture-recognition` `imu-sensors` `fpga` `machine-learning` `real-time-inference` `embedded-systems` `wearable-computing` `xilinx` `pynq` `accelerometer` `gyroscope` `hand-gestures`

