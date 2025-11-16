# Presentation Control Glove with Real-time Gesture Recognition with FPGA Acceleration

> A complete end-to-end system for real-time hand gesture recognition using dual IMU sensors, deployed on FPGA hardware for ultra-low latency inference. This includes code used in data collection, preprocessing, training on Google Colab and files needed to create IP block to run inference on FPGA.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/suveenellawela/hand-gesture-classification-2-imu-glove)

## 🎯 Overview

This project implements a real-time gesture recognition system using data from two IMU (Inertial Measurement Unit) sensors - one on the wrist and one on the index finger. The system:

- Captures 6-axis IMU data (3-axis accelerometer + 3-axis gyroscope) from 2 sensors
- Extracts 84 hand-crafted features from 1-second windows
- Classifies gestures using a Multi-Layer Perceptron (MLP) neural network
- Achieves **sub 3 ms inference time** on FPGA hardware (Xilinx Ultra96/ZCU104)
- Recognizes **8 gesture classes** with **99% accuracy**

### Supported Gestures
0 - NONE
1 - SLIDE_LEFT
2 - SLIDE_RIGHT
3 - WRIST_TURN_CLOCKWISE
4 - WRIST_TURN_ANTI_CLOCKWISE
5 - SLIDE_UP
6 - SLIDE_DOWN
7 - SHAKE

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

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SuveenE/presentation-control-glove.git
cd presentation-control-glove
```

2. **Install Python dependencies**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

3. **Download the dataset**
- Dataset is available on Kaggle: [https://www.kaggle.com/datasets/suveenellawela/hand-gesture-classification-2-imu-glove]

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

## 📝 Citation

If you use this work in your research or project, please cite:

```bibtex
@software{gesture_recognition_fpga_2025,
  author = {Suveen Ellawela},
  title = {Presentation Control Glove with Real-time Gesture Recognition with FPGA Acceleration},
  year = {2025},
  url = {https://github.com/SuveenE/presentation-control-glove},
  note = {Dataset: https://www.kaggle.com/datasets/suveenellawela/hand-gesture-classification-2-imu-glove}
}
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Dataset collected as part of CG4002 Embedded Systems Design Project, NUS
- FPGA deployment using Xilinx Vivado HLS and PYNQ framework
- IMU sensors: MPU6050 (wrist and index finger)

## 📧 Contact

- **GitHub**: https://github.com/SuveenE/
- **Email**: suveen.te1[at]gmail.com

---
