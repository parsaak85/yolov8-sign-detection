# 🚦 Traffic Sign Detection System

A real-time traffic sign detection and classification system using **OAK-D camera**, **YOLOv8**, **HOG features**, and **SVM classifier**. The system can detect 12 different types of traffic signs and classify turn directions and slope orientations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)

## 📋 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Detected Signs](#-detected-signs)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Real-time Detection**: Processes video stream from OAK-D camera at 30 FPS
- **12 Traffic Sign Classes**: Detects various traffic signs including regulatory and warning signs
- **Direction Classification**: Identifies turn direction (left/right) and slope orientation (uphill/downhill)
- **Hybrid Approach**: Combines YOLOv8 for detection with HOG+SVM for fine-grained classification
- **Fallback Mechanism**: Classical image processing as backup for slope classification
- **High Accuracy**: Confidence threshold filtering ensures reliable detections

## 🏗️ System Architecture

```
┌─────────────────┐
│   OAK-D Camera  │
│   (640×352px)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLOv8 Model   │
│  (Detection)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Sign Detected? │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌──────────────┐
│ Other │  │ Turn/Slope?  │
│ Signs │  └──────┬───────┘
└───────┘         │
                  ▼
         ┌────────────────┐
         │   HOG Feature  │
         │   Extraction   │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  SVM Classifier│
         │  + Fallback    │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ Final Result   │
         │ with Direction │
         └────────────────┘
```

## 🚸 Detected Signs

The system can detect and classify the following traffic signs:

| Category | Signs |
|----------|-------|
| **Regulatory** | Stop, No Passing (Begin/End), Priority, Straight |
| **Warning** | Turn (Left/Right), Slope (Uphill/Downhill) |
| **Information** | Parking, Crosswalk, Barred Area |
| **Infrastructure** | Tunnel (Begin/End) |

### Special Classifications
- **Turn Signs**: Classified as `turnL` (left) or `turnR` (right)
- **Slope Signs**: Classified as `uphill` or `downhill`

## 📦 Requirements

### Hardware
- **OAK-D Camera** (or compatible DepthAI device)
- Computer with USB 3.0 port
- Minimum 4GB RAM recommended

### Software
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/traffic-sign-detection.git
cd traffic-sign-detection
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
Place your trained models in the `models` directory:
```
models/
├── best.pt                    # YOLOv8 trained weights
└── hog_svm_turn_slope.pkl    # SVM classifier
```

## 📁 Project Structure

```
traffic-sign-detection/
├── main.py                    # Main application script
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── models/                   # Model files directory
│   ├── best.pt              # YOLOv8 weights
│   └── hog_svm_turn_slope.pkl  # SVM model
├── docs/                     # Documentation
│   └── performance/         # Performance screenshots
└── training/                # Training scripts (optional)
    ├── train_yolo.py
    └── train_svm.py
```

## 🚀 Usage

### Basic Usage

1. **Connect your OAK-D camera** to your computer via USB

2. **Run the detection system:**
   ```bash
   python main.py
   ```

3. **Controls:**
   - Press `q` to quit the application
   - The live video feed will show detections with bounding boxes and labels

### Expected Output

```
============================================================
Traffic Sign Detection System
OAK-D Camera + YOLOv8 + HOG + SVM
============================================================
✓ YOLOv8 model loaded
✓ SVM model loaded

🎥 OAK-D camera running...
Press 'q' to quit
```

### Detection Format

Detected signs are displayed with:
- **Bounding box** around the sign
- **Label** showing:
  - Sign class name
  - Direction (for turn/slope signs)
  - Confidence score

Example: `turn:turnL 0.87` or `slope:uphill 0.92`

## 🎓 Model Training

### YOLOv8 Training

To train your own YOLOv8 model:

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='traffic_signs.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='traffic_sign_detector'
)
```

### SVM Training

For HOG+SVM classifier:

```python
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
import joblib

# Extract HOG features from your dataset
# Train SVM classifier
# Save model

svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
joblib.dump((svm, label_encoder), 'hog_svm_turn_slope.pkl')
```

## 📊 Performance

### Detection Examples

<!-- Add your performance images here -->
<p align="center">
  <img src="docs/performance/example1.jpg" width="45%" alt="Detection Example 1"/>
  <img src="docs/performance/example2.jpg" width="45%" alt="Detection Example 2"/>
</p>

### Metrics
- **Detection Speed**: ~30 FPS on OAK-D
- **YOLOv8 Confidence Threshold**: 0.5
- **SVM Confidence Threshold**: 0.6 (for turn/slope classification)
- **Input Resolution**: 640×352 pixels

## 🔍 Troubleshooting

### Common Issues

**1. "YOLO model not found" or "SVM model not found"**
- Ensure model files are in the `models/` directory
- Check file names match exactly: `best.pt` and `hog_svm_turn_slope.pkl`

**2. OAK-D camera not detected**
- Check USB connection (use USB 3.0 port)
- Install DepthAI udev rules (Linux):
  ```bash
  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

**3. Low FPS or performance issues**
- Close other applications using the camera
- Reduce confidence threshold
- Use GPU acceleration if available

**4. ImportError for depthai**
- Reinstall depthai: `pip install --force-reinstall depthai`

## 🛠️ Advanced Configuration

### Adjust Confidence Thresholds

Edit in `main.py`:
```python
# YOLO confidence threshold
oak = OAKCameraYOLOv8Custom(YOLO_MODEL_PATH, conf_thresh=0.5)

# SVM confidence threshold (in classify_turn_or_slope function)
if pred in ["turnL", "turnR"] and conf > 0.6:  # Change 0.6 to your value
```

### Change Camera Resolution

Edit in `create_pipeline()` method:
```python
cam.setPreviewSize(640, 352)  # Change to your desired resolution
```

### Modify Classes

Update the `classes` list in `OAKCameraYOLOv8Custom.__init__()`:
```python
self.classes = [
    "barred", "crossWalk", "NoPassB", "NoPassE",
    "park", "priority", "stop", "straight",
    "tunnelB", "tunnelE", "turn", "slope"
]
```


## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [Luxonis](https://github.com/luxonis/depthai-python) for OAK-D camera support
- [scikit-image](https://scikit-image.org/) for HOG feature extraction
- [scikit-learn](https://scikit-learn.org/) for SVM classification
