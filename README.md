# PCB Defect Detection System

A complete AI-powered system for detecting and classifying defects in printed circuit boards (PCBs) using deep learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Training Your Own Model](#training-your-own-model)
- [Web Interface](#web-interface)
- [Export Options](#export-options)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This system uses computer vision and deep learning (EfficientNet-B4) to automatically detect and classify PCB defects. It compares a test board against a template (defect-free) board to identify anomalies.

**Detectable Defect Types:**
- Missing Hole
- Mouse Bite
- Open Circuit
- Short Circuit
- Spur
- Spurious Copper

## ✨ Features

- **Automated Detection**: Finds defects by comparing test vs template boards
- **AI Classification**: Identifies defect types with confidence scores
- **Web Interface**: User-friendly Streamlit app
- **Multiple Export Formats**: 
  - Annotated images (JPG)
  - CSV logs
  - PDF reports
- **High Accuracy**: 99.8% test accuracy
- **Fast Performance**: Cached model loading for speed
- **Beginner-Friendly**: Simple, well-documented code

## 💻 System Requirements

- **Python**: 3.9 or higher (tested on 3.13)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended for faster training
- **Storage**: 2GB for model and data
- **OS**: Windows, Linux, or macOS

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/pcb-defect-detection.git
cd pcb-defect-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; import cv2; import streamlit; print('All dependencies installed!')"
```

## 📁 Project Structure

```
Circuit_Guard_New/
├── data/                          # Dataset directory
│   ├── rois/                      # Extracted defect regions
│   │   ├── Missing_hole/
│   │   ├── Mouse_bite/
│   │   └── ...
│   └── splits/                    # Train/val/test split
│       ├── train/
│       ├── val/
│       ├── test/
│       └── class_mapping.json
│
├── PCB_DATASET/                   # Original dataset
│   ├── images/                    # Test PCB images
│   ├── Annotations/               # XML annotations
│   └── PCB_USED/                  # Template images
│
├── data prepration/               # Data preprocessing scripts
│   ├── xml_parser.py              # Parse XML annotations
│   ├── extract_rois.py            # Extract defect regions
│   ├── split_dataset.py           # Train/val/test split
│   └── main.py                    # Run complete pipeline
│
├── Training Pipeline/             # Model training
│   ├── dataset.py                 # PyTorch dataset
│   ├── efficientnet_model.py      # Model definition
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation metrics
│   ├── checkpoints/               # Saved models
│   └── results/                   # Confusion matrix, reports
│
├── interface/                     # Web application
│   ├── app.py                     # Streamlit interface
│   ├── detect_defects.py          # Detection pipeline
│   └── classify_defects.py        # Classification logic
│
├── simple_app/                    # Simplified version
│   ├── detect_defects.py          # All-in-one detection
│   ├── classify_defects.py        # Simple classification
│   ├── app.py                     # Streamlit app
│   └── main.py                    # CLI menu
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── USER_GUIDE.md                  # Detailed usage guide
```

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

```bash
cd interface
streamlit run app.py
```

1. Upload template PCB (defect-free reference)
2. Upload test PCB (board to inspect)
3. Adjust settings (confidence threshold, minimum area)
4. Click "Detect Defects"
5. View results and export reports

### Option 2: Command Line

```bash
cd simple_app
python main.py
```

Follow the interactive menu to:
- Detect defects only
- Detect and classify defects
- Launch web interface

### Option 3: Python Script

```python
from interface.detect_defects import detect_defects
from interface.classify_defects import classify_defects

# Detect defects
aligned, filtered, defects = detect_defects(
    "test_board.jpg",
    "template.jpg",
    min_area=120
)

# Classify defects
results = classify_defects(
    aligned,
    defects,
    "Training Pipeline/checkpoints/best_model.pth",
    "data/splits/class_mapping.json"
)

# Print results
for det in results:
    print(f"{det['class']}: {det['confidence']:.0%}")
```

## 📖 Usage Guide

### 1. Prepare Your Data

If you have your own PCB images:

```bash
cd "data prepration"
python main.py
```

This will:
- Parse XML annotations
- Extract defect ROIs (128x128 images)
- Split into train/val/test sets (70/15/15)
- Create class mapping

### 2. Train Model (Optional)

If you want to train on your own data:

```bash
cd "Training Pipeline"
python train.py
```

Training parameters:
- Epochs: 50
- Batch size: 32
- Learning rate: 0.0001
- Model: EfficientNet-B4 (~17.5M parameters)

Monitor progress in console. Best model saves automatically.

### 3. Evaluate Model

```bash
cd "Training Pipeline"
python evaluate.py
```

Generates:
- Confusion matrix (PNG)
- Classification report (JSON)
- Per-class metrics (precision, recall, F1-score)

### 4. Run Detection

**Web Interface:**
```bash
cd interface
streamlit run app.py
```

**Command Line:**
```bash
cd interface
python detect_defects.py --test <test_image> --template <template_image>
```

## 🖥️ Web Interface

### Features

1. **Image Upload**
   - Drag & drop or browse
   - Supports JPG, JPEG, PNG

2. **Settings**
   - Model path
   - Class mapping path
   - Minimum defect area (50-500 pixels)
   - Confidence threshold (0.0-1.0)

3. **Results Display**
   - Annotated image with colored boxes
   - Metrics: total defects, high confidence count
   - Expandable defect details with crops

4. **Export Options**
   - 📥 Annotated Image (JPG)
   - 📊 CSV Report (detailed log)
   - 📄 PDF Report (professional summary)

### Settings Guide

**Minimum Defect Area:**
- Lower (50-100): Finds small defects, more noise
- Default (120): Balanced detection
- Higher (200-500): Only large defects, less noise

**Confidence Threshold:**
- 0.0-0.4: Show all predictions (may include false positives)
- 0.5: Balanced (default)
- 0.6-0.8: High confidence only
- 0.9-1.0: Very strict (may miss some defects)

## 📤 Export Options

### 1. Annotated Image
- JPG format
- Colored bounding boxes
- Class labels with confidence
- Ready for presentation

### 2. CSV Report
Contains:
- Timestamp
- Image names
- Settings used
- Defect table (ID, type, confidence, position, size, area)

Example:
```csv
Timestamp, 2025-11-20 12:30:45
Template Image, template_01.jpg
Test Image, test_board_05.jpg
Confidence Threshold, 0.50
Total Defects, 3

ID, Type, Confidence, X, Y, Width, Height, Area
1, Missing hole, 95%, 120, 150, 18, 20, 360
2, Short, 88%, 250, 200, 25, 22, 550
3, Spur, 75%, 180, 300, 15, 18, 270
```

### 3. PDF Report
Professional report with:
- Cover page with summary
- Annotated image
- Defect details table
- Timestamp and metadata
- Ready to share with stakeholders

## 🔧 Troubleshooting

### Installation Issues

**Problem:** `torch` won't install
```bash
# Try CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Problem:** `cv2` import error
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### Detection Issues

**No defects found:**
- Lower `min_area` setting (try 80-100)
- Check if images are different
- Verify image quality (clear, well-lit)
- Ensure template is defect-free

**Too many false positives:**
- Raise `min_area` (try 150-200)
- Increase confidence threshold (0.6-0.8)
- Use better quality images
- Re-align images if tilted

**Model not found:**
```bash
# Check path
ls "Training Pipeline/checkpoints/best_model.pth"

# Train model if missing
cd "Training Pipeline"
python train.py
```

### Performance Issues

**Slow detection:**
- Close other applications
- Use GPU if available
- Model is cached after first load

**Out of memory:**
- Reduce batch size in training
- Use smaller images
- Close other programs

## 🎓 Training Your Own Model

### 1. Prepare Dataset

Your dataset should have:
- Images: PCB test images (JPG/PNG)
- Annotations: Pascal VOC XML format
- Template: Defect-free reference boards

```xml
<!-- Example annotation -->
<annotation>
  <filename>01_missing_hole_01.jpg</filename>
  <size>
    <width>600</width>
    <height>600</height>
  </size>
  <object>
    <name>Missing_hole</name>
    <bndbox>
      <xmin>120</xmin>
      <ymin>150</ymin>
      <xmax>138</xmax>
      <ymax>170</ymax>
    </bndbox>
  </object>
</annotation>
```

### 2. Extract ROIs

```bash
cd "data prepration"
python main.py
```

### 3. Train

```bash
cd "Training Pipeline"
python train.py
```

Monitor training:
- Loss should decrease
- Accuracy should increase
- Best model saves automatically

### 4. Evaluate

```bash
python evaluate.py
```

Check confusion matrix for problem classes.

### 5. Fine-tune (Optional)

If accuracy is low:
- Collect more data
- Adjust learning rate
- Train more epochs
- Try data augmentation

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 📧 Support

For issues or questions:
- Open GitHub issue
- Email: support@example.com
- Documentation: See USER_GUIDE.md

## 🙏 Acknowledgments

- EfficientNet: https://github.com/lukemelas/EfficientNet-PyTorch
- Dataset: PCB defect dataset
- Libraries: PyTorch, OpenCV, Streamlit

## 📊 Performance Metrics

On test set:
- **Overall Accuracy:** 99.8%
- **Missing Hole:** 100% precision, 100% recall
- **Mouse Bite:** 98.7% precision, 100% recall
- **Open Circuit:** 100% precision, 98.6% recall
- **Short:** 100% precision, 100% recall
- **Spur:** 100% precision, 100% recall
- **Spurious Copper:** 100% precision, 100% recall

## 🔄 Version History

- **v1.0.0** (2025-11-20): Initial release
  - Complete detection pipeline
  - EfficientNet-B4 classifier
  - Web interface
  - Export to JPG, CSV, PDF

---

**Made with ❤️ for PCB quality control**
