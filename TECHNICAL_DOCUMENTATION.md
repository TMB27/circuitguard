# Technical Documentation: PCB Defect Detection System

## System Architecture

### Overview

The PCB Defect Detection System consists of three main components:

1. **Data Preparation Pipeline** - Processes raw images and annotations
2. **Training Pipeline** - Trains deep learning model for classification
3. **Inference Pipeline** - Detects and classifies defects in production

```
┌─────────────────────────────────────────────────────────────┐
│                    PCB Defect Detection System               │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌──────────────┐    ┌──────────────────┐
│     Data      │     │   Training   │    │    Inference     │
│  Preparation  │───▶│   Pipeline   │───▶│     Pipeline     │
└───────────────┘     └──────────────┘    └──────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   XML + Images          Trained Model        Defect Reports
```

---

## 1. Data Preparation Pipeline

### Purpose
Convert raw PCB images and Pascal VOC XML annotations into training-ready dataset.

### Components

#### 1.1 XML Parser (`xml_parser.py`)

**Function:** `parse_xml(xml_path)`
- Parses Pascal VOC format XML files
- Extracts bounding boxes and class labels
- Returns structured annotation dictionary

**Input:**
```xml
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

**Output:**
```python
{
    'filename': '01_missing_hole_01.jpg',
    'width': 600,
    'height': 600,
    'boxes': [
        {
            'class': 'Missing_hole',
            'xmin': 120,
            'ymin': 150,
            'xmax': 138,
            'ymax': 170,
            'width': 18,
            'height': 20,
            'area': 360
        }
    ]
}
```

**Time Complexity:** O(n) where n = number of objects
**Space Complexity:** O(n)

---

#### 1.2 ROI Extractor (`extract_rois.py`)

**Function:** `extract_rois_from_image(image, annotation, target_size=128)`

**Process:**
1. Load original image
2. Extract bounding boxes from annotation
3. Crop each defect region (with padding)
4. Resize to 128×128 pixels
5. Save as PNG with metadata JSON

**Algorithm:**
```python
for bbox in annotation['boxes']:
    # Add padding
    x1 = max(0, bbox['xmin'] - padding)
    y1 = max(0, bbox['ymin'] - padding)
    x2 = min(image.width, bbox['xmax'] + padding)
    y2 = min(image.height, bbox['ymax'] + padding)
    
    # Crop
    roi = image[y1:y2, x1:x2]
    
    # Resize
    roi = cv2.resize(roi, (128, 128))
    
    # Save
    cv2.imwrite(output_path, roi)
```

**Output Structure:**
```
data/rois/
├── Missing_hole/
│   ├── 01_missing_hole_01_0.png
│   ├── 01_missing_hole_01_0_metadata.json
│   ├── 01_missing_hole_01_1.png
│   └── ...
├── Mouse_bite/
├── Open_circuit/
├── Short/
├── Spur/
└── Spurious_copper/
```

**Performance:**
- Speed: ~10-50 ROIs/second (CPU)
- Memory: O(1) per ROI (streaming)

---

#### 1.3 Dataset Splitter (`split_dataset.py`)

**Function:** `split_by_class(rois, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)`

**Strategy:** Stratified split
- Maintains class distribution across splits
- Random shuffling with fixed seed
- Copies files to train/val/test directories

**Algorithm:**
```python
for class_name in classes:
    class_rois = filter_by_class(rois, class_name)
    shuffle(class_rois, seed=42)
    
    n = len(class_rois)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_set = class_rois[:train_end]
    val_set = class_rois[train_end:val_end]
    test_set = class_rois[val_end:]
    
    copy_to_split(train_set, 'train')
    copy_to_split(val_set, 'val')
    copy_to_split(test_set, 'test')
```

**Output:**
```
data/splits/
├── train/
│   ├── Missing_hole/
│   ├── Mouse_bite/
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── class_mapping.json
```

**Class Mapping:**
```json
{
    "Missing_hole": 0,
    "Mouse_bite": 1,
    "Open_circuit": 2,
    "Short": 3,
    "Spur": 4,
    "Spurious_copper": 5
}
```

---

## 2. Training Pipeline

### Architecture

**Model:** EfficientNet-B4
- Pretrained on ImageNet
- Modified final layer for 6 classes
- ~17.5M parameters

### Components

#### 2.1 Dataset (`dataset.py`)

**Class:** `PCBDataset(torch.utils.data.Dataset)`

**Methods:**
- `__init__`: Load image paths and labels
- `__len__`: Return dataset size
- `__getitem__`: Load and preprocess single image

**Preprocessing:**
```python
def __getitem__(self, idx):
    # Load image
    image = cv2.imread(self.images[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (128, 128))
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    # Convert to tensor (C, H, W)
    image = torch.FloatTensor(image).permute(2, 0, 1)
    
    label = self.labels[idx]
    return image, label
```

**DataLoader Configuration:**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0  # Windows compatibility
)
```

---

#### 2.2 Model (`efficientnet_model.py`)

**Class:** `PCBModel(nn.Module)`

**Architecture:**
```python
class PCBModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Pretrained backbone
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Custom classifier
        in_features = self.model._fc.in_features  # 1792
        self.model._fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
```

**Model Summary:**
```
EfficientNet-B4
├── Stem: Conv2d(3, 48)
├── Blocks: 32 MBConv blocks
├── Head: Conv2d(272, 1792)
├── Pool: AdaptiveAvgPool2d(1, 1)
└── Classifier: Linear(1792, 6)

Total parameters: 17,559,374
Trainable parameters: 17,559,374
```

---

#### 2.3 Training Script (`train.py`)

**Function:** `train_one_epoch(model, train_loader, criterion, optimizer, device)`

**Training Loop:**
```python
for epoch in range(num_epochs):
    model.train()
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
```

**Hyperparameters:**
```python
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
OPTIMIZER = Adam
LOSS = CrossEntropyLoss
```

**Training Time:**
- CPU: ~2-4 hours (50 epochs)
- GPU (GTX 1060): ~30-60 minutes

---

#### 2.4 Evaluation (`evaluate.py`)

**Metrics:**

1. **Confusion Matrix**
   ```python
   cm = confusion_matrix(y_true, y_pred)
   # Visualize as heatmap
   plt.imshow(cm, cmap='Blues')
   ```

2. **Classification Report**
   ```python
   report = classification_report(
       y_true, y_pred,
       target_names=class_names,
       output_dict=True
   )
   ```

**Output Metrics:**
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
- Accuracy: (TP + TN) / Total

**Performance on Test Set:**
```
                   Precision  Recall  F1-Score  Support
Missing_hole         1.000    1.000     1.000       75
Mouse_bite           0.987    1.000     0.993       74
Open_circuit         1.000    0.986     0.993       73
Short                1.000    1.000     1.000       74
Spur                 1.000    1.000     1.000       74
Spurious_copper      1.000    1.000     1.000       76

Accuracy                                0.998      446
Macro avg            0.998    0.998     0.998      446
Weighted avg         0.998    0.998     0.998      446
```

---

## 3. Inference Pipeline

### Detection Process

#### 3.1 Image Subtraction

**Function:** `subtract_images(test_img, template_img)`

**Algorithm:**
```python
def subtract_images(test, template):
    # Align images
    aligned = align_images(test, template)
    
    # Convert to grayscale
    test_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Absolute difference
    diff = cv2.absdiff(test_gray, template_gray)
    
    return diff
```

**Alignment Method:** Feature-based (ORB)
```python
def align_images(test, template):
    # Detect features
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(test, None)
    kp2, des2 = orb.detectAndCompute(template, None)
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]
    
    # Find homography
    src_pts = [kp1[m.queryIdx].pt for m in matches]
    dst_pts = [kp2[m.trainIdx].pt for m in matches]
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    # Warp image
    aligned = cv2.warpPerspective(test, H, template.shape[:2][::-1])
    return aligned
```

**Time Complexity:** O(n²) where n = number of features
**Typical Runtime:** 2-5 seconds

---

#### 3.2 Thresholding

**Function:** `threshold_image(diff_img)`

**Algorithm:** Otsu's Method
```python
def threshold_image(diff):
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # Otsu's thresholding
    threshold, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary
```

**Otsu's Method:**
- Automatically finds optimal threshold
- Maximizes inter-class variance
- Works well for bimodal histograms

**Formula:**
```
σ²_between = w₀ * w₁ * (μ₀ - μ₁)²
where:
  w₀, w₁ = class probabilities
  μ₀, μ₁ = class means
```

---

#### 3.3 Morphological Filtering

**Function:** `filter_noise(binary_img)`

**Operations:**
1. **Opening** (Erosion → Dilation)
   - Removes small noise
   - Preserves larger structures

2. **Dilation**
   - Enhances defect regions
   - Fills small gaps

**Algorithm:**
```python
def filter_noise(binary):
    # Opening kernel (3×3 ellipse)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)
    
    # Dilation kernel (5×5 ellipse)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    enhanced = cv2.dilate(opened, kernel2, iterations=2)
    
    return enhanced
```

**Visual Effect:**
```
Before:           After:
. . . X . .       . . . . . .
. X X X . .  →    . X X X . .
X X X X X .       X X X X X .
. . X . . .       . . X X . .
```

---

#### 3.4 Contour Detection

**Function:** `find_defects(filtered_img, min_area=100)`

**Algorithm:**
```python
def find_defects(filtered, min_area):
    # Find contours
    contours, _ = cv2.findContours(
        filtered,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    defects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            defects.append({
                'x': x, 'y': y,
                'width': w, 'height': h,
                'area': area
            })
    
    return defects
```

**Time Complexity:** O(n) where n = number of pixels
**Typical Runtime:** <1 second

---

#### 3.5 Classification

**Function:** `predict_defect(model, image, device, class_mapping)`

**Process:**
```python
def predict_defect(model, image, device, class_mapping):
    # Preprocess
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = probabilities.max(1)
    
    # Get class name
    class_id = predicted_class.item()
    id_to_class = {v: k for k, v in class_mapping.items()}
    class_name = id_to_class[class_id]
    
    return class_name, confidence.item()
```

**Inference Time:** 
- CPU: ~50-100ms per defect
- GPU: ~10-20ms per defect

---

## 4. Web Interface

### Technology Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **Image Processing:** OpenCV
- **Deep Learning:** PyTorch
- **Reporting:** ReportLab (PDF), CSV

### Architecture

```
User Browser
     ↓
Streamlit Server (app.py)
     ↓
┌────────────────────────────────┐
│  Cached Resources              │
│  - Model (cached)              │
│  - Session state               │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│  Detection Pipeline            │
│  1. Upload images              │
│  2. Image alignment            │
│  3. Subtraction & thresholding │
│  4. Morphological filtering    │
│  5. Contour detection          │
│  6. Classification             │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│  Export                        │
│  - Annotated image (JPG)       │
│  - CSV report                  │
│  - PDF report                  │
└────────────────────────────────┘
```

### Optimization Techniques

**1. Model Caching**
```python
@st.cache_resource
def load_model_cached(model_path, class_mapping_path):
    # Load once, reuse for all requests
    model = load_model(model_path)
    return model
```

**Benefit:** 10-100x faster for subsequent detections

**2. Session State**
```python
st.session_state['result_img'] = result
st.session_state['classified'] = classified
```

**Benefit:** Preserves results across reruns

**3. Progress Indicators**
```python
progress = st.progress(0)
status = st.empty()

status.text("Finding defects...")
progress.progress(40)
```

**Benefit:** Better user experience

---

## 5. Data Flow

### Complete Pipeline

```
Input: Test PCB Image + Template Image
  ↓
┌──────────────────────────┐
│  1. Image Alignment      │ (2-5s)
│     - ORB feature detect │
│     - Homography warp    │
└──────────────────────────┘
  ↓
┌──────────────────────────┐
│  2. Subtraction          │ (<1s)
│     - Grayscale convert  │
│     - Absolute diff      │
└──────────────────────────┘
  ↓
┌──────────────────────────┐
│  3. Thresholding         │ (<1s)
│     - Gaussian blur      │
│     - Otsu's threshold   │
└──────────────────────────┘
  ↓
┌──────────────────────────┐
│  4. Filtering            │ (<1s)
│     - Morphological ops  │
│     - Noise removal      │
└──────────────────────────┘
  ↓
┌──────────────────────────┐
│  5. Contour Detection    │ (<1s)
│     - Find contours      │
│     - Filter by area     │
└──────────────────────────┘
  ↓
┌──────────────────────────┐
│  6. Classification       │ (0.5-1s per defect)
│     - Crop defect ROI    │
│     - Model inference    │
│     - Softmax output     │
└──────────────────────────┘
  ↓
Output: Defect List with Classifications
```

**Total Time:** 5-30 seconds (depending on number of defects)

---

## 6. Performance Analysis

### Computational Complexity

| Component         | Time        | Space    | Notes                |
|-------------------|-------------|----------|----------------------|
| Image Loading     | O(n)        | O(n)     | n = pixels           |
| Feature Detection | O(n log n)  | O(k)     | k = features         |
| Homography        | O(k²)       | O(1)     | RANSAC               |
| Subtraction       | O(n)        | O(n)     | Element-wise         |
| Thresholding      | O(n)        | O(n)     | Single pass          |
| Morphology        | O(n × m²)   | O(n)     | m = kernel size      |
| Contours          | O(n)        | O(c)     | c = contours         |
| Classification    | O(1)        | O(1)     | Fixed input size     |

### Memory Usage

| Component         | Memory       | Notes                     |
|-------------------|--------------|---------------------------|
| Original Image    | 600×600×3    | ~1.08 MB (uint8)          |
| Aligned Image     | 600×600×3    | ~1.08 MB                  |
| Difference Map    | 600×600      | ~360 KB                   |
| Binary Mask       | 600×600      | ~360 KB                   |
| Model Weights     | 17.5M params | ~70 MB (float32)          |
| ROI (per defect)  | 128×128×3    | ~49 KB                    |

**Total Peak Memory:** ~150 MB (typical)

### Bottlenecks

1. **Feature Detection** (2-5s)
   - Solution: Reduce number of features or use faster matcher

2. **Model Inference** (0.5-1s per defect)
   - Solution: Use GPU, batch predictions, quantization

3. **Image I/O** (0.5-2s)
   - Solution: Use faster storage, parallel loading

---

## 7. Algorithm Details

### Otsu's Thresholding

**Purpose:** Find optimal threshold to separate foreground/background

**Algorithm:**
```
1. Calculate histogram of image
2. For each possible threshold t:
   a. Compute class probabilities w₀(t), w₁(t)
   b. Compute class means μ₀(t), μ₁(t)
   c. Compute between-class variance:
      σ²(t) = w₀(t) × w₁(t) × [μ₀(t) - μ₁(t)]²
3. Choose t that maximizes σ²(t)
```

**Pseudocode:**
```python
def otsu_threshold(image):
    histogram = compute_histogram(image)
    total_pixels = image.width * image.height
    
    best_threshold = 0
    best_variance = 0
    
    for t in range(256):
        # Compute class probabilities
        w0 = sum(histogram[0:t]) / total_pixels
        w1 = 1 - w0
        
        if w0 == 0 or w1 == 0:
            continue
        
        # Compute class means
        mu0 = sum(i * histogram[i] for i in range(t)) / (w0 * total_pixels)
        mu1 = sum(i * histogram[i] for i in range(t, 256)) / (w1 * total_pixels)
        
        # Compute variance
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        if variance > best_variance:
            best_variance = variance
            best_threshold = t
    
    return best_threshold
```

---

### ORB Feature Matching

**Purpose:** Align test image to template

**Algorithm:**
```
1. Detect keypoints using FAST
2. Compute ORB descriptors (256-bit binary)
3. Match descriptors using Hamming distance
4. Filter matches by distance threshold
5. Estimate homography using RANSAC
6. Warp test image using homography
```

**RANSAC for Homography:**
```python
def ransac_homography(matches, iterations=1000, threshold=5.0):
    best_H = None
    best_inliers = 0
    
    for i in range(iterations):
        # Randomly select 4 matches
        sample = random.sample(matches, 4)
        
        # Compute homography
        H = compute_homography(sample)
        
        # Count inliers
        inliers = count_inliers(H, matches, threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H
    
    return best_H
```

---

## 8. File Formats

### XML Annotation (Pascal VOC)

```xml
<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <folder>Missing_hole</folder>
  <filename>01_missing_hole_01.jpg</filename>
  <path>PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg</path>
  <source>
    <database>PCB Defect Dataset</database>
  </source>
  <size>
    <width>600</width>
    <height>600</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>Missing_hole</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>120</xmin>
      <ymin>150</ymin>
      <xmax>138</xmax>
      <ymax>170</ymax>
    </bndbox>
  </object>
</annotation>
```

### ROI Metadata (JSON)

```json
{
    "filename": "01_missing_hole_01_0.png",
    "original_image": "01_missing_hole_01.jpg",
    "class_name": "Missing_hole",
    "bbox": {
        "xmin": 120,
        "ymin": 150,
        "xmax": 138,
        "ymax": 170
    },
    "area": 360,
    "roi_size": [128, 128]
}
```

### Class Mapping (JSON)

```json
{
    "Missing_hole": 0,
    "Mouse_bite": 1,
    "Open_circuit": 2,
    "Short": 3,
    "Spur": 4,
    "Spurious_copper": 5
}
```

### Model Checkpoint (PyTorch)

```python
{
    'model_state_dict': OrderedDict([...]),
    'optimizer_state_dict': OrderedDict([...]),  # Optional
    'epoch': 50,
    'best_val_acc': 0.998
}
```

---

## 9. API Reference

### Detection API

```python
def detect_defects(test_path, template_path, min_area=100):
    """
    Detect defects in PCB image.
    
    Args:
        test_path (str): Path to test PCB image
        template_path (str): Path to template PCB image
        min_area (int): Minimum defect area in pixels
    
    Returns:
        tuple: (aligned_image, filtered_mask, defects_list)
        
    Example:
        aligned, filtered, defects = detect_defects(
            "test.jpg",
            "template.jpg",
            min_area=120
        )
    """
```

### Classification API

```python
def classify_defects(aligned_image, defects, model_path, class_mapping_path):
    """
    Classify detected defects.
    
    Args:
        aligned_image (np.ndarray): Aligned test image
        defects (list): List of defect dictionaries
        model_path (str): Path to trained model
        class_mapping_path (str): Path to class mapping JSON
    
    Returns:
        list: Classified defects with confidence scores
        
    Example:
        classified = classify_defects(
            aligned,
            defects,
            "model.pth",
            "class_mapping.json"
        )
    """
```

### Export API

```python
def generate_csv_report(defects, template_name, test_name, min_confidence):
    """
    Generate CSV report.
    
    Args:
        defects (list): Classified defects
        template_name (str): Template image filename
        test_name (str): Test image filename
        min_confidence (float): Confidence threshold
    
    Returns:
        str: CSV formatted string
    """

def generate_pdf_report(result_img, defects, template_name, test_name, min_confidence):
    """
    Generate PDF report.
    
    Args:
        result_img (np.ndarray): Annotated result image
        defects (list): Classified defects
        template_name (str): Template image filename
        test_name (str): Test image filename
        min_confidence (float): Confidence threshold
    
    Returns:
        bytes: PDF file content
    """
```

---

## 10. Error Handling

### Common Errors

**1. Model Not Found**
```python
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")
```

**2. Invalid Image**
```python
image = cv2.imread(path)
if image is None:
    raise ValueError(f"Could not load image: {path}")
```

**3. Alignment Failure**
```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
if H is None:
    logger.warning("Alignment failed, using original image")
    return test_image
```

**4. No Defects Found**
```python
if len(defects) == 0:
    logger.info("No defects detected")
    return aligned_image, [], []
```

---

## 11. Testing

### Unit Tests

```python
def test_xml_parser():
    annotation = parse_xml("test.xml")
    assert annotation['filename'] == "test.jpg"
    assert len(annotation['boxes']) > 0

def test_roi_extraction():
    image = cv2.imread("test.jpg")
    bbox = {'x': 100, 'y': 100, 'width': 20, 'height': 20}
    roi = crop_roi(image, bbox, target_size=128)
    assert roi.shape == (128, 128, 3)

def test_classification():
    model, device = load_model("model.pth")
    image = cv2.imread("defect.png")
    class_name, confidence = predict_defect(model, image, device, mapping)
    assert 0.0 <= confidence <= 1.0
```

### Integration Tests

```python
def test_full_pipeline():
    aligned, filtered, defects = detect_defects(
        "test.jpg",
        "template.jpg"
    )
    assert aligned is not None
    assert len(defects) >= 0
    
    if len(defects) > 0:
        classified = classify_defects(aligned, defects, "model.pth", "mapping.json")
        assert len(classified) == len(defects)
```

---

## 12. Deployment

### Production Considerations

**1. Model Serving**
- Use model caching
- Consider batch inference
- GPU acceleration for high throughput

**2. Scalability**
- Horizontal scaling with load balancer
- Async processing for multiple requests
- Queue system for batch jobs

**3. Monitoring**
- Log detection times
- Track accuracy metrics
- Monitor resource usage

**4. Security**
- Validate file uploads
- Sanitize inputs
- Rate limiting

---

## Conclusion

This technical documentation covers the complete PCB Defect Detection System, from data preparation through inference and web deployment. The system achieves 99.8% accuracy while maintaining fast inference times suitable for production use.

**Key Takeaways:**
- Modular architecture for maintainability
- Efficient algorithms for real-time performance
- Comprehensive error handling
- Production-ready web interface
- Extensible design for future improvements

---

*Last Updated: November 20, 2025*
*Version: 1.0.0*
