# User Guide: PCB Defect Detection System

## Table of Contents
1. [Getting Started](#getting-started)
2. [Using the Web Interface](#using-the-web-interface)
3. [Understanding Results](#understanding-results)
4. [Export and Reporting](#export-and-reporting)
5. [Advanced Usage](#advanced-usage)
6. [Best Practices](#best-practices)
7. [Common Scenarios](#common-scenarios)
8. [FAQ](#faq)

---

## Getting Started

### First Time Setup

1. **Install Python** (if not already installed)
   - Download from https://www.python.org/
   - Version 3.9 or higher required
   - Check: `python --version`

2. **Install Dependencies**
   ```bash
   cd Circuit_Guard_New
   pip install -r requirements.txt
   ```
   
   Wait 5-10 minutes for installation to complete.

3. **Verify Installation**
   ```bash
   python -c "import torch, cv2, streamlit; print('Ready!')"
   ```
   
   Should print "Ready!" without errors.

### Launch the Application

**Web Interface**
```bash
cd inference
streamlit run app.py
```

Your browser will open automatically at http://localhost:8501

---

## Using the Web Interface

### Step-by-Step Guide

#### 1. Open the Application

After running `streamlit run app.py`, you'll see the PCB Defect Detector interface.

#### 2. Configure Settings (Sidebar)

**Model Path:**
- Default: `../training/checkpoints/best_model.pth`
- Only change if you have a custom trained model

**Class Mapping:**
- Default: `../data/splits/class_mapping.json`
- Maps defect IDs to names

**Minimum Defect Area:**
- Slider: 50-500 pixels
- **Lower (50-100)**: Detects smaller defects, may include noise
- **Default (120)**: Balanced detection
- **Higher (200-500)**: Only large defects, cleaner results

**Minimum Confidence:**
- Slider: 0.0-1.0
- **0.5 (default)**: Show moderately confident predictions
- **0.7-0.8**: High confidence only
- **0.9+**: Very strict, most reliable

#### 3. Upload Images

**Template PCB (Left Side):**
- Upload a defect-free reference board
- This is your *Template Sample*
- Must be same PCB design as test board
- Format: JPG, JPEG, or PNG

**Test PCB (Right Side):**
- Upload the board you want to inspect
- Should have similar lighting/angle as template
- Format: JPG, JPEG, or PNG

**Tips:**
- Use clear, well-lit images
- Avoid shadows or glare
- Keep camera angle consistent
- Higher resolution = better detection

#### 4. Run Detection

Click the blue **"Detect Defects"** button.

Progress indicators show:
1. Saving images... (20%)
2. Finding defects... (40%)
3. Loading model... (60%)
4. Classifying defects... (75%)
5. Drawing results... (90%)
6. Complete! (100%)

Processing time: 5-30 seconds depending on image size and hardware.

#### 5. View Results

**Annotated Image:**
- Shows test board with colored boxes
- Each defect type has unique color
- Labels show class name and confidence

**Metrics Cards:**
- Total Defects Found
- High Confidence Count
- Confidence Threshold Used

**Defect Details:**
- Click each expander to see:
  - Cropped defect image
  - Type (e.g., "Missing hole")
  - Confidence percentage
  - Position coordinates (X, Y)
  - Size (Width × Height)
  - Area in pixels

---

## Understanding Results

### Defect Types Explained

**1. Missing Hole**
- Expected hole is not present
- Common in drilling errors
- Color: Red

**2. Mouse Bite**
- Irregular hole edge (looks like bite marks)
- Caused by routing issues
- Color: Blue

**3. Open Circuit**
- Break in copper trace
- No electrical connection
- Color: Yellow

**4. Short Circuit**
- Unwanted connection between traces
- Causes electrical malfunction
- Color: Magenta

**5. Spur**
- Extra copper protruding from trace
- Sharp edge defect
- Color: Green

**6. Spurious Copper**
- Unwanted copper in wrong location
- Should have been etched away
- Color: Orange

### Confidence Scores

**95-100%**: Very reliable, definitely a defect
**80-94%**: High confidence, likely a defect
**65-79%**: Medium confidence, should verify
**50-64%**: Lower confidence, may be false positive
**Below 50%**: Low confidence, probably noise (filtered out by default)

### Reading the Annotated Image

Each defect box shows:
- **Color**: Defect type (see legend above)
- **Label**: "Type: XX%" (e.g., "Short: 88%")
- **Box**: Defect location and size

Multiple defects may overlap if close together.

---

## Export and Reporting

### 1. Annotated Image (JPG)

**What it includes:**
- Original test board image
- Colored bounding boxes around defects
- Class labels with confidence
- High resolution (original image size)

**Use cases:**
- Quick visual reference
- Include in presentations
- Share with team
- Archive for records

**How to use:**
1. Click "📥 Annotated Image" button
2. Choose save location
3. File saves as `defect_detection_result.jpg`

---

### 2. CSV Report

**What it includes:**
```csv
Timestamp, 2025-11-20 14:30:00
Template Image, template_pcb.jpg
Test Image, board_123.jpg
Confidence Threshold, 0.50
Total Defects, 5

ID, Type, Confidence, X, Y, Width, Height, Area
1, Missing hole, 95%, 120, 150, 18, 20, 360
2, Short, 88%, 250, 200, 25, 22, 550
3, Spur, 75%, 180, 300, 15, 18, 270
4, Open circuit, 92%, 300, 180, 20, 15, 300
5, Spurious copper, 85%, 400, 220, 22, 24, 528
```

**Use cases:**
- Import into Excel/spreadsheet
- Statistical analysis
- Database storage
- Quality tracking over time
- Automated processing

**How to analyze:**
1. Open in Excel/Google Sheets
2. Filter by defect type
3. Sort by confidence
4. Create charts/graphs
5. Calculate defect rates

---

### 3. PDF Report

**What it includes:**
- **Cover Page**: Title, date, summary
- **Inspection Details**: 
  - Template and test image names
  - Confidence threshold
  - Total defects found
- **Annotated Image**: Visual results
- **Defect Table**: Detailed list (up to 10 defects shown)
- **Footer**: Report metadata

**Use cases:**
- Professional documentation
- Share with clients/stakeholders
- Quality audit records
- Compliance documentation
- Management reports

**Report sections:**

*Page 1:*
- Header: "PCB Defect Detection Report"
- Summary table
- Annotated image (scaled to fit)

*Page 2 (if needed):*
- Defect details
- Position and size information
- Confidence scores

**How to use:**
1. Click "📄 PDF Report" button
2. Save as `defect_report.pdf`
3. Open in any PDF reader
4. Print or email as needed

---

## Advanced Usage

### Command Line Detection

For automation or batch processing:

```bash
cd inference
python detect_defects.py
```

Customize parameters:
```python
from detect_defects import detect_defects

aligned, filtered, defects = detect_defects(
    test_path="boards/test_01.jpg",
    template_path="templates/template_A.jpg",
    min_area=150  # Adjust threshold
)

print(f"Found {len(defects)} defects")
```

### Batch Processing

Process multiple boards:

```python
import os
import sys
sys.path.append('inference')
from detect_defects import detect_defects
from classify_defects import classify_defects
import cv2

template = "template.jpg"
test_folder = "test_boards/"
output_folder = "results/"

for test_file in os.listdir(test_folder):
    if test_file.endswith('.jpg'):
        print(f"Processing {test_file}...")
        
        # Detect
        test_path = os.path.join(test_folder, test_file)
        aligned, filtered, defects = detect_defects(test_path, template, min_area=120)
        
        # Classify
        if len(defects) > 0:
            classified = classify_defects(
                aligned, defects, 
                "model.pth", "mapping.json"
            )
            result = draw_classified_defects(aligned, classified, 0.5)
            
            # Save
            output_path = os.path.join(output_folder, f"result_{test_file}")
            cv2.imwrite(output_path, result)
            
        print(f"✓ {test_file}: {len(defects)} defects")
```

### Custom Confidence Thresholds

Different thresholds for different defect types:

```python
classified = classify_defects(aligned, defects, model_path, mapping_path)

# Strict for critical defects
critical_defects = [d for d in classified 
                   if d['class'] in ['Open_circuit', 'Short'] 
                   and d['confidence'] >= 0.8]

# Lenient for cosmetic issues
cosmetic_defects = [d for d in classified 
                   if d['class'] in ['Spur', 'Mouse_bite'] 
                   and d['confidence'] >= 0.6]
```

---

## Best Practices

### Image Quality

**DO:**
✅ Use consistent lighting
✅ Keep camera angle perpendicular
✅ Use high resolution (at least 600×600)
✅ Clean board before imaging
✅ Use same camera for template and test

**DON'T:**
❌ Mix different lighting conditions
❌ Use blurry or out-of-focus images
❌ Include shadows or reflections
❌ Rotate or skew boards
❌ Use compressed/low-quality images

### Template Selection

**Good Template:**
- 100% defect-free board
- Same PCB design revision
- Clear, high-quality image
- Representative of "golden sample"

**Bad Template:**
- Has any defects (even minor)
- Different PCB revision
- Poor image quality
- Different angle or lighting

### Settings Optimization

**For Small Components:**
- Minimum area: 80-100 pixels
- Confidence: 0.6-0.7

**For General Inspection:**
- Minimum area: 120 pixels (default)
- Confidence: 0.5

**For Large Components Only:**
- Minimum area: 200-300 pixels
- Confidence: 0.5-0.6

**For High Precision:**
- Minimum area: 100 pixels
- Confidence: 0.8-0.9

### Workflow Tips

1. **Start with defaults** - Only adjust if needed
2. **Check false positives** - Raise confidence if too many
3. **Check false negatives** - Lower min_area if missing defects
4. **Save good templates** - Keep library of defect-free boards
5. **Document settings** - Note what works for each board type

---

## Common Scenarios

### Scenario 1: Mass Production QC

**Goal:** Inspect 100 boards/day

**Setup:**
- Use fixed camera rig
- Standard lighting
- Same template for batch
- Automate with Python script

**Settings:**
- Min area: 120
- Confidence: 0.7 (stricter)
- Save CSV logs for tracking

**Process:**
1. Capture image of each board
2. Run batch script
3. Review flagged boards
4. Generate daily summary report

---

### Scenario 2: R&D Prototype Testing

**Goal:** Find all possible defects

**Setup:**
- Manual inspection
- Web interface
- Multiple angles if needed

**Settings:**
- Min area: 80 (find smaller defects)
- Confidence: 0.5 (show all possibilities)

**Process:**
1. Upload template and test
2. Review all detections carefully
3. Manually verify each defect
4. Export PDF for documentation

---

### Scenario 3: Customer Return Analysis

**Goal:** Identify failure cause

**Setup:**
- Careful image capture
- Compare to known-good board
- Multiple inspections

**Settings:**
- Min area: 100
- Confidence: 0.6

**Process:**
1. Clean returned board
2. Capture high-res image
3. Run detection multiple times
4. Focus on high-confidence defects
5. Generate detailed PDF report
6. Include in RMA documentation

---

### Scenario 4: Supplier Audit

**Goal:** Verify incoming board quality

**Setup:**
- Random sampling
- Standard inspection protocol
- Track statistics

**Settings:**
- Min area: 120
- Confidence: 0.7

**Process:**
1. Select random samples (e.g., 10 per batch)
2. Run inspection on each
3. Export CSV logs
4. Calculate defect rate
5. Pass/fail based on threshold
6. Feedback to supplier

---

## FAQ

### General Questions

**Q: Do I need a GPU?**
A: No, but it's faster. CPU works fine for inference.

**Q: How long does detection take?**
A: 5-30 seconds depending on image size and hardware.

**Q: Can I use my own images?**
A: Yes! Any PCB images with matching template and test boards.

**Q: What image formats are supported?**
A: JPG, JPEG, PNG

**Q: Can I inspect boards in production?**
A: Yes! Set up fixed camera and automate with Python script.

---

### Technical Questions

**Q: How accurate is the detection?**
A: 99.8% accuracy on our test set. Your results may vary based on image quality.

**Q: What model is used?**
A: EfficientNet-B4 with ~17.5M parameters, pretrained on ImageNet.

**Q: Can I train on my own data?**
A: Yes! See Training Pipeline documentation.

**Q: How much data is needed for training?**
A: Minimum 50-100 examples per defect class. More is better.

**Q: Can I add new defect types?**
A: Yes, retrain model with new classes and update class mapping.

---

### Troubleshooting

**Q: "Model not found" error**
A: Check the model path in settings or train a model first.

**Q: No defects detected**
A: Try lowering min_area or check if template/test are actually different.

**Q: Too many false detections**
A: Raise min_area or confidence threshold.

**Q: Slow performance**
A: Model caches after first load. Close other apps, use GPU if available.

**Q: Images won't align**
A: Ensure template and test are similar orientation. Try manual pre-alignment.

---

### Export Questions

**Q: Can I customize the PDF report?**
A: Currently uses standard template. Edit `app.py` to customize.

**Q: What CSV format is used?**
A: Standard CSV, opens in Excel/Google Sheets/any spreadsheet software.

**Q: Can I export raw data?**
A: Yes, CSV includes all detection data. Use Python API for more control.

**Q: How do I automate exports?**
A: Use Python API to generate reports programmatically.

---

## Getting Help

**Documentation:**
- README.md - Project overview
- USER_GUIDE.md - This guide
- Code comments - Inline documentation

**Support Channels:**
- GitHub Issues: Report bugs or request features
- Email: support@example.com
- Documentation: Check README first

**Community:**
- Star the repo if helpful
- Contribute improvements
- Share your use cases

---

## Version Control & Git

### Using Git with This Project

The project is configured with `.gitignore` to prevent committing large files:

**Ignored (not committed):**
- Dataset images and annotations
- Trained model checkpoints
- Generated results
- Python cache files
- Virtual environments

**Tracked (committed):**
- Source code
- Documentation
- Configuration files
- Project structure

### Typical Workflow

```bash
# Check what changed
git status

# Stage specific files
git add filename.py

# Commit with message
git commit -m "Fix: Improved detection accuracy"

# Push to GitHub
git push origin main
```

### Working with Models

Since model files (`.pth`) are large and ignored by git:

**Option 1: GitHub Releases**
```bash
# After training a good model
# Create a release on GitHub and attach the .pth file
```

**Option 2: External Storage**
- Use Google Drive, Dropbox, or AWS S3
- Share download link in README
- Download manually to `training/checkpoints/`

**Option 3: Git LFS (for teams)**
```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"

# Commit normally
git add training/checkpoints/best_model.pth
git commit -m "Add trained model"
```

---

## Appendix

### Keyboard Shortcuts (Streamlit)

- `Ctrl+R` - Rerun app
- `Ctrl+S` - Save screenshot
- `Esc` - Close sidebar

### File Locations

- Models: `training/checkpoints/`
- Results: `training/results/`
- Preprocessed data: `data/splits/`
- Original dataset: `PCB_DATASET/`
- Logs: Streamlit logs in terminal
- Temp files: System temp folder (auto-cleaned)

### Performance Benchmarks

**Detection Speed** (on test hardware):
- Image alignment: 2-5 seconds
- Preprocessing: 1-2 seconds
- Model inference: 0.5-1 second per defect
- Total: 5-15 seconds typical

**Hardware tested:**
- CPU: Intel i5/i7
- RAM: 8GB
- GPU: NVIDIA GTX 1060 (optional)

### Color Codes

Defect type colors in annotated images:
- Missing_hole: `RGB(0, 0, 255)` - Red
- Mouse_bite: `RGB(255, 0, 0)` - Blue  
- Open_circuit: `RGB(0, 255, 255)` - Yellow
- Short: `RGB(255, 0, 255)` - Magenta
- Spur: `RGB(0, 255, 0)` - Green
- Spurious_copper: `RGB(255, 165, 0)` - Orange

---

**End of User Guide**

*Last updated: November 20, 2025*
*Version: 1.0.0*
