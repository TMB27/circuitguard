# Simple web interface for PCB defect detection
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import csv
import io
from datetime import datetime
from detect_defects import detect_defects
from classify_defects import classify_defects, draw_classified_defects


# Cache model loading for performance
@st.cache_resource
def load_model_cached(model_path, class_mapping_path):
    """Load model once and cache it"""
    import torch
    import torch.nn as nn
    from efficientnet_pytorch import EfficientNet
    import json
    
    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Create model
    class DefectClassifier(nn.Module):
        def __init__(self, num_classes=6):
            super(DefectClassifier, self).__init__()
            self.model = EfficientNet.from_pretrained('efficientnet-b4')
            in_features = self.model._fc.in_features
            self.model._fc = nn.Linear(in_features, num_classes)
        
        def forward(self, x):
            return self.model(x)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DefectClassifier(num_classes=len(class_mapping))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device, class_mapping


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def generate_csv_report(defects, template_name, test_name, min_confidence):
    """Generate CSV log of defects"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow(['Template Image', template_name])
    writer.writerow(['Test Image', test_name])
    writer.writerow(['Confidence Threshold', f"{min_confidence:.2f}"])
    writer.writerow(['Total Defects', len(defects)])
    writer.writerow([])
    
    # Defect details
    writer.writerow(['ID', 'Type', 'Confidence', 'X', 'Y', 'Width', 'Height', 'Area'])
    for i, det in enumerate(defects, 1):
        bbox = det['bbox']
        writer.writerow([
            i,
            det['class'].replace('_', ' '),
            f"{det['confidence']:.2%}",
            bbox['x'],
            bbox['y'],
            bbox['width'],
            bbox['height'],
            bbox['area']
        ])
    
    return output.getvalue()


def generate_pdf_report(result_img, defects, template_name, test_name, min_confidence):
    """Generate simple PDF report"""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "PCB Defect Detection Report")
    
    # Info
    c.setFont("Helvetica", 12)
    y = height - 90
    c.drawString(50, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    c.drawString(50, y, f"Template: {template_name}")
    y -= 20
    c.drawString(50, y, f"Test Image: {test_name}")
    y -= 20
    c.drawString(50, y, f"Confidence Threshold: {min_confidence:.0%}")
    y -= 20
    c.drawString(50, y, f"Total Defects Found: {len(defects)}")
    y -= 40
    
    # Result image
    if result_img is not None:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Annotated Image:")
        y -= 20
        
        # Convert image
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb)
        
        # Scale image to fit page
        img_width = 500
        aspect = pil_img.height / pil_img.width
        img_height = img_width * aspect
        
        if img_height > 300:
            img_height = 300
            img_width = img_height / aspect
        
        img_reader = ImageReader(pil_img)
        c.drawImage(img_reader, 50, y - img_height, width=img_width, height=img_height)
        y -= (img_height + 30)
    
    # Defect summary table
    if y < 200:
        c.showPage()
        y = height - 50
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Defect Summary:")
    y -= 25
    
    c.setFont("Helvetica", 10)
    for i, det in enumerate(defects[:10], 1):  # Show first 10
        bbox = det['bbox']
        text = f"{i}. {det['class'].replace('_', ' ')} - {det['confidence']:.0%} at ({bbox['x']}, {bbox['y']}) [{bbox['width']}x{bbox['height']}px]"
        c.drawString(60, y, text)
        y -= 15
        
        if y < 50:
            c.showPage()
            y = height - 50
    
    if len(defects) > 10:
        c.drawString(60, y, f"... and {len(defects) - 10} more defects")
    
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def classify_defects_fast(aligned_image, defects, model, device, class_mapping):
    """Optimized classification using cached model"""
    import torch
    
    results = []
    
    for i, defect in enumerate(defects):
        # Crop defect region
        x = defect['x']
        y = defect['y']
        w = defect['width']
        h = defect['height']
        
        padding = 5
        img_height, img_width = aligned_image.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_width, x + w + padding)
        y2 = min(img_height, y + h + padding)
        
        crop = aligned_image[y1:y2, x1:x2]
        
        # Prepare image
        image = cv2.resize(crop, (128, 128))
        image = image / 255.0
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = probabilities.max(1)
        
        # Get class name
        predicted_id = predicted_class.item()
        id_to_class = {v: k for k, v in class_mapping.items()}
        class_name = id_to_class.get(predicted_id, "unknown")
        
        results.append({
            'defect_id': i + 1,
            'class': class_name,
            'confidence': confidence.item(),
            'bbox': defect,
            'crop': crop
        })
    
    return results


def main():
    st.set_page_config(page_title="PCB Defect Detector", page_icon="🔍", layout="wide")
    
    st.title("🔍 PCB Defect Detector")
    st.write("Upload a template (good PCB) and test PCB to find defects")
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        
        # Model paths
        model_path = st.text_input("Model Path", 
                                   "../training/checkpoints/best_model.pth")
        class_mapping_path = st.text_input("Class Mapping", 
                                          "../data/splits/class_mapping.json")
        
        # Detection settings
        min_area = st.slider("Minimum Defect Area (pixels)", 50, 500, 120)
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
        
        st.info("💡 Lower minimum area finds smaller defects")
    
    # Upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Template PCB (Good)")
        template_file = st.file_uploader("Upload template image", 
                                        type=['jpg', 'jpeg', 'png'],
                                        key="template")
        if template_file:
            template_img = Image.open(template_file)
            st.image(template_img, caption="Template PCB", use_container_width=True)
    
    with col2:
        st.subheader("Test PCB (Check for defects)")
        test_file = st.file_uploader("Upload test image", 
                                    type=['jpg', 'jpeg', 'png'],
                                    key="test")
        if test_file:
            test_img = Image.open(test_file)
            st.image(test_img, caption="Test PCB", use_container_width=True)
    
    # Run detection button
    if template_file and test_file:
        if st.button("🔍 Detect Defects", type="primary", use_container_width=True):
            
            # Progress bar
            progress = st.progress(0)
            status = st.empty()
            
            try:
                # Save uploaded files
                status.text("Saving images...")
                progress.progress(20)
                template_path = save_uploaded_file(template_file)
                test_path = save_uploaded_file(test_file)
                
                # Check if model exists
                if not os.path.exists(model_path):
                    st.error(f"Model not found: {model_path}")
                    return
                
                if not os.path.exists(class_mapping_path):
                    st.error(f"Class mapping not found: {class_mapping_path}")
                    return
                
                # Step 1: Detect defects
                status.text("Finding defects...")
                progress.progress(40)
                aligned, filtered, defects = detect_defects(test_path, template_path, min_area=min_area)
                
                if aligned is None:
                    st.error("Failed to process images")
                    return
                
                # Step 2: Classify defects (optimized with cached model)
                if len(defects) > 0:
                    status.text("Loading model...")
                    progress.progress(60)
                    model, device, class_mapping = load_model_cached(model_path, class_mapping_path)
                    
                    status.text("Classifying defects...")
                    progress.progress(75)
                    classified = classify_defects_fast(aligned, defects, model, device, class_mapping)
                    
                    # Filter by confidence
                    filtered_results = [d for d in classified if d['confidence'] >= min_confidence]
                    
                    # Draw results
                    status.text("Drawing results...")
                    progress.progress(90)
                    result_img = draw_classified_defects(aligned, classified, min_confidence=min_confidence)
                else:
                    classified = []
                    filtered_results = []
                    result_img = aligned
                
                # Complete
                progress.progress(100)
                status.text("Complete!")
                
                # Save to session state
                st.session_state['result_img'] = result_img
                st.session_state['aligned'] = aligned
                st.session_state['classified'] = classified
                st.session_state['filtered_results'] = filtered_results
                st.session_state['min_confidence'] = min_confidence
                st.session_state['template_name'] = template_file.name
                st.session_state['test_name'] = test_file.name
                
                # Clean up temp files
                os.remove(template_path)
                os.remove(test_path)
                
                # Clear progress
                progress.empty()
                status.empty()
                
                st.success(f"✅ Found {len(filtered_results)} defects (confidence ≥ {min_confidence:.0%})")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                progress.empty()
                status.empty()
    
    # Display results
    if 'result_img' in st.session_state:
        st.divider()
        st.subheader("Results")
        
        # Show annotated image
        result_rgb = cv2.cvtColor(st.session_state['result_img'], cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Detected Defects", width=600)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Defects Found", len(st.session_state['classified']))
        with col2:
            st.metric("High Confidence", len(st.session_state['filtered_results']))
        with col3:
            min_conf = st.session_state['min_confidence']
            st.metric("Confidence Threshold", f"{min_conf:.0%}")
        
        # Show defect list
        if len(st.session_state['filtered_results']) > 0:
            st.subheader("Defect Details")
            
            for i, det in enumerate(st.session_state['filtered_results'], 1):
                with st.expander(f"Defect #{i}: {det['class'].replace('_', ' ')} - {det['confidence']:.0%}"):
                    col_img, col_info = st.columns([1, 2])
                    
                    with col_img:
                        # Show cropped defect
                        crop_rgb = cv2.cvtColor(det['crop'], cv2.COLOR_BGR2RGB)
                        st.image(crop_rgb, caption="Defect Image", use_container_width=True)
                    
                    with col_info:
                        bbox = det['bbox']
                        st.write(f"**Type:** {det['class'].replace('_', ' ')}")
                        st.write(f"**Confidence:** {det['confidence']:.1%}")
                        st.write(f"**Position:** ({bbox['x']}, {bbox['y']})")
                        st.write(f"**Size:** {bbox['width']} × {bbox['height']} px")
                        st.write(f"**Area:** {bbox['area']} px²")
            
            # Export options
            st.divider()
            st.subheader("Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Annotated image
                _, buf = cv2.imencode('.jpg', st.session_state['result_img'])
                st.download_button(
                    label="📥 Annotated Image",
                    data=buf.tobytes(),
                    file_name="defect_detection_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            
            with col2:
                # CSV log
                csv_data = generate_csv_report(
                    st.session_state['filtered_results'],
                    st.session_state.get('template_name', 'template'),
                    st.session_state.get('test_name', 'test'),
                    st.session_state['min_confidence']
                )
                st.download_button(
                    label="📊 CSV Report",
                    data=csv_data,
                    file_name="defect_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # PDF report
                pdf_data = generate_pdf_report(
                    st.session_state['result_img'],
                    st.session_state['filtered_results'],
                    st.session_state.get('template_name', 'template'),
                    st.session_state.get('test_name', 'test'),
                    st.session_state['min_confidence']
                )
                st.download_button(
                    label="📄 PDF Report",
                    data=pdf_data,
                    file_name="defect_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("✅ No defects found above confidence threshold")
    
    else:
        st.info("👆 Upload both images and click 'Detect Defects' to get started")


if __name__ == "__main__":
    main()
