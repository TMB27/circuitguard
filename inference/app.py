import streamlit as st
import cv2
from PIL import Image
import tempfile
import os
from detect_defects import detect_defects
from classify_defects import classify_defects, draw_classified_defects
from export_reports import generate_csv_report, generate_pdf_report


# Simple function to save uploaded file
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def main():
    # Set page title and icon
    st.set_page_config(page_title="PCB Defect Detector", page_icon="🔍", layout="wide")
    
    st.title("🔍 PCB Defect Detector")
    st.write("Upload a template (good PCB) and test PCB to find defects")
    
    # Settings in sidebar
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model Path", "../training/checkpoints/best_model.pth")
    class_mapping_path = st.sidebar.text_input("Class Mapping", "../data/splits/class_mapping.json")
    min_area = st.sidebar.slider("Minimum Defect Area (pixels)", 50, 500, 120)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
    st.sidebar.info("💡 Lower minimum area finds smaller defects")
    
    # Create two columns for image upload
    col1, col2 = st.columns(2)
    
    # Column 1: Template image
    with col1:
        st.subheader("Template PCB (Good)")
        template_file = st.file_uploader("Upload template image", type=['jpg', 'jpeg', 'png'], key="template")
        if template_file:
            template_img = Image.open(template_file)
            st.image(template_img, caption="Template PCB", use_container_width=True)
    
    # Column 2: Test image
    with col2:
        st.subheader("Test PCB (Check for defects)")
        test_file = st.file_uploader("Upload test image", type=['jpg', 'jpeg', 'png'], key="test")
        if test_file:
            test_img = Image.open(test_file)
            st.image(test_img, caption="Test PCB", use_container_width=True)
    
    # Detect button - only show if both images are uploaded
    if template_file and test_file:
        if st.button("🔍 Detect Defects", type="primary", use_container_width=True):
            
            # Show progress
            status = st.empty()
            status.text("Processing images...")
            
            # Save uploaded files
            template_path = save_uploaded_file(template_file)
            test_path = save_uploaded_file(test_file)
            
            # Step 1: Detect defects
            status.text("Finding defects...")
            aligned, filtered, defects = detect_defects(test_path, template_path, min_area=min_area)
            
            # Step 2: Classify defects if found
            if len(defects) > 0:
                status.text("Classifying defects...")
                classified = classify_defects(aligned, defects, model_path, class_mapping_path)
                
                # Filter by confidence
                filtered_results = [d for d in classified if d['confidence'] >= min_confidence]
                
                # Draw results on image
                result_img = draw_classified_defects(aligned, classified, min_confidence=min_confidence)
            else:
                classified = []
                filtered_results = []
                result_img = aligned
            
            # Clean up temp files
            os.remove(template_path)
            os.remove(test_path)
            
            # Save results
            st.session_state['result_img'] = result_img
            st.session_state['classified'] = classified
            st.session_state['filtered_results'] = filtered_results
            st.session_state['min_confidence'] = min_confidence
            st.session_state['template_name'] = template_file.name
            st.session_state['test_name'] = test_file.name
            
            # Clear status and show success message
            status.empty()
            st.success(f"✅ Found {len(filtered_results)} defects!")
    
    # Show results if available
    if 'result_img' in st.session_state:
        st.divider()
        st.subheader("Results")
        
        # Show the image with detected defects
        result_rgb = cv2.cvtColor(st.session_state['result_img'], cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Detected Defects", width=600)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Defects", len(st.session_state['classified']))
        col2.metric("High Confidence", len(st.session_state['filtered_results']))
        col3.metric("Threshold", f"{st.session_state['min_confidence']:.0%}")
        
        # Show each defect details
        if len(st.session_state['filtered_results']) > 0:
            st.subheader("Defect Details")
            
            for i, det in enumerate(st.session_state['filtered_results'], 1):
                with st.expander(f"Defect #{i}: {det['class'].replace('_', ' ')} - {det['confidence']:.0%}"):
                    col_img, col_info = st.columns([1, 2])
                    
                    with col_img:
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
                # Download annotated image
                _, buf = cv2.imencode('.jpg', st.session_state['result_img'])
                st.download_button(
                    label="📥 Annotated Image",
                    data=buf.tobytes(),
                    file_name="defect_detection_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            
            with col2:
                # Download CSV report
                csv_data = generate_csv_report(
                    st.session_state['filtered_results'],
                    st.session_state.get('template_name', 'template.jpg'),
                    st.session_state.get('test_name', 'test.jpg'),
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
                # Download PDF report
                pdf_data = generate_pdf_report(
                    st.session_state['result_img'],
                    st.session_state['filtered_results'],
                    st.session_state.get('template_name', 'template.jpg'),
                    st.session_state.get('test_name', 'test.jpg'),
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
