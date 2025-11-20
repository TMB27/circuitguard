import csv
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import cv2


def generate_csv_report(defects, template_name, test_name, min_confidence):
    """Generate simple CSV report with all defect details"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Report header
    writer.writerow(['PCB DEFECT DETECTION REPORT'])
    writer.writerow([])
    writer.writerow(['Generated On:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow(['Template Image:', template_name])
    writer.writerow(['Test Image:', test_name])
    writer.writerow(['Confidence Threshold:', f"{min_confidence:.1%}"])
    writer.writerow(['Total Defects Found:', len(defects)])
    writer.writerow([])
    writer.writerow([])
    
    # Defect details
    writer.writerow(['DEFECT DETAILS'])
    writer.writerow(['ID', 'Defect Type', 'Confidence', 'X Position', 'Y Position', 'Width (px)', 'Height (px)', 'Area (px²)', 'Status'])
    
    for i, det in enumerate(defects, 1):
        bbox = det['bbox']
        confidence = det['confidence']
        
        # Determine status
        if confidence >= 0.9:
            status = 'Critical'
        elif confidence >= 0.7:
            status = 'High Priority'
        else:
            status = 'Medium Priority'
        
        writer.writerow([
            i,
            det['class'].replace('_', ' ').title(),
            f"{confidence:.1%}",
            bbox['x'],
            bbox['y'],
            bbox['width'],
            bbox['height'],
            bbox['area'],
            status
        ])
    
    # Summary statistics
    writer.writerow([])
    writer.writerow([])
    writer.writerow(['SUMMARY'])
    
    # Count by type
    defect_types = {}
    for det in defects:
        defect_type = det['class'].replace('_', ' ').title()
        defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
    
    writer.writerow(['Defect Type', 'Count'])
    for defect_type, count in defect_types.items():
        writer.writerow([defect_type, count])
    
    # Confidence breakdown
    writer.writerow([])
    writer.writerow(['Confidence Level', 'Count'])
    critical = sum(1 for d in defects if d['confidence'] >= 0.9)
    high = sum(1 for d in defects if 0.7 <= d['confidence'] < 0.9)
    medium = sum(1 for d in defects if d['confidence'] < 0.7)
    writer.writerow(['Critical (≥90%)', critical])
    writer.writerow(['High Priority (70-89%)', high])
    writer.writerow(['Medium Priority (<70%)', medium])
    
    return output.getvalue()


def generate_pdf_report(result_img, defects, template_name, test_name, min_confidence):
    """Generate detailed PDF report with images and analysis"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Page 1 - Cover and Summary
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 60, "PCB DEFECT DETECTION REPORT")
    
    # Report info
    c.setFont("Helvetica", 12)
    y = height - 120
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}")
    y -= 30
    c.drawString(50, y, f"Template Image: {template_name}")
    y -= 20
    c.drawString(50, y, f"Test Image: {test_name}")
    y -= 20
    c.drawString(50, y, f"Confidence Threshold: {min_confidence:.0%}")
    y -= 40
    
    # Summary box
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "INSPECTION SUMMARY")
    y -= 30
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(70, y, f"Total Defects Found: {len(defects)}")
    y -= 25
    
    # Count by priority
    critical = sum(1 for d in defects if d['confidence'] >= 0.9)
    high = sum(1 for d in defects if 0.7 <= d['confidence'] < 0.9)
    medium = sum(1 for d in defects if d['confidence'] < 0.7)
    
    c.setFont("Helvetica", 12)
    c.drawString(70, y, f"Critical Priority (≥90%): {critical}")
    y -= 20
    c.drawString(70, y, f"High Priority (70-89%): {high}")
    y -= 20
    c.drawString(70, y, f"Medium Priority (<70%): {medium}")
    y -= 35
    
    # Count by type
    c.setFont("Helvetica-Bold", 14)
    c.drawString(70, y, "Defects by Type:")
    y -= 20
    
    defect_types = {}
    for det in defects:
        defect_type = det['class'].replace('_', ' ').title()
        defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
    
    c.setFont("Helvetica", 11)
    for defect_type, count in defect_types.items():
        c.drawString(90, y, f"• {defect_type}: {count}")
        y -= 18
    
    y -= 20
    
    # Annotated image
    if result_img is not None:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "ANNOTATED IMAGE")
        y -= 25
        
        # Convert image
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb)
        
        # Scale to fit
        img_width = 500
        aspect = pil_img.height / pil_img.width
        img_height = img_width * aspect
        
        if img_height > 350:
            img_height = 350
            img_width = img_height / aspect
        
        img_reader = ImageReader(pil_img)
        c.drawImage(img_reader, 50, y - img_height, width=img_width, height=img_height)
        y -= (img_height + 20)
    
    # New page for detailed list
    c.showPage()
    y = height - 50
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "DETAILED DEFECT LIST")
    y -= 35
    
    # Table header
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "ID")
    c.drawString(90, y, "Type")
    c.drawString(220, y, "Confidence")
    c.drawString(320, y, "Position")
    c.drawString(420, y, "Size")
    c.drawString(500, y, "Priority")
    y -= 5
    c.line(50, y, 560, y)
    y -= 20
    
    # List all defects
    c.setFont("Helvetica", 9)
    for i, det in enumerate(defects, 1):
        bbox = det['bbox']
        confidence = det['confidence']
        
        # Priority
        if confidence >= 0.9:
            priority = 'Critical'
        elif confidence >= 0.7:
            priority = 'High'
        else:
            priority = 'Medium'
        
        # Write row
        c.drawString(50, y, f"{i}")
        c.drawString(90, y, det['class'].replace('_', ' ').title())
        c.drawString(220, y, f"{confidence:.1%}")
        c.drawString(320, y, f"({bbox['x']}, {bbox['y']})")
        c.drawString(420, y, f"{bbox['width']}×{bbox['height']}px")
        c.drawString(500, y, priority)
        y -= 18
        
        # New page if needed
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y, "ID")
            c.drawString(90, y, "Type")
            c.drawString(220, y, "Confidence")
            c.drawString(320, y, "Position")
            c.drawString(420, y, "Size")
            c.drawString(500, y, "Priority")
            y -= 5
            c.line(50, y, 560, y)
            y -= 20
            c.setFont("Helvetica", 9)
    
    # Footer on last page
    y -= 20
    if y < 100:
        c.showPage()
        y = height - 50
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "RECOMMENDATIONS")
    y -= 25
    c.setFont("Helvetica", 10)
    
    if critical > 0:
        c.drawString(70, y, f"• {critical} critical defect(s) require immediate attention")
        y -= 18
    
    if high > 0:
        c.drawString(70, y, f"• {high} high priority defect(s) should be reviewed")
        y -= 18
    
    if len(defects) == 0:
        c.drawString(70, y, "• No defects found - PCB passed inspection")
        y -= 18
    else:
        c.drawString(70, y, "• Review all defects and determine corrective actions")
        y -= 18
        c.drawString(70, y, "• Document findings and update manufacturing process")
    
    # Page numbers and date on all pages
    for page_num in range(1, c.getPageNumber() + 1):
        c.setFont("Helvetica", 8)
        c.drawString(500, 30, f"Page {page_num}")
        c.drawString(50, 30, f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
    
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
