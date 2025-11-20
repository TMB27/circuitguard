# Simple ROI extraction for cropping defect regions from PCB images
import cv2
import os
import json
from xml_parser import parse_xml


def crop_roi(image, xmin, ymin, xmax, ymax, target_size=128):
    # Crop the region from image
    roi = image[ymin:ymax, xmin:xmax]
    
    # Resize to target size
    roi = cv2.resize(roi, (target_size, target_size))
    
    return roi


def extract_rois_from_image(image_path, xml_path, output_folder, target_size=128):
    # Load image
    image = cv2.imread(image_path)
    
    # Parse XML annotation
    annotation = parse_xml(xml_path)
    
    # Create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # List to store metadata
    metadata = []
    
    # Extract each defect region
    for i, bbox in enumerate(annotation['boxes']):
        # Crop ROI
        roi = crop_roi(image, bbox['xmin'], bbox['ymin'], 
                      bbox['xmax'], bbox['ymax'], target_size)
        
        # Save ROI with 3-digit numbering
        roi_filename = f"{base_name}_roi_{i:03d}_{bbox['class']}.png"
        roi_path = os.path.join(output_folder, roi_filename)
        cv2.imwrite(roi_path, roi)
        
        # Calculate area
        area = (bbox['xmax'] - bbox['xmin']) * (bbox['ymax'] - bbox['ymin'])
        
        # Add metadata entry
        metadata.append({
            "filename": roi_filename,
            "roi_id": i,
            "class_name": bbox['class'],
            "bbox": {
                "xmin": bbox['xmin'],
                "ymin": bbox['ymin'],
                "xmax": bbox['xmax'],
                "ymax": bbox['ymax']
            },
            "area": area,
            "source_image": annotation['filename']
        })
        
        print(f"Saved: {roi_filename}")
    
    # Save metadata JSON file
    metadata_filename = f"{base_name}_metadata.json"
    metadata_path = os.path.join(output_folder, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return len(annotation['boxes'])


def batch_extract_rois(images_folder, annotations_folder, output_folder, target_size=128):
    # Get all image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    
    total_rois = 0
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        xml_file = img_file.replace('.jpg', '.xml')
        xml_path = os.path.join(annotations_folder, xml_file)
        
        if os.path.exists(xml_path):
            num_rois = extract_rois_from_image(img_path, xml_path, 
                                              output_folder, target_size)
            total_rois += num_rois
            print(f"Processed {img_file}: {num_rois} ROIs\n")
    
    print(f"Total ROIs extracted: {total_rois}")