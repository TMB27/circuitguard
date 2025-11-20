# Simple XML parser for PCB defect annotations
import xml.etree.ElementTree as ET
import os


def parse_xml(xml_file):
    # Read the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image info
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get all bounding boxes
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        
        boxes.append({
            'class': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'boxes': boxes
    }


def load_all_annotations(folder_path):
    # Get all XML files from folder
    annotations = []
    for file in os.listdir(folder_path):
        if file.endswith('.xml'):
            xml_path = os.path.join(folder_path, file)
            data = parse_xml(xml_path)
            annotations.append(data)
    
    return annotations