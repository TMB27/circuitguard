# Simple PCB defect classifier - loads model and predicts defect types
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import cv2
import json
import os


class DefectClassifier(nn.Module):
    """Simple neural network for classifying defects"""
    
    def __init__(self, num_classes=6):
        super(DefectClassifier, self).__init__()
        
        # Load pretrained model
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Change last layer
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def load_model(model_path, num_classes=6):
    """Load the trained AI model"""
    # Use GPU if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = DefectClassifier(num_classes=num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device


def prepare_image(image, image_size=128):
    """Make image ready for AI model"""
    # Resize to 128x128
    image = cv2.resize(image, (image_size, image_size))
    
    # Convert to 0-1 range
    image = image / 255.0
    
    # Convert to PyTorch format
    image = torch.FloatTensor(image).permute(2, 0, 1)
    image = image.unsqueeze(0)
    
    return image


def predict_defect(model, image, device, class_mapping):
    """Use AI to predict what type of defect"""
    # Prepare image for model
    image_tensor = prepare_image(image).to(device)
    
    # Run prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = probabilities.max(1)
    
    # Get predicted class ID
    predicted_id = predicted_class.item()
    
    # Convert ID to class name
    id_to_class = {v: k for k, v in class_mapping.items()}
    class_name = id_to_class[predicted_id]
    
    return class_name, confidence.item()


def classify_defects(aligned_image, defects, model_path, class_mapping_path):
    """
    Classify each defect using AI model
    """
    # Load class mapping file
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Load trained model
    model, device = load_model(model_path, num_classes=len(class_mapping))
    
    # Check each defect
    results = []
    
    for i, defect in enumerate(defects):
        # Get defect position
        x = defect['x']
        y = defect['y']
        w = defect['width']
        h = defect['height']
        
        # Add some padding around defect
        padding = 5
        img_height, img_width = aligned_image.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_width, x + w + padding)
        y2 = min(img_height, y + h + padding)
        
        # Crop defect from image
        crop = aligned_image[y1:y2, x1:x2]
        
        # Ask AI model what type of defect
        class_name, confidence = predict_defect(model, crop, device, class_mapping)
        
        # Save result
        results.append({
            'defect_id': i + 1,
            'class': class_name,
            'confidence': confidence,
            'bbox': defect,
            'crop': crop
        })
        
        print(f"Defect {i+1}: {class_name} ({confidence:.0%})")
    
    return results


def draw_classified_defects(image, classified_defects, min_confidence=0.5):
    """Draw labeled boxes around classified defects"""
    result = image.copy()
    
    # Colors for different defect types
    colors = {
        'Missing_hole': (0, 0, 255),      # Red
        'Mouse_bite': (255, 0, 0),        # Blue
        'Open_circuit': (0, 255, 255),    # Yellow
        'Short': (255, 0, 255),           # Magenta
        'Spur': (0, 255, 0),              # Green
        'Spurious_copper': (255, 165, 0)  # Orange
    }
    
    for det in classified_defects:
        # Skip low confidence
        if det['confidence'] < min_confidence:
            continue
        
        bbox = det['bbox']
        x = bbox['x']
        y = bbox['y']
        w = bbox['width']
        h = bbox['height']
        
        # Get color
        color = colors.get(det['class'], (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        label = f"{det['class']}: {det['confidence']:.0%}"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x, y-20), (x+text_w, y), color, -1)
        
        # Text
        cv2.putText(result, label, (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result


# Simple test
if __name__ == "__main__":
    from detect_defects import detect_defects
    
    # File paths
    test_path = "../PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg"
    template_path = "../PCB_DATASET/PCB_USED/01.JPG"
    model_path = "../training/checkpoints/best_model.pth"
    class_mapping_path = "../data/splits/class_mapping.json"
    
    # Step 1: Find defects
    print("Finding defects...")
    aligned, filtered, defects = detect_defects(test_path, template_path, min_area=120)
    
    # Step 2: Classify defects
    print("Classifying defects...")
    classified = classify_defects(aligned, defects, model_path, class_mapping_path)
    
    # Step 3: Draw results
    print("Drawing results...")
    result = draw_classified_defects(aligned, classified, min_confidence=0.5)
    
    # Save
    cv2.imwrite("classified_defects.jpg", result)
    print("Done! Saved result.")
