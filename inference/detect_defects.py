# Simple PCB defect detection - combines all preprocessing steps
import cv2
import numpy as np
import os


def align_images(test_img, template_img):
    """Align test image to template using feature matching"""
    # Convert to grayscale
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    # Find features using ORB
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(test_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)
    
    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Get best 100 matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
    
    # Calculate transformation matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Apply transformation
    height, width = template_img.shape[:2]
    aligned_img = cv2.warpPerspective(test_img, H, (width, height))
    
    return aligned_img


def subtract_images(test_img, template_img):
    """Subtract template from test to find differences"""
    # Resize to same size
    height, width = template_img.shape[:2]
    test_resized = cv2.resize(test_img, (width, height))
    
    # Align images
    test_aligned = align_images(test_resized, template_img)
    
    # Convert to grayscale
    test_gray = cv2.cvtColor(test_aligned, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    # Get difference
    diff = cv2.absdiff(test_gray, template_gray)
    
    return diff, test_aligned


def threshold_image(diff_img):
    """Convert difference to black and white"""
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(diff_img, (5, 5), 0)
    
    # Threshold to binary
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def filter_noise(binary_img):
    """Remove small noise and enhance defects"""
    # Create kernel for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Remove noise (opening)
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    
    # Enhance defects (dilation)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    enhanced = cv2.dilate(opened, kernel2, iterations=2)
    
    return enhanced


def find_defects(filtered_img, min_area=100):
    """Find defect regions"""
    # Find contours
    contours, _ = cv2.findContours(filtered_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    defects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            defects.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': int(area)
            })
    
    return defects


def crop_defect(image, defect, target_size=128, padding=5):
    """Crop defect region from image"""
    x = defect['x']
    y = defect['y']
    w = defect['width']
    h = defect['height']
    
    img_height, img_width = image.shape[:2]
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_width, x + w + padding)
    y2 = min(img_height, y + h + padding)
    
    # Crop
    roi = image[y1:y2, x1:x2]
    
    # Resize
    roi = cv2.resize(roi, (target_size, target_size))
    
    return roi


def detect_defects(test_path, template_path, min_area=100):
    """
    Main function to detect defects in PCB
    
    Steps:
    1. Load images
    2. Subtract template from test
    3. Convert to black and white
    4. Remove noise
    5. Find defect areas
    """
    # Load images
    test_img = cv2.imread(test_path)
    template_img = cv2.imread(template_path)
    
    # Subtract template from test
    diff, test_aligned = subtract_images(test_img, template_img)
    
    # Convert to binary (black and white)
    binary = threshold_image(diff)
    
    # Remove small noise
    filtered = filter_noise(binary)
    
    # Find defect regions
    defects = find_defects(filtered, min_area=min_area)
    
    print(f"Found {len(defects)} defects")
    
    return test_aligned, filtered, defects


def draw_defects(image, defects):
    """Draw boxes around defects"""
    result = image.copy()
    
    for defect in defects:
        x = defect['x']
        y = defect['y']
        w = defect['width']
        h = defect['height']
        
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw label
        label = f"Area: {defect['area']}"
        cv2.putText(result, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return result


# Simple test
if __name__ == "__main__":
    test_path = "../PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg"
    template_path = "../PCB_DATASET/PCB_USED/01.JPG"
    
    # Run detection
    aligned, filtered, defects = detect_defects(test_path, template_path, min_area=120)
    
    # Draw boxes around defects
    result = draw_defects(aligned, defects)
    
    # Save result
    cv2.imwrite("detected_defects.jpg", result)
    print("Saved result!")
