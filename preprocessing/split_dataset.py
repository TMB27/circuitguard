# Simple dataset splitter for train/val/test splits
import json
import shutil
import os
import random


def load_all_rois(rois_folder):
    # Load all ROI images from data/rois folder
    all_files = []
    
    # Go through each defect type folder
    for defect_type in os.listdir(rois_folder):
        defect_path = os.path.join(rois_folder, defect_type)
        
        if os.path.isdir(defect_path):
            # Get all PNG files (ROI images)
            for file in os.listdir(defect_path):
                if file.endswith('.png'):
                    all_files.append({
                        'file_path': os.path.join(defect_path, file),
                        'filename': file,
                        'class': defect_type
                    })
    
    return all_files


def split_by_class(all_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Group files by class
    class_files = {}
    for file_info in all_files:
        class_name = file_info['class']
        if class_name not in class_files:
            class_files[class_name] = []
        class_files[class_name].append(file_info)
    
    train_files = []
    val_files = []
    test_files = []
    
    # Split each class separately
    for class_name, files in class_files.items():
        # Shuffle files
        random.shuffle(files)
        
        # Calculate split points
        total = len(files)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        # Split files
        train_files.extend(files[:train_end])
        val_files.extend(files[train_end:val_end])
        test_files.extend(files[val_end:])
        
        print(f"{class_name}: {total} files -> Train: {train_end}, Val: {val_end - train_end}, Test: {total - val_end}")
    
    return train_files, val_files, test_files


def copy_files_to_split(files, output_folder, split_name):
    # Copy files to train/val/test folders
    for file_info in files:
        class_name = file_info['class']
        
        # Create output folder
        output_path = os.path.join(output_folder, split_name, class_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Copy file
        src = file_info['file_path']
        dst = os.path.join(output_path, file_info['filename'])
        shutil.copy2(src, dst)
    
    print(f"Copied {len(files)} files to {split_name}/")


def create_class_mapping(all_files, output_folder):
    # Create class to number mapping
    classes = sorted(set(f['class'] for f in all_files))
    class_mapping = {class_name: i for i, class_name in enumerate(classes)}
    
    # Create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save to JSON
    mapping_file = os.path.join(output_folder, 'class_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"\nClass mapping: {class_mapping}")
    return class_mapping