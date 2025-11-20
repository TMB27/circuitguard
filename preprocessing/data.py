# Main script to prepare PCB dataset
# This script extracts ROIs from images and splits them into train/val/test sets

import os
import sys

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_rois import batch_extract_rois
from split_dataset import load_all_rois, split_by_class, copy_files_to_split, create_class_mapping
import random


def main():
    print("="*60)
    print("PCB DATASET PREPARATION")
    print("="*60)
    
    # All defect types
    defect_types = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
    
    # STEP 1: Extract ROIs from all images
    print("\nSTEP 1: EXTRACTING ROIs FROM IMAGES")
    print("-"*60)
    
    for defect_type in defect_types:
        print(f"\nProcessing {defect_type}...")
        
        # Paths
        images_folder = f"../PCB_DATASET/images/{defect_type}"
        annotations_folder = f"../PCB_DATASET/Annotations/{defect_type}"
        output_folder = f"../data/rois/{defect_type}"
        
        # Extract ROIs if folders exist
        if os.path.exists(images_folder) and os.path.exists(annotations_folder):
            batch_extract_rois(images_folder, annotations_folder, output_folder)
        else:
            print(f"  Skipping - folders not found")
    
    # STEP 2: Split dataset into train/val/test
    print("\n" + "="*60)
    print("STEP 2: SPLITTING DATASET")
    print("-"*60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load all ROI files
    rois_folder = "../data/rois"
    splits_folder = "../data/splits"
    
    print("\nLoading all ROI files...")
    all_files = load_all_rois(rois_folder)
    print(f"Found {len(all_files)} ROI images")
    
    # Split into train/val/test (70/15/15)
    print("\nSplitting dataset...")
    train_files, val_files, test_files = split_by_class(all_files)
    
    print(f"\nSplit summary:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val:   {len(val_files)} images")
    print(f"  Test:  {len(test_files)} images")
    
    # Copy files to split folders
    print("\nCopying files to split folders...")
    copy_files_to_split(train_files, splits_folder, "train")
    copy_files_to_split(val_files, splits_folder, "val")
    copy_files_to_split(test_files, splits_folder, "test")
    
    # Create class mapping file
    create_class_mapping(all_files, splits_folder)
    
    # Done!
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nOutput locations:")
    print(f"  ROIs: {os.path.abspath(rois_folder)}")
    print(f"  Splits: {os.path.abspath(splits_folder)}")
    print(f"\nYou can now use the data for training!")


if __name__ == "__main__":
    main()
