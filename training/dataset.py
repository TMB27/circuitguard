import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json


class PCBDataset(Dataset):
    def __init__(self, data_folder, split='train', image_size=128):
        # data_folder = "data/splits"
        # split = "train", "val", or "test"
        self.image_size = image_size
        self.split = split
        
        # Load class mapping
        mapping_file = os.path.join(data_folder, 'class_mapping.json')
        with open(mapping_file, 'r') as f:
            self.class_mapping = json.load(f)
        
        # Load all image paths and labels
        self.images = []
        self.labels = []
        
        split_folder = os.path.join(data_folder, split)
        
        # Go through each class folder
        for class_name, class_id in self.class_mapping.items():
            class_folder = os.path.join(split_folder, class_name)
            
            if os.path.exists(class_folder):
                # Get all images in this class
                for img_file in os.listdir(class_folder):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(class_folder, img_file)
                        self.images.append(img_path)
                        self.labels.append(class_id)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to 0-1
        image = image / 255.0
        
        # Convert to tensor (channels first)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        
        # Get label
        label = self.labels[idx]
        
        return image, label


def create_dataloaders(data_folder, batch_size=32, image_size=128):
    # Create datasets
    train_dataset = PCBDataset(data_folder, split='train', image_size=image_size)
    val_dataset = PCBDataset(data_folder, split='val', image_size=image_size)
    test_dataset = PCBDataset(data_folder, split='test', image_size=image_size)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader