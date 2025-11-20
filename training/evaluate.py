# Simple evaluation script for PCB defect classifier
import torch
import json
import os
import numpy as np
from efficientnet_model import create_model
from dataset import create_dataloaders
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model_path, data_folder):
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load class mapping
    with open(os.path.join(data_folder, 'class_mapping.json'), 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    
    # Reverse mapping (id -> name)
    id_to_class = {v: k for k, v in class_mapping.items()}
    
    # Load data
    print("Loading test dataset...")
    _, _, test_loader = create_dataloaders(data_folder, batch_size=32)
    
    # Create model
    print("\nLoading model...")
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Collect predictions
    all_labels = []
    all_predictions = []
    
    print("\nRunning predictions...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Create results folder
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 1. Generate Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    # Add labels
    class_names = [id_to_class[i] for i in range(num_classes)]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to: results/confusion_matrix.png")
    plt.close()
    
    # 2. Generate Classification Report
    print("\nGenerating classification report...")
    report = classification_report(
        all_labels, 
        all_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    # Save as JSON
    with open('results/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("✓ Classification report saved to: results/classification_report.json")
    
    # Print report to console
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    # Print per-class metrics
    for class_name in class_names:
        metrics = report[class_name]
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1-score']:.3f}")
        print(f"  Support:   {int(metrics['support'])}")
    
    # Print overall metrics
    print("\n" + "-"*60)
    print(f"\nAccuracy: {report['accuracy']:.3f}")
    
    print(f"\nMacro Average:")
    print(f"  Precision: {report['macro avg']['precision']:.3f}")
    print(f"  Recall:    {report['macro avg']['recall']:.3f}")
    print(f"  F1-Score:  {report['macro avg']['f1-score']:.3f}")
    
    print(f"\nWeighted Average:")
    print(f"  Precision: {report['weighted avg']['precision']:.3f}")
    print(f"  Recall:    {report['weighted avg']['recall']:.3f}")
    print(f"  F1-Score:  {report['weighted avg']['f1-score']:.3f}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


# Run evaluation
if __name__ == "__main__":
    model_path = "checkpoints/best_model.pth"
    data_folder = "../data/splits"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running: python train.py")
    else:
        evaluate_model(model_path, data_folder)
