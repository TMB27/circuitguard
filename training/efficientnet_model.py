# Simple EfficientNet model for PCB defect classification
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class PCBModel(nn.Module):
    def __init__(self, num_classes=6):
        super(PCBModel, self).__init__()
        
        # Load pretrained EfficientNet-B4
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Change last layer for our classes
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.model(x)


def create_model(num_classes=6):
    # Create model
    model = PCBModel(num_classes=num_classes)
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    print(f"Device: {device}")
    
    return model


# Test model
if __name__ == "__main__":
    print("Creating model...")
    model = create_model(num_classes=6)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 128, 128)
    
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model works!")
