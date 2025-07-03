import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.segmentation_model import SemanticSegmentationModel
from dataset.segmentation_dataset import SegmentationDataset
import numpy as np
import os

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))  # Assuming masks have shape (N, H, W)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SemanticSegmentationModel(encoder_type='resnet50', pretrained=True, num_classes=21).to(device)
    
    # Define dataset and dataloader
    image_dir = "path/to/images"  # Update with your image directory
    mask_dir = "path/to/masks"     # Update with your mask directory
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Evaluate the model
    average_loss = evaluate_model(model, dataloader, device)
    print(f'Average Loss: {average_loss:.4f}')

if __name__ == "__main__":
    main()