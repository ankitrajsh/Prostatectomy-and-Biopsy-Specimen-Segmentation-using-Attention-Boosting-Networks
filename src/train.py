import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.segmentation_model import SemanticSegmentationModel
from dataset.segmentation_dataset import SegmentationDataset

def train_model(image_dir, mask_dir, num_classes=21, batch_size=4, num_epochs=50, learning_rate=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    dataset = SegmentationDataset(image_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SemanticSegmentationModel(encoder_type='resnet50', pretrained=True, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Resize target masks to match output dimensions
            resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=outputs.shape[2:], mode="nearest").long()

            # Compute loss
            loss = criterion(outputs, resized_masks.squeeze(1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print loss after each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

if __name__ == "__main__":
    image_dir = "path/to/images"  # Update with your image directory
    mask_dir = "path/to/masks"      # Update with your mask directory
    train_model(image_dir, mask_dir)