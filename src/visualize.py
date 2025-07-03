import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

def visualize_segmentation(output, color_map):
    predicted_labels = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    segmented_image = np.zeros((predicted_labels.shape[0], predicted_labels.shape[1], 3), dtype=np.uint8)
    
    for label, color in enumerate(color_map):
        segmented_image[predicted_labels == label] = color

    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()

def save_segmentation(segmented_image, save_path):
    Image.fromarray(segmented_image).save(save_path)

def create_color_map(num_classes):
    color_map = []
    for i in range(num_classes):
        color_map.append(np.random.randint(0, 255, size=3).tolist())
    return color_map