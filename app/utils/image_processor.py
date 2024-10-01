# File: app/utils/image_processor.py

from PIL import Image
import torch
from torchvision import transforms

class ImageProcessor:
    def __init__(self, device):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to model's expected input size
            transforms.ToTensor(),          # Convert PIL Image to Tensor
            transforms.Normalize(           # Normalize as per model's requirements
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def process_images(self, image_paths):
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
            images.append(image)
        pixel_values = torch.stack(images).to(self.device)
        return pixel_values