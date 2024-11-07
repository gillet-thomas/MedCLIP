import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Normalize
from src.CLIP_model import ImageEncoder, TextEncoder
import pickle
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import random

class ImageNetDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.iterations_per_epoch = config['iterations_per_epoch']
        self.imagenet_labels_path = "./src/data/IMAGENET_labels.json"

        # Initialize encoders
        self.image_encoder = ImageEncoder(config).to(self.device)
        self.text_encoder = TextEncoder(config).to(self.device)
        self.image_encoder.eval()
        self.text_encoder.eval()

        # Define the preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Train is 1 281 167 samples, Val is 50 000 samples. Take 10% of the data for faster training
        self.data = torchvision.datasets.ImageNet(root='/mnt/data/iai/datasets/ImageNet', split=mode, transform=preprocess)
        self.data = torch.utils.data.Subset(self.data, range(0, len(self.data) // 10)) 

        self.load_class_description()
        print(f"Data initialized: {len(self.data)} {mode} samples and {len(self.class_descriptions)} classes")
    
    def load_class_description(self):
        with open(self.imagenet_labels_path, 'r') as f:
            data = json.load(f)
            self.class_descriptions = {int(key): value for key, value in data.items()}

        # Caption templates for variety
        self.templates = [
            "This is a photo of {}",
            "The image shows {}",
            "A picture containing {}",
            "This image features {}",
            "A {} can be seen in this photo",
            "The photograph displays {}",
            "In this image, there is {}",
            "We can see {} in this picture",
        ]
        
        # Additional details templates
        self.detail_templates = [
            " in a natural setting",
            " with interesting lighting",
            " from a close-up view",
            " from a distance",
            " in sharp focus",
            " in an outdoor environment",
            " in an indoor setting",
        ]
        
    def generate_captions(self, class_idx, num_captions=5):
        base_description = self.class_descriptions[class_idx]
        
        captions = []
        for _ in range(num_captions):
            # Randomly choose between short and detailed description
            description = random.choice(base_description)
            template = random.choice(self.templates)
            detail = random.choice(self.detail_templates)
            
            caption = template.format(description) + detail
            captions.append(caption)
            
        return captions
    
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        image, target = self.data[idx]                                               ## self.data[idx] returns a tuple (image, target)
        captions = self.generate_captions(target)

        encoded_image = self.image_encoder(image.unsqueeze(0).to(self.device))
        encoded_image = encoded_image.squeeze(0)                                     ## Tensor shape (1, 2048) -> (2048)

        combined_caption = ". ".join(captions)
        encoded_captions = self.text_encoder(combined_caption).squeeze(0)           ## Tensor shape (1, 768) -> (768)

        return encoded_image, encoded_captions, image, target                       ## Shapes (2048), (768), Tensor (3, 224, 224), int
    
    def __len__(self):
        # return self.batch_size * self.iterations_per_epoch if self.mode else len(self.data)
        return len(self.data)