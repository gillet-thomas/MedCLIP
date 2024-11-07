import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Normalize
from src.CLIP_model import ImageEncoder, TextEncoder
import pickle
import torchvision.transforms as transforms
from tqdm import tqdm

class ImageNetDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.iterations_per_epoch = config['iterations_per_epoch']

        # Initialize image encoder
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
        
        file = open('./src/data/imagenet.pickle', 'wb')
        file_val = open('./src/data/imagenet_val.pickle', 'wb')
        # self.train_data = pickle.load(file)            ## 1 281 167 training samples
        # self.val_data = pickle.load(file_val)          ## 50 000 validation samples
        self.train_data = torchvision.datasets.ImageNet(root='/mnt/data/iai/datasets/ImageNet', split='train', transform=preprocess)
        self.val_data = torchvision.datasets.ImageNet(root='/mnt/data/iai/datasets/ImageNet', split='val', transform=preprocess)
        pickle.dump(self.train_data, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.val_data, file_val, protocol=pickle.HIGHEST_PROTOCOL)


        # Initialize a layer
        layer = torch.nn.Linear(128, 64)
        # Apply Xavier uniform initialization
        torch.nn.init.xavier_uniform_(layer.weight)
        print(layer.weight)  # See the randomly assigned weights

        self.data = self.train_data if mode == 'train' else self.val_data
        print(f"Data initialized: {len(self.data)} {mode} samples")
    
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        clean_image = self.data[idx][0]
        clean_target = self.data[idx][1]

        encoded_image = self.image_encoder(clean_image.unsqueeze(0).to(self.device))            ## Tensor shape (1, 3, 224, 224)
        # encoded_text = self.text_encoder(clean_text)
        encoded_target = torch.tensor([clean_target])      ## Clean target is the class index, no need to encode it

        print(f"Encoded image shape: {encoded_image.shape}, Encoded text shape: {encoded_target.shape}, type: {clean_target},")

        return encoded_image, encoded_target, clean_image, clean_target
    
    def __len__(self):
        # return self.batch_size * self.iterations_per_epoch if self.mode else len(self.data)
        return len(self.data)