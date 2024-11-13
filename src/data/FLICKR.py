import os
import cv2
import torch
import pandas as pd
from PIL import Image
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from src.CLIP_model import ImageEncoder, TextEncoder
from collections import defaultdict

class Flickr8kDataset(Dataset):
    def __init__(self, config, mode='train', transforms=None):
        self.mode = mode
        self.config = config
        self.transforms = transforms
        self.device = config['device']
        self.batch_size = config['batch_size']
        # self.iterations_per_epoch = config['iterations_per_epoch']
        
        # Initialize encoders
        self.image_encoder = ImageEncoder(config).to(self.device)   ## Used in get_data()
        self.text_encoder = TextEncoder(config).to(self.device)     ## Used in get_data()
        self.image_encoder.eval()
        self.text_encoder.eval()
        
        # Load and split data
        # data = self.get_data()
        with open('./src/data/FLICKR_data.pickle', 'rb') as file:
            # pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            data = pickle.load(file)
            
        self.train_data, self.val_data = torch.utils.data.random_split(data, [0.95, 0.05])
        self.data = self.train_data if mode == 'train' else self.val_data
        print(f"Data initialized: {len(self.data)} {mode} samples")
    
    def get_data(self):
        # Read captions file
        captions_path = os.path.join('src', 'data', 'captions.txt')
        df = pd.read_csv(captions_path, header=None, names=['image', 'caption'])
        
        # Group captions by image
        image_captions = defaultdict(list)
        for _, row in df.iterrows():
            image_captions[row['image']].append(row['caption'].strip())

        encoded_data_pairs = []
        images_path = os.path.join('src', 'data', 'Images')
        for image_name, captions in tqdm(image_captions.items()):
            image_path = os.path.join(images_path, image_name)
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
                
            # Encode image and caption
            with torch.no_grad():
                image = self.load_image(image_path)
                encoded_image = self.image_encoder(image)
                
                # Encode all captions for this image
                combined_caption = " ".join(captions)
                encoded_captions = self.text_encoder(combined_caption)            ## Tensor shape (1, 768) -> (768)

            # Store tensors on CPU to save GPU memory
            # Data is [2048], [768], image_path, [caption1, caption2, ...] of type Tensor, Tensor, str, list
            encoded_data_pairs.append((encoded_image.squeeze(0).cpu(), encoded_captions.squeeze(0).cpu(), image_path, combined_caption))
            
        return encoded_data_pairs
    
    def load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = image.astype('float32') / 255.0
        image = torch.tensor(image).to(self.device)
        image = image.permute(2, 0, 1).float().unsqueeze(0)  # Shape: (1, 3, 224, 224) for ResNet encoder
        
        return image
    
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        image, encoded_caption, image_path, combined_caption = self.data[idx]   ## Shapes (2048), (768), str, list
        # encoded_caption = self.text_encoder(" ".join(combined_caption)).squeeze(0)    ## If using data_raw.pickle

        if self.transforms:
            image = self.transforms(image)

        return image, encoded_caption, image_path, combined_caption
    
    def __len__(self):
        # return self.batch_size * self.iterations_per_epoch if self.mode else len(self.data)
        return len(self.data)