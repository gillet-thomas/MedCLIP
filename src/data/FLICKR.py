import os
import cv2
import torch
import pandas as pd
from PIL import Image
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from src.CLIP_model import ImageEncoder, TextEncoder

class Flickr8kDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.device = config['device']
        self.batch_size = config['batch_size']
        
        # Initialize encoders
        self.image_encoder = ImageEncoder(config).to(self.device)
        self.text_encoder = TextEncoder(config).to(self.device)
        
        self.image_encoder.eval()
        self.text_encoder.eval()
        
        # Load and split data
        # data = self.get_data()
        # pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        file = open('./src/data/flickr.pickle', 'rb')
        data = pickle.load(file)
        self.train_data, self.val_data = torch.utils.data.random_split(data, [0.9, 0.1])
        
        self.data = self.train_data if mode == 'train' else self.val_data
        print(f"Data initialized: {len(self.data)} {mode} samples")
    
    def get_data(self):
        # Read captions file
        captions_path = os.path.join('src', 'data', 'captions.txt')
        df = pd.read_csv(captions_path, header=None, names=['image', 'caption'])
        
        image_caption_pairs = []
        images_path = os.path.join('src', 'data', 'Images')
        
        # Process each image-caption pair
        for idx, row in tqdm(df.iterrows()):
            image_path = os.path.join(images_path, row['image'])
            caption = row['caption']
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
                
            # Encode image and caption
            with torch.no_grad():
                image = self.load_image(image_path)
                encoded_image = self.image_encoder(image)
                encoded_caption = self.text_encoder(caption)
            
            # Store tensors on CPU to save GPU memory
            image_caption_pairs.append((encoded_image.squeeze(0).cpu(), encoded_caption.squeeze(0).cpu(), image_path, caption))
            
        return image_caption_pairs
    
    def load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = image.astype('float32') / 255.0
        image = torch.tensor(image).to(self.device)
        image = image.permute(2, 0, 1).float().unsqueeze(0)  # Shape: (1, 3, 512, 512) for ResNet encoder
        
        return image
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2], self.data[idx][3]
    
    def __len__(self):
        return len(self.data) // self.batch_size