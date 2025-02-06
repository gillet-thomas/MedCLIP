import torch
import torch.nn as nn
import pandas as pd

from glob import glob
from tqdm import tqdm
from vit_pytorch import ViT
from vit_pytorch.vit_3d import ViT
from vit_pytorch.simple_vit_3d import SimpleViT
from nilearn.image import load_img
from torch.utils.data import Dataset

class ABCDETimeDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.device = config['device']
        self.batch_size = config['batch_size']
        
        # Initialize encoders
        self.fmris_encoder = fmrisEncoder(config).to(self.device)   ## Used in get_data()
        self.pain_encoder = painEncoder(config).to(self.device)     ## Used in get_data()
        self.fmris_encoder.eval()
        self.pain_encoder.eval()
        
        # Load and split data
        data = self.get_data()
        # with open('./src/data/CLIP_FLICKR_base.pickle', 'rb') as file:
            # pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            # data = pickle.load(file)

        # self.train_data, self.val_data = torch.utils.data.random_split(data, [0.80, 0.20])
        # self.data = self.train_data if mode == 'train' else self.val_data
        # print(f"Data initialized: {len(self.data)} {mode} samples")
    
    def get_data(self):
        
        # Load fMRIs files
        fmris_path = sorted(glob('src/data/abcde/resampled/*.nii'))
        print(f"Found {len(fmris_path)} fMRI files")                    # 64 fmris of shape (64, 64, 33, 158)

        # Encode the fMRIs
        encoded_fmris = []
        for fmri_path in tqdm(fmris_path[:1]):
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.get_fdata()
            fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32).to(self.device)  # (64, 64, 33, 158) shape
            
            encoded_fmri = self.fmris_encoder(fmri_tensor)            # (1, 512) shape
            encoded_fmris.append(encoded_fmri)

        print(f"Encoded {len(encoded_fmris)} fMRIs")

        return None
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
class fmrisEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.to(config["device"])

        self.vit3d_encoder = Vit3DEncoder(config)
        self.time_encoder = TimeEncoder(config)
    
    def forward(self, x):
        #  x is fmri tensor of shape (64, 64, 33, 158)
        # Unbind the 4th dimension (timepoints dim)
        timepoints = x.unbind(3)                                # Result is tuple of 158 Tensors of shape (64, 64, 33)

        timepoints_encodings = []
        for timepoint in timepoints:
            timepoint = timepoint.unsqueeze(0)                  # ([1, 64, 64, 33])
            timepoint = timepoint.permute(0, 3, 1, 2)           # ([1, 33, 64, 64])
            timepoint = timepoint.unsqueeze(2)                  # Add timepoint dimension ([1, 33, 1, 64, 64])
            encoding = self.vit3d_encoder(timepoint)            # Encode each timepoint with 3D-ViT         
            timepoints_encodings.append(encoding)

        print(f"Total timepoints encoded: {len(timepoints_encodings)}")
        result = self.time_encoder(timepoints_encodings)         # Encode all timepoints with Transformer

        return result


class Vit3DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.v = SimpleViT(
            image_size = 64,           # image size
            channels = 33,             # number of channels
            frames = 1,                # number of frames
            frame_patch_size = 1,      # number of frames processed at once
            image_patch_size = 16,     # size of 2D patches extracted from each frame
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048
        )
            
    def forward(self, x):
        preds = self.v(x)   # (1, 1000)
        return preds
    
class TimeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):
        return x


class painEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.pain_embedding = config["pain_embedding"]

        self.encoder = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, self.pain_embedding)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
