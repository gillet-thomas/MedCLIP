import torch
import torch.nn as nn
import pandas as pd

from glob import glob
from tqdm import tqdm
from nilearn.image import load_img
from torch.utils.data import Dataset

class ABCDEDataset(Dataset):
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

        self.train_data, self.val_data = torch.utils.data.random_split(data, [0.80, 0.20])
        self.data = self.train_data if mode == 'train' else self.val_data
        print(f"Data initialized: {len(self.data)} {mode} samples")
    
    def get_data(self):
        
        # Load fMRIs files
        fmris_path = sorted(glob('src/data/abcde/resampled/*.nii'))
        print(f"Found {len(fmris_path)} fMRI files")                    # 64 fmris of shape (64, 64, 33, 158)

        # Encode the fMRIs
        encoded_fmris = []
        for fmri_path in tqdm(fmris_path):
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.get_fdata()
            fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32).to(self.device)  # (64, 64, 33, 158) shape
            averaged_fmri = torch.mean(fmri_tensor, dim=3)              # Average across timepoints (64, 64, 33, 158) to (64, 64, 33)
            averaged_fmri = averaged_fmri.unsqueeze(0)                  # Add batch dimension (1, 64, 64, 33) shape
            encoded_fmri = self.fmris_encoder(averaged_fmri)            # (1, 512) shape
            encoded_fmris.append(encoded_fmri)

        print(f"Encoded {len(encoded_fmris)} fMRIs")

        # Load pain scores
        pain_path = 'src/data/abcde/pain_scores.xlsx'
        pain_df = pd.read_excel(pain_path, header=[0, 1])                           # 64 rows, 30 columns
        print(f"Loaded pain scores: {pain_df.shape}")

        # Preprocess pain scores
        pain_df.columns = [' '.join(col).strip() for col in pain_df.columns.values] # Flatten multi-level columns
        pain_df.dropna(subset=['Participant ID'], inplace=True)                     # Drop rows with missing participant ID
        pain_df = pain_df.astype('float32')                                         # Convert all columns to float
        pain_df=(pain_df-pain_df.min())/(pain_df.max()-pain_df.min())               # Normalize pain scores
        pain_df = torch.tensor(pain_df.values).to(self.device)                      # (64, 30) shape

        # Encode pain scores
        encoded_pain = []
        for pain in tqdm(pain_df):
            pain = self.pain_encoder(pain)                                  # (1, 512) shape
            encoded_pain.append(pain)
        
        print(f"Encoded {len(encoded_pain)} pain scores")
        
        return list(zip(encoded_fmris, encoded_pain))
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
class fmrisEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.fmris_embedding = config["fmris_embedding"]

        self.encoder = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            # Second 3D conv block
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            # Third 3D conv block
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
        )
        
         # Calculate the flattened size
        with torch.no_grad():
            # Assuming input size of (64, 64, 33)
            dummy_input = torch.zeros(1, 1, 64, 64, 33)
            dummy_output = self.encoder(dummy_input)
            flattened_size = dummy_output.numel() // dummy_output.size(0)  # Get the number of elements in the tensor, 32768
        
        # Final embedding layer
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, self.fmris_embedding),
            nn.ReLU()
        )

        self.to(config["device"])
    
    def forward(self, x):
        # x is averaged_fmri of shape (1, 64, 64, 33)
        # Reshape to (batch_size, channels, depth, height, width)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)      # ([1, 1, 33, 64, 64])
        x = self.encoder(x)                         # 3D CNN encoder ([1, 128, 4, 8, 8])
        
        # Flatten and get embedding
        x = x.flatten(1)                            # ([1, 32768])
        x = self.fc(x)                              # ([1, 512])
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
    
