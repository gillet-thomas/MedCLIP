import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.image_embedding = config["image_embedding"]
        self.text_embedding = config["text_embedding"]
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.image_projection = ProjectionHead(config, embedding_dim=self.image_embedding)
        self.text_projection = ProjectionHead(config, embedding_dim=self.text_embedding)

    def forward(self, sources, targets):
        # # Getting Image and Text Embeddings (with same dimension)
        # sources = F.normalize(sources, dim=-1)
        # targets = F.normalize(targets, dim=-1)

        image_embeddings = self.image_projection(sources)    ## Project embeddings to 256 dimension space, shape: (batch_size, 256)
        text_embeddings = self.text_projection(targets)      ## Project embeddings to 256 dimension space, shape: (batch_size, 256)

        # L2 Normalization of embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) * torch.exp(self.temperature)

        # Defines the label index to be maximized on the diagonal (image 1 should match with text 1, ...)
        labels = torch.arange(logits.shape[0]).to(self.device)

        # Calculate loss in both directions and average them
        texts_loss = F.cross_entropy(logits, labels)
        images_loss = F.cross_entropy(logits.T, labels)
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)

        return loss
    
class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_name = config["image_encoder"]
        self.dropout = config["dropout"]
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        num_features = self.model.num_features

        for param in self.model.parameters():
            param.requires_grad = False

        self.fine_tune_layers = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(num_features // 2, num_features)
        )


    def forward(self, x):
        with torch.no_grad():  # Ensure base model doesn't compute gradients
            base_features = self.model(x)
            # fine_tuned_features = self.fine_tune_layers(base_features)
        return base_features
    
class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.dropout = config["dropout"]
        self.model_name = config["text_encoder"]
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # Load the model
        self.model = DistilBertModel.from_pretrained(self.model_name).to(self.device)
        hidden_size = self.model.config.hidden_size

        # Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.fine_tune_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0 ## Index 0 is CLS token represented by value 101

    def forward(self, text):
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')

        # Move to device
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)
        
        # Use torch.no_grad() for the base model inference
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = output.last_hidden_state[:, self.target_token_idx, :]       ## Output is shape (batch_size, hidden_size)
            
        # fine_tuned_embedding = self.fine_tune_layers(cls_embedding)
        return cls_embedding

    
class ProjectionHead(nn.Module):
    def __init__(self, config, embedding_dim):
        super().__init__()
        
        # Embedding dim is 2048 for image and 768 for text, projection_dim is 256
        self.projection_dim = config["projection_dim"]
        self.dropout = config["dropout"]

        self.projection = nn.Linear(embedding_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        # x = self.gelu(projected)
        # x = self.fc(x)
        # x = self.dropout(x)
        # x = x + projected       ## Skip connection
        x = self.layer_norm(projected)
        return x