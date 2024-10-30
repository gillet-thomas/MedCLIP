import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel, ResNetConfig, ResNetModel

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.image_embedding = config["image_embedding"]
        self.text_embedding = config["text_embedding"]
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_projection = ProjectionHead(config, embedding_dim=self.image_embedding)
        self.text_projection = ProjectionHead(config, embedding_dim=self.text_embedding)

    def forward(self, sources, targets):
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(sources)    ## Project embeddings to 256 dimension space, shape: (batch_size, 256)
        text_embeddings = self.text_projection(targets)      ## Project embeddings to 256 dimension space, shape: (batch_size, 256)

        # # L2 Normalization of embeddings -> Already done in ProjectionHead
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # print(f"Image embeddings min and max values: {image_embeddings.min()}, {image_embeddings.max()}")
        # print(f"Text embeddings min and max values: {text_embeddings.min()}, {text_embeddings.max()}")

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T)  * torch.exp(self.temperature)

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

        self.trainable = config["trainable"]
        self.pretrained = config["pretrained"]
        self.model_name = config["image_encoder"]

        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)

    def forward(self, x):
        # # Get embedding and normalize
        # image_emb = self.model.image_projection(image_tensor)
        # image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        return self.model(x)
    
class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.trainable = config["trainable"]
        self.pretrained = config["pretrained"]
        self.model_name = config["text_encoder"]
        self.max_length = config["text_encoder_max_length"]
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # Load the model
        self.model = DistilBertModel.from_pretrained(self.model_name).to(self.device)

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0 ## Index 0 is CLS token represented by value 101

    def forward(self, text):
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        # Move to device
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)
        
        # Get the model output
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]  ## Output is shape (batch_size, hidden_size)
    
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
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected       ## Skip connection
        x = self.layer_norm(x)
        return x