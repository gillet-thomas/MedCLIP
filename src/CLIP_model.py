import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
from torch.nn import init

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.image_embedding = config["image_embedding"]
        self.text_embedding = config["text_embedding"]
        self.temperature = nn.Parameter(torch.ones([], device=self.device) * np.log(1 / 0.07))

        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.image_projection = ProjectionHead(config, embedding_dim=self.image_embedding)
        self.text_projection = ProjectionHead(config, embedding_dim=self.text_embedding)

    def forward(self, sources, targets):
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(sources)    ## Project embeddings to 256 dimension space, shape: (batch_size, 256)
        text_embeddings = self.text_projection(targets)      ## Project embeddings to 256 dimension space, shape: (batch_size, 256)

        # L2 Normalization of embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Calculating the Loss, multiplication is (batch_size, 256) @ (256, batch_size) = (batch_size, batch_size)
        logits = (text_embeddings @ image_embeddings.T) * torch.exp(self.temperature)

        # Defines the label index to be maximized on the diagonal (image 1 should match with text 1, ...)
        # Each label indicates the "correct" index in the logits row that should be maximized for each text-image pair
        # Create an array of indices from 0 to batch_size
        labels = torch.arange(logits.shape[0]).to(self.device) ## shape[0] is batch_size (64)

        # Calculate loss in both directions and average them
        texts_loss = F.cross_entropy(logits, labels)
        images_loss = F.cross_entropy(logits.T, labels)
        loss =  (images_loss + texts_loss) / 2.0

        return loss
    
class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Load model + Freeze all base model parameters
        self.model = timm.create_model(config["image_encoder"], pretrained=True, num_classes=0)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
    
class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.model_name = config["text_encoder"]
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # Load model + Freeze all base model parameters
        self.model = DistilBertModel.from_pretrained(self.model_name).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        # Use the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0 ## Index 0 is CLS token represented by value 101

    def forward(self, text):
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')

        # Move to device
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, self.target_token_idx, :]       ## Output is shape (batch_size, hidden_size)

    
class ProjectionHead(nn.Module):
    def __init__(self, config, embedding_dim):
        super().__init__()
        
        # Embedding dim is 2048 for image and 768 for text, projection_dim is 256
        self.projection = nn.Linear(embedding_dim, config["projection_dim"])
        init.xavier_uniform_(self.projection.weight)

    def forward(self, x):
        return self.projection(x)