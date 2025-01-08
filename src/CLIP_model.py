import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.image_embedding = config["image_embedding"]
        self.text_embedding = config["text_embedding"]
        self.temperature = nn.Parameter(torch.ones([], device=self.device) * np.log(1 / 0.07))
        self.image_projection = ProjectionHead(config, embedding_dim=self.image_embedding)
        self.text_projection = ProjectionHead(config, embedding_dim=self.text_embedding)

    def forward(self, sources, targets):
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(sources)    ## Project embeddings to 256 dimension space, shape: (batch_size, 256)
        text_embeddings = self.text_projection(targets)      ## Project embeddings to 256 dimension space, shape: (batch_size, 256)

        # L2 Normalization of embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Cosine similarity, multiplication is (batch_size, 256) @ (256, batch_size) = (batch_size, batch_size)
        logits = (text_embeddings @ image_embeddings.T) * torch.exp(self.temperature)

        # Defines the label index to be maximized on the diagonal (image 1 should match with text 1, ...)
        # Each label indicates the "correct" index in the logits row that should be maximized for each text-image pair
        # Create an array of indices from 0 to batch_size
        labels = torch.arange(logits.shape[0]).to(self.device) ## shape[0] is batch_size (64)

        # Calculate loss in both directions and average them
        # cross-entropy loss is used to maximize the similarity between matching pairs (diagonal elements of logits)
        # and minimize it for non-matching pairs (off-diagonal elements).
        texts_loss = F.cross_entropy(logits, labels)
        images_loss = F.cross_entropy(logits.T, labels)
        loss =  (images_loss + texts_loss) / 2.0

        return loss
    
class ProjectionHead(nn.Module):
    def __init__(self, config, embedding_dim):
        super().__init__()
        
        # Embedding dim is 2048 for image and 768 for text, projection_dim is 1024
        self.projection = nn.Linear(embedding_dim, config["projection_dim"])
        nn.init.xavier_normal_(self.projection.weight)

    def forward(self, x):
        return self.projection(x)
