import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import ResNetConfig, ResNetModel

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.image_embedding = config["image_embedding"]
        self.text_embedding = config["text_embedding"]
        self.temperature = nn.Parameter(torch.ones([], device=self.device) * np.log(1 / 0.07))

        self.image_encoder = ImageEncoder2(config)
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
    
class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Load model + Freeze all base model parameters
        self.model = timm.create_model(config["image_encoder"], pretrained=True, num_classes=0)
        # for param in self.model.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder2(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Configure the ResNet model and load pretrained weights
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")

        # Freeze the base model parameters ## no difference
        # for i, param in enumerate(self.model.parameters()):
        #         param.requires_grad = False

        # Add a fine-tuning head
        self.fine_tuning_head = nn.Sequential(
            nn.Linear(config["image_embedding"], config["projection_dim"]),
            nn.ReLU(),
            nn.Linear(config["projection_dim"], config["projection_dim"])
        )

        # Initialize the fine-tuning head ## no difference
        # for param in self.fine_tuning_head.parameters():
        #     if isinstance(param, nn.Linear):
        #         nn.init.xavier_normal_(param.weight)
        #         nn.init.zeros_(param.bias)

    def forward(self, x):
        # Pass the input through the base ResNet model
        output = self.model(x)

        # Pass the output through the fine-tuning head
        output = self.fine_tuning_head(output.pooler_output)

        return output
    
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
        nn.init.xavier_normal_(self.projection.weight)

    def forward(self, x):
        return self.projection(x)
