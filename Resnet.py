import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import wandb
from tqdm import tqdm
from torchvision import models
from src.data.FLICKR import Flickr8kDataset


class ResNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Modify the final layer for your number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, config['num_classes'])
        self.model = self.model.to(self.device)
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transform (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize datasets
        train_dataset = Flickr8kDataset(config, mode="train", transforms=self.transform)
        val_dataset = Flickr8kDataset(config, mode="val", transforms=self.val_transform)
        
        # Initialize dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize optimizer with different learning rates for different layers
        params = [
            {'params': self.model.fc.parameters(), 'lr': config['learning_rate']},
            {'params': list(set(self.model.parameters()) - set(self.model.fc.parameters())), 
             'lr': config['learning_rate'] * 0.1}
        ]
        
        self.optimizer = torch.optim.AdamW(params, weight_decay=config['weight_decay'])
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize wandb
        if config['use_wandb']:
            wandb.init(project=config['wandb_project_name'])
            wandb.config.update(config)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}') as pbar:
            for batch_idx, (data, target, image_path, captions) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                print(output.shape, target.shape)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                total_loss += loss.item()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'lr': f'{current_lr:.6f}'
                })
                
                if self.config['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': loss.item(),
                        'train_acc': 100.*correct/total,
                        'learning_rate': current_lr,
                        'batch': batch_idx
                    })

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += self.criterion(output, target).item()
                    
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'val_loss': f'{val_loss/total:.4f}',
                        'val_acc': f'{100.*correct/total:.2f}%'
                    })
        
        val_loss /= len(self.val_loader)
        accuracy = 100. * correct / total
        
        print(f'\nValidation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        if self.config['use_wandb']:
            wandb.log({
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': accuracy
            })
        
        return val_loss, accuracy

    def train(self):
        best_val_loss = float('inf')
        best_accuracy = 0
        
        for epoch in range(self.config['epochs']):
            self.train_epoch(epoch)
            val_loss, accuracy = self.validate(epoch)
            
            # Save best model (based on validation loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'accuracy': accuracy
                }, os.path.join(self.config['save_dir'], 'best_model_loss.pth'))
            
            # Save best model (based on accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'accuracy': accuracy
                }, os.path.join(self.config['save_dir'], 'best_model_accuracy.pth'))
            
            # Regular checkpoint saving
            if (epoch + 1) % self.config['checkpoint_frequency'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'accuracy': accuracy
                }, os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))

# Example configuration
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 10,  # modify based on your dataset
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'checkpoint_frequency': 5,
    'save_dir': 'results/resnet',
    'use_wandb': False,
    'wandb_project_name': 'resnet50-training',
    'text_encoder': "distilbert-base-uncased",
    'text_embedding': 768
}

# Create trainer and start training
trainer = ResNetTrainer(config)
trainer.train()