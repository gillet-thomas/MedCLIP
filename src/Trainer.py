import wandb
import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, config, model, dataset_train, dataset_val):
        self.config = config
        self.data = dataset_train
        self.val_data = dataset_val
        self.device = config['device']
        self.model = model.to(self.device)

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'CLIP model total parameters: {total_params/1e6:.2f}M (trainable {trainable_params/1e6:.2f}M and frozen {(total_params-trainable_params)/1e6:.2f}M)')
        print(f"Number of batches training: {len(self.dataloader)} of size {self.batch_size}")          ## 114 batches of size 64
        print(f"Number of batches validation: {len(self.val_dataloader)} of size {self.batch_size}")    ## 13 batches of size 64

    def run(self):
        for epoch in tqdm(range(self.epochs)):
            self.train(epoch)
            self.validate(epoch)

        torch.save(self.model.state_dict(), './results/CLIP_ImageNet.pth')
        print("Model saved to ./results/CLIP_ImageNet.pth")

    def train(self, epoch):
        self.model.train()
        running_loss = 0.0

        learning_rate = self.config['learning_rate']
        val_interval = self.config['val_interval']
        weight_decay = self.config['weight_decay']
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)

        item = next(iter(self.dataloader))
        print(f"Check point 0: {item[0].shape} and {item[1].shape}, {item[2].shape}, {item[3].shape}")
        
        for i, (sources, targets, _, _) in enumerate(self.dataloader):
            sources, targets = sources.to(self.device), targets.to(self.device)  ## (batch_size, 2048) and (batch_size, 768)
            print(f"Check point 1: {sources.shape} and {targets.shape}")
            loss = self.model(sources, targets)
            print(f"Check point 2: {loss}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            running_loss += loss.item()

            if i != 0 and i % val_interval == 0:
                print(f"Epoch {epoch}, Batch {i}: train loss {running_loss/val_interval}")
                wandb.log({"epoch": epoch, "batch": i, "train loss": running_loss/val_interval})
                running_loss = 0.0


    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i, (sources, targets, _, _) in enumerate(self.val_dataloader):
                sources, targets = sources.to(self.device), targets.to(self.device)  ## (batch_size, 2048) and (batch_size, 768)
                loss = self.model(sources, targets)
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(self.val_dataloader)
            print(f"VALIDATION - Epoch {epoch}, Total batch {i}: validation loss {avg_val_loss}")
            wandb.log({"epoch": epoch, "val loss": avg_val_loss})