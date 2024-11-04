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

    def run(self):
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

        print(f'CLIP model {sum(p.numel() for p in self.model.parameters())/1e6} M parameters running on {self.device}')
        print(f"Number of batches training: {len(self.dataloader)} of size {self.batch_size}")
        print(f"Number of batches validation: {len(self.val_dataloader)} of size {self.batch_size}")

        for epoch in tqdm(range(self.epochs)):
            self.train(epoch)
            self.validate(epoch)

        torch.save(self.model.state_dict(), './results/CLIP_Flickr.pth')
        print("Model saved to ./results/CLIP_Flickr.pth")

    def train(self, epoch):
        self.model.train()
        running_loss = 0.0

        learning_rate = self.config['learning_rate']
        val_interval = self.config['val_interval']
        weight_decay = self.config['weight_decay']
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        for i, (sources, targets, _, _) in enumerate(self.dataloader):
            sources, targets = sources.to(self.device), targets.to(self.device)  ## (batch_size, 2048) and (batch_size, 768)
            loss = self.model(sources, targets)
            
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
            for i, (sources, targets, _, _) in enumerate(self.dataloader):
                sources, targets = sources.to(self.device), targets.to(self.device)  ## (batch_size, 2048) and (batch_size, 768)
                loss = self.model(sources, targets)
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(self.val_dataloader)
            print(f"VALIDATION - Epoch {epoch}, Total batch {i}: validation loss {avg_val_loss}")
            wandb.log({"epoch": epoch, "val loss": avg_val_loss})