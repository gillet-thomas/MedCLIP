import sys
import yaml
import wandb
import torch

from src.Trainer import Trainer
from src.CLIP_model import CLIP
from src.CLIP_retrieval import CLIPRetrieval
from src.CLIP_retrievalIN import CLIPRetrievalIN
from src.data.FLICKR import Flickr8kDataset
from src.data.ImageNet import ImageNetDataset

if __name__ == "__main__":
    device = 'cuda:3' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'
    config = yaml.safe_load(open("./configs/config.yaml"))
    config["device"] = device
    print(f"Device: {device}")
    torch.manual_seed(config["seed"])

    # Initialize wandb
    args = sys.argv[1:]
    name = args[0] if len(args) > 0 else None
    wandb_mode = 'online' if config["wandb_enabled"] == 1 else 'disabled'
    wandb.init(project="CLIP_MIMIC_CXR", mode=wandb_mode, config=config, name=name)

    if config['training_enabled']:
        dataset_train = ImageNetDataset(config, mode="train")
        dataset_val = ImageNetDataset(config, mode="val")
        model = CLIP(config)
        trainer = Trainer(config, model, dataset_train, dataset_val)
        trainer.run()
    else:
        print("Training is disabled. Inference mode enabled.")
        dataset = ImageNetDataset(config, mode="val")
        model = CLIP(config).to(device)
        model.load_state_dict(torch.load('./results/CLIP_ImageNet.pth', map_location=device, weights_only=True))
        retrieval = CLIPRetrievalIN(config, model, dataset)
        results = retrieval.retrieve_similar_content()