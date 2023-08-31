import train
import wandb

cfg = 'configs/15classes_resnet18.yaml'
model = train(cfg, dataLoader, model, optimizer)

config = {
    "project": "15classes-classifier",
    "num_of_classes": 15
}
run = wandb.init(project = config["project"], config = config)