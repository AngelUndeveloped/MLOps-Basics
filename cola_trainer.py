import torch
import wandb
import pandas as pd
import lightning as L
from lightning.callbacks import ModelCheckpoint
from lightning.callbacks.early_stopping import EarlyStopping
from lightning.loggers import wandbLogger

from data_cola_module import ColaDataModule
from cola_lightning_model import cola_model

wandb_logger = WandbLogger(project="MLOps-Basics")

def main():
    cola_data = ColaDataModule()
    cola_model = cola_model()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        monitor="val_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping()

    cola_trainer = L.Trainer(
        default_root_dir = "logs",
        gpus = (1 if torch.cuda.is_available() else 0),
        max_epochs = 1,
        fast_dev_run = True,
        logger=L.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback]
    )

    cola_trainer.fit(cola_model, cola_data)

if __name__=="__main__":
    main()