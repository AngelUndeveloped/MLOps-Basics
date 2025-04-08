import torch
import lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import AGNewsModel

def main():
    ag_news_model = AGNewsModel()
    ag_news_datamodule = DataModule()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        monitor="val_loss",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        default_root_dir="./logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=5,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("logs/", name="ag-news-model", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(ag_news_model, ag_news_datamodule)

    if __name__ == "__main__":
        main()