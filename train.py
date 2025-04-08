import torch
import lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data import DataModule
from model import AGNewsModel

def main():
    """
    Main function to train the AG News classification model.
    
    This function:
    1. Initializes the model and data module
    2. Sets up callbacks for model checkpointing and early stopping
    3. Configures the PyTorch Lightning trainer
    4. Trains the model on the AG News dataset
    """
    # Initialize model and data module
    ag_news_model = AGNewsModel()
    ag_news_datamodule = DataModule()

    # Set up model checkpointing to save the best model during training
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",  # Directory to save model checkpoints
        monitor="val_loss",  # Metric to monitor for saving the best model
        mode="min",          # We want to minimize the validation loss
    )
    
    # Set up early stopping to prevent overfitting
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor for early stopping
        patience=3,          # Number of epochs to wait before stopping
        verbose=True,        # Print a message when early stopping is triggered
        mode="min",          # We want to minimize the validation loss
    )

    # Configure the PyTorch Lightning trainer
    trainer = pl.Trainer(
        default_root_dir="./logs",  # Directory to save logs
        gpus=(1 if torch.cuda.is_available() else 0),  # Use GPU if available
        max_epochs=5,                # Maximum number of training epochs
        fast_dev_run=False,          # Don't run in fast dev mode
        logger=TensorBoardLogger("logs/", name="ag-news-model", version=1),  # Use TensorBoard for logging
        callbacks=[checkpoint_callback, early_stopping_callback],  # Add callbacks
    )
    
    # Train the model
    try:
        trainer.fit(ag_news_model, ag_news_datamodule)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()