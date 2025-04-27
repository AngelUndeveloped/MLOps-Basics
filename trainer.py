"""
Training module for the CoLA (Corpus of Linguistic Acceptability) task.

This module handles the training process for the CoLA model, including:
- Model and data initialization
- Callback configuration (checkpointing, early stopping)
- Weights & Biases logging
- Training visualization

Classes:
    SampleVisualizationLogger: A callback for logging misclassified examples.
"""

import torch
import wandb
import pandas as pd
import lightning as L
from lightning.callbacks import ModelCheckpoint
from lightning.callbacks.early_stopping import EarlyStopping
from lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel


class SampleVisualizationLogger(L.Callback):
    """
    A callback for logging misclassified examples during validation.
    
    This callback logs examples where the model's predictions differ from
    the ground truth labels to Weights & Biases for analysis.
    
    Args:
        datamodule (DataModule): The data module containing validation data.
    """
    
    def __init__(self, datamodule):
        """
        Initialize the SampleVisualizationLogger.
        
        Args:
            datamodule (DataModule): The data module containing validation data.
        """
        super().__init__()
        self.datamodule = datamodule
    
    def on_validation_end(self, trainer, pl_module):
        """
        Called at the end of each validation epoch.
        
        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (ColaModel): The model being trained.
            
        Note:
            Logs a table of misclassified examples to Weights & Biases,
            including the original sentence, true label, and predicted label.
        """
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentence = val_batch["sentence"]
        
        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentence, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_html=True),
                "global_step": trainer.global_step,
            }
        )


def main():
    """
    Main training function for the CoLA model.
    
    This function:
    1. Initializes the data and model
    2. Sets up callbacks for checkpointing and early stopping
    3. Configures Weights & Biases logging
    4. Creates and runs the trainer
    
    The training process includes:
    - Model checkpointing based on validation loss
    - Early stopping to prevent overfitting
    - Logging of misclassified examples
    - Anomaly detection for debugging
    """
    # Initialize data and model
    cola_data = DataModule()
    cola_model = ColaModel()  # Fixed incorrect model instantiation

    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint.ckpt",
        monitor="valid/loss",
        mode="min"
    )

    # Configure early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss",
        patience=3,
        verbose=True,
        mode="min"
    )

    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(
        project="MLOps-Basics",
        entity="angel_undeveloped-n-a"
    )

    # Create trainer with callbacks and configuration
    cola_trainer = L.Trainer(
        max_epochs=1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            SampleVisualizationLogger(cola_data)
        ],
        log_every_n_steps=10,
        detect_anomaly=True,
        # limit_train_batches=0.25,  # Uncomment to limit training data
        # limit_val_batches=0.25,   # Uncomment to limit validation data
    )

    # Start training
    cola_trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()