"""
Model module for the CoLA (Corpus of Linguistic Acceptability) task.

This module implements a PyTorch Lightning model for binary classification of
grammatical acceptability in sentences. It uses a pre-trained BERT model as
the base architecture and adds appropriate metrics for model evaluation.

Classes:
    ColaModel: A PyTorch Lightning model for grammatical acceptability classification.
"""

import torch
import wandb
# import numpy as np
# import pandas as pd
import lightning as L
from transformers import AutoModelForSequenceClassification
import torchmetrics
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
from typing import Dict, List, Optional, Union, Any
from torch import Tensor


class ColaModel(L.LightningModule):
    """
    A PyTorch Lightning model for the CoLA (Corpus of Linguistic Acceptability) task.
    
    This model uses a pre-trained BERT model as a base and adds a classification head
    to determine if a sentence is linguistically acceptable or not. It implements
    the complete training loop including forward pass, loss computation, and metrics
    tracking.
    
    The model tracks various metrics during training and validation:
    - Accuracy
    - F1 Score
    - Precision (macro and micro)
    - Recall (macro and micro)
    - Confusion Matrix (via Weights & Biases)
    
    Attributes:
        model_name (str): Name of the pre-trained BERT model to use
        lr (float): Learning rate for model training
        bert: The pre-trained BERT model
        num_classes (int): Number of output classes (2 for binary classification)
        train_accuracy_metric (torchmetrics.Accuracy): Training accuracy metric
        val_accuracy_metric (torchmetrics.Accuracy): Validation accuracy metric
        f1_metric (torchmetrics.F1Score): F1 score metric
        precision_macro_metric (torchmetrics.Precision): Macro-averaged precision
        recall_macro_metric (torchmetrics.Recall): Macro-averaged recall
        precision_micro_metric (torchmetrics.Precision): Micro-averaged precision
        recall_micro_metric (torchmetrics.Recall): Micro-averaged recall
    """

    def __init__(self, model_name: str = "google/bert_uncased_L-2_H-128_A-2", lr: float = 3e-5):
        """
        Initialize the CoLA model.
        
        Args:
            model_name (str, optional): Name of the pre-trained BERT model to use.
                Defaults to a small BERT model ("google/bert_uncased_L-2_H-128_A-2").
            lr (float, optional): Learning rate for model training. Defaults to 3e-5.
        
        Note:
            The model is initialized with a pre-trained BERT model and various
            metrics for tracking model performance during training and validation.
        """
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        try:
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained model {model_name}: {str(e)}")

        self.num_classes = 2
        
        # Initialize metrics with proper task specification
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.F1Score(task="binary")
        self.precision_macro_metric = torchmetrics.Precision(task="binary", average="macro")
        self.recall_macro_metric = torchmetrics.Recall(task="binary", average="macro")
        self.precision_micro_metric = torchmetrics.Precision(task="binary", average="micro")
        self.recall_micro_metric = torchmetrics.Recall(task="binary", average="micro")

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Optional[Tensor] = None) -> Dict[str, Any]:
        """
        Forward pass of the model.
        
        Args:
            input_ids (Tensor): Tokenized input sequence
            attention_mask (Tensor): Attention mask for the input sequence
            labels (Tensor, optional): Ground truth labels for training.
                Defaults to None.
        
        Returns:
            Dict[str, Any]: Model outputs containing logits and loss (if labels are provided)
        
        Raises:
            ValueError: If input tensors have incorrect shapes or types
        """
        if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
            raise ValueError("input_ids and attention_mask must be torch.Tensor")
        
        if labels is not None and not isinstance(labels, Tensor):
            raise ValueError("labels must be torch.Tensor if provided")
        
        try:
            outputs = self.bert(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            return outputs
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {str(e)}")

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for the model.
        
        Args:
            batch (Dict[str, Tensor]): A batch of training data containing:
                - input_ids: Tokenized input sequences
                - attention_mask: Attention masks
                - label: Ground truth labels
            batch_idx (int): Index of the current batch
        
        Returns:
            Tensor: Training loss for the current batch
        
        Note:
            Logs training loss and accuracy to the logger.
        """
        try:
            outputs = self.forward(
                batch["input_ids"], batch["attention_mask"], labels=batch["label"]
            )
            preds = torch.argmax(outputs.logits, 1)
            train_acc = self.train_accuracy_metric(preds, batch["label"])
            
            self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
            self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
            
            return outputs.loss
        except Exception as e:
            raise RuntimeError(f"Training step failed: {str(e)}")

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Validation step for the model.
        
        Args:
            batch (Dict[str, Tensor]): A batch of validation data containing:
                - input_ids: Tokenized input sequences
                - attention_mask: Attention masks
                - label: Ground truth labels
            batch_idx (int): Index of the current batch
        
        Returns:
            Dict[str, Tensor]: Dictionary containing labels and logits for the current batch
        
        Note:
            Logs various validation metrics including loss, accuracy, precision,
            recall, and F1 score to the logger.
        """
        try:
            labels = batch["label"]
            outputs = self.forward(
                batch["input_ids"], batch["attention_mask"], labels=labels
            )
            preds = torch.argmax(outputs.logits, 1)

            # Compute metrics
            valid_acc = self.val_accuracy_metric(preds, labels)
            precision_macro = self.precision_macro_metric(preds, labels)
            recall_macro = self.recall_macro_metric(preds, labels)
            precision_micro = self.precision_micro_metric(preds, labels)
            recall_micro = self.recall_micro_metric(preds, labels)
            f1 = self.f1_metric(preds, labels)

            # Log metrics
            self.log("valid/loss", outputs.loss, prog_bar=True, on_epoch=True)
            self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
            self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
            self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
            self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
            self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
            self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
            
            return {"labels": labels, "logits": outputs.logits}
        except Exception as e:
            raise RuntimeError(f"Validation step failed: {str(e)}")

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """
        Called at the end of a validation epoch.
        
        Args:
            outputs (List[Dict[str, Tensor]]): List of outputs from validation_step
        
        Note:
            Logs a confusion matrix to Weights & Biases for visualization.
            The confusion matrix is computed using the model's predictions
            and ground truth labels from the entire validation set.
        """
        try:
            labels = torch.cat([x["labels"] for x in outputs])
            logits = torch.cat([x["logits"] for x in outputs])
            
            # Move tensors to CPU for numpy conversion
            labels = labels.cpu()
            logits = logits.cpu()
            
            # Log confusion matrix using W&B
            self.logger.experiment.log(
                {
                    "conf": wandb.plot.confusion_matrix(
                        probs=logits.numpy(), y_true=labels.numpy()
                    )
                }
            )
        except Exception as e:
            raise RuntimeError(f"Validation epoch end failed: {str(e)}")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for model training.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer with the specified learning rate
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
