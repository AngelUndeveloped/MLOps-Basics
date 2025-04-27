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
        f1_metric (torchmetrics.F1): F1 score metric
        precision_macro_metric (torchmetrics.Precision): Macro-averaged precision
        recall_macro_metric (torchmetrics.Recall): Macro-averaged recall
        precision_micro_metric (torchmetrics.Precision): Micro-averaged precision
        recall_micro_metric (torchmetrics.Recall): Micro-averaged recall
    """
    
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
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

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name=model_name,
            num_labels=2,
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro",
            num_classes=self.num_classes,
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro",
            num_classes=self.num_classes,
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Tokenized input sequence
            attention_mask (torch.Tensor): Attention mask for the input sequence
            labels (torch.Tensor, optional): Ground truth labels for training.
                Defaults to None.
        
        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: Model outputs
                containing logits and loss (if labels are provided)
        """
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        
        Args:
            batch (dict): A batch of training data containing:
                - input_ids: Tokenized input sequences
                - attention_mask: Attention masks
                - label: Ground truth labels
            batch_idx (int): Index of the current batch
        
        Returns:
            torch.Tensor: Training loss for the current batch
        
        Note:
            Logs training loss and accuracy to the logger.
        """
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        
        Args:
            batch (dict): A batch of validation data containing:
                - input_ids: Tokenized input sequences
                - attention_mask: Attention masks
                - label: Ground truth labels
            batch_idx (int): Index of the current batch
        
        Returns:
            dict: Dictionary containing labels and logits for the current batch
        
        Note:
            Logs various validation metrics including loss, accuracy, precision,
            recall, and F1 score to the logger.
        """
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of a validation epoch.
        
        Args:
            outputs (list): List of outputs from validation_step
        
        Note:
            Logs a confusion matrix to Weights & Biases for visualization.
            The confusion matrix is computed using the model's predictions
            and ground truth labels from the entire validation set.
        """
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        # Log confusion matrix using W&B
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        # )

    def configure_optimizers(self):
        """
        Configure the optimizer for model training.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer with the specified learning rate
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])