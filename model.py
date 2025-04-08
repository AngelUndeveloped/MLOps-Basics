import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.metrics import accuracy_score

class AGNewsModel(pl.LightningModule):
    """
    A PyTorch Lightning module for AG News text classification using BERT.
    
    This model uses a pre-trained BERT model as a feature extractor and adds
    a classification head on top for the AG News dataset classification task.
    
    Attributes:
        bert: The pre-trained BERT model
        W: Linear layer for classification
        num_classes: Number of output classes (4 for AG News)
    """
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        """
        Initialize the AG News classification model.
        
        Args:
            model_name (str): Name of the pre-trained BERT model to use
            lr (float): Learning rate for the optimizer
        """
        super(AGNewsModel, self).__init__()
        self.save_hyperparameters()
        
        # Initialize BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Initialize classification head
        # Note: Using bert.config.hidden_size to match BERT's output dimension
        self.W = nn.Linear(self.bert.config.hidden_size, 4)  # 4 classes for AG News
        self.num_classes = 4  # Fixed number of classes for AG News dataset

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs from BERT tokenizer
            attention_mask (torch.Tensor): Attention mask from BERT tokenizer
            
        Returns:
            torch.Tensor: Logits for classification
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation for classification
        h_cls = outputs.last_hidden_state[:, 0]  # Take first token ([CLS]) representation
        
        # Project to classification space
        logits = self.W(h_cls)
        return logits
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch (dict): Batch of data containing input_ids, attention_mask, and labels
            batch_idx (int): Index of the batch
            
        Returns:
            torch.Tensor: Training loss
        """
        # Forward pass
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        
        # Calculate loss
        loss = F.cross_entropy(logits, batch["label"])
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch (dict): Batch of data containing input_ids, attention_mask, and labels
            batch_idx (int): Index of the batch
        """
        # Forward pass
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        
        # Calculate loss
        loss = F.cross_entropy(logits, batch["label"])
        
        # Get predictions
        _, preds = torch.max(logits, dim=1)
        
        # Calculate accuracy (using torch operations for GPU compatibility)
        val_acc = (preds == batch["label"]).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Returns:
            torch.optim.Optimizer: The Adam optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])