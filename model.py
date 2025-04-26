import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoModelForSequenceClassification
import torchmetrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class cola_model(L.LightningModule):
    """
    A PyTorch Lightning model for the CoLA (Corpus of Linguistic Acceptability) task.
    This model uses a pre-trained BERT model as a base and adds a classification head
    to determine if a sentence is linguistically acceptable or not.
    
    The model implements the complete training loop including:
    - Forward pass for predictions
    - Training step with loss computation
    - Validation step with accuracy tracking
    - Optimizer configuration
    
    Attributes:
        model_name (str): Name of the pre-trained BERT model to use
        lr (float): Learning rate for model training
        bert: The pre-trained BERT model
        w: Linear layer for classification
        num_classes (int): Number of output classes (2 for binary classification)
    """
    
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=2e-5):
        """
        Initialize the CoLA model.
        
        Args:
            model_name (str, optional): Name of the pre-trained BERT model to use.
                Defaults to a small BERT model ("google/bert_uncased_L-2_H-128_A-2").
            lr (float, optional): Learning rate for model training. Defaults to 2e-5,
                which is a common learning rate for fine-tuning transformer models.
        """
        # Initialize the parent LightningModule class
        super(cola_model, self).__init__()

        # Save all hyperparameters for logging and checkpointing
        self.save_hyperparameters()

        # Load the pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)

        # Add a linear classification layer on top of BERT
        # Input size is BERT's hidden size, output size is 2 for binary classification
        self.w = nn.Linear(self.bert.config.hidden_size, 2)

        # Set number of classes for the classification task
        self.num_classes = 2

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Tokenized input text converted to numerical IDs
            attention_mask (torch.Tensor): Binary mask indicating which tokens are actual words (1) vs padding (0)
            
        Returns:
            torch.Tensor: Raw logits for the two classes (acceptable/not acceptable)
        """
        # Process input through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the [CLS] token representation (first token) for classification
        # [CLS] token is designed to capture the meaning of the entire sentence
        h_cls = outputs.last_hidden_state[:, 0]
        
        # Pass through classification layer to get logits
        logits = self.w(h_cls)
        
        return logits

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        
        Args:
            batch (dict): Dictionary containing input_ids, attention_mask, and labels
            batch_idx (int): Index of the current batch
            
        Returns:
            torch.Tensor: Training loss for the current batch
        """
        # Get predictions and compute loss
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        
        # Log training loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        
        Args:
            batch (dict): Dictionary containing input_ids, attention_mask, and labels
            batch_idx (int): Index of the current batch
            
        Returns:
            None: Metrics are logged internally
        """
        # Get predictions and compute loss
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        
        # Compute accuracy
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(preds, batch["label"])
        val_acc = torch.tensor(val_acc)
        
        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_acc", val_acc, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer with the specified learning rate
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
