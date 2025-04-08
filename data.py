from sklearn.model_selection import train_test_split
from datasets import Dataset
import lightning as pl
import pandas as pd
import torch

from datasets import load_dataset
from transformers import AutoTokenizer

class DataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling the AG News dataset.
    This class manages data loading, preprocessing, and creating DataLoaders for training and validation.
    """
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        """
        Initialize the DataModule with a specific BERT model and batch size.
        
        Args:
            model_name (str): The name of the pre-trained BERT model to use for tokenization
            batch_size (int): The batch size for training and validation DataLoaders
        """
        super().__init__

        self.batch_size = batch_size
        # Initialize the tokenizer for the specified BERT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        """
        Load and prepare the AG News dataset.
        This method:
        1. Loads the dataset from Hugging Face
        2. Converts it to a pandas DataFrame
        3. Splits it into training and validation sets
        4. Converts the splits back to Hugging Face Dataset format
        """
        # Load the AG News dataset and convert to pandas DataFrame
        ag_news_df = load_dataset("wangrongsheng/ag_news", split="train").to_pandas()
        # Split the data into training (80%) and validation (20%) sets
        ag_news_train_df, ag_news_validation_df = train_test_split(ag_news_df, test_size=0.2, random_state=42)
        # Convert pandas DataFrames back to Hugging Face Dataset format
        self.train_data= Dataset.from_pandas(ag_news_train_df)
        self.val_data= Dataset.from_pandas(ag_news_validation_df)

    def tokenize_data(self, example):
        """
        Tokenize a single example from the dataset.
        
        Args:
            example (dict): A dictionary containing the text and label
            
        Returns:
            dict: Tokenized text with attention masks and other BERT-specific features
        """
        return self.tokenizer(
            example["text"],
            truncation=True,  # Truncate sequences longer than max_length
            padding="max_length",  # Pad sequences shorter than max_length
            max_length=512  # BERT's maximum sequence length
        )

    def setup(self, stage=None):
        """
        Set up the dataset for training or validation.
        This method:
        1. Tokenizes all examples in both training and validation sets
        2. Converts the data to PyTorch tensors
        3. Sets up the format for model input
        
        Args:
            stage (str): The stage of training ('fit' or None)
        """
        # Setup only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            # Tokenize and format training data
            self.train_data = self.train_data.map(self.tokenize_data,batched=True)
            self.train_data.set_format(
                type="torch",
                columns=['text','label']
            )

            # Tokenize and format validation data
            self.val_data = self.val_data.map(self.tokenize_data,batched=True)
            self.val_data.set_format(
                type="torch",
                columns=['text','label']
            )

    def train_dataloader(self):
        """
        Create a DataLoader for the training set.
        
        Returns:
            DataLoader: A PyTorch DataLoader for the training data
        """
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True  # Shuffle the data for each epoch
        )

    def val_dataloader(self):
        """
        Create a DataLoader for the validation set.
        
        Returns:
            DataLoader: A PyTorch DataLoader for the validation data
        """
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False  # Don't shuffle validation data
        )

if __name__ == "__main__":
    # Example usage of the DataModule
    data_module = DataModule()
    data_module.prepare_data()
    data_module.setup()
    # Print the shape of the first batch of text data
    print(next(iter(data_module.train_dataloader()))["text"].shape)
