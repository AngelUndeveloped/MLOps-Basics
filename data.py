"""
Data module for the CoLA (Corpus of Linguistic Acceptability) dataset.

This module handles the loading, preprocessing, and batching of the CoLA dataset
for training and evaluating language models. It uses the Hugging Face datasets
library and transformers tokenizer to prepare the data for model training.

The CoLA dataset consists of English sentences labeled as either grammatically
correct or incorrect. It is commonly used for evaluating the grammatical
understanding capabilities of language models.

Classes:
    DataModule: A LightningDataModule for handling the CoLA dataset.

Attributes:
    model_name (str): Name of the pre-trained model to use for tokenization.
        Defaults to "google/bert_uncased_L-2_H-128_A-2".
    batch_size (int): Number of samples per batch. Defaults to 32.
    max_length (int): Maximum sequence length for tokenization. Defaults to 128.
"""

import torch
import lightning as L
from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(L.LightningDataModule):
    """
    A LightningDataModule for handling the CoLA dataset.

    This class manages the loading, preprocessing, and batching of the CoLA dataset
    for training and evaluating language models. It inherits from LightningDataModule
    and implements the necessary methods for data handling in PyTorch Lightning.

    The class handles:
    - Dataset loading and splitting
    - Tokenization of text data
    - Creation of PyTorch DataLoaders
    - Data formatting for model input

    Args:
        model_name (str): Name of the pre-trained model to use for tokenization.
            Defaults to "google/bert_uncased_L-2_H-128_A-2".
        batch_size (int): Number of samples per batch. Defaults to 32.
        max_length (int): Maximum sequence length for tokenization. Defaults to 128.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer instance for text preprocessing.
        train_data (Dataset): The training dataset.
        val_data (Dataset): The validation dataset.
        batch_size (int): Number of samples per batch.
        max_length (int): Maximum sequence length for tokenization.

    Methods:
        prepare_data: Downloads and prepares the dataset.
        setup: Sets up the dataset for training, validation, and testing.
        tokenize_data: Tokenizes a single example from the dataset.
        train_dataloader: Returns the training dataloader.
        val_dataloader: Returns the validation dataloader.
    """
    def __init__(
            self,
            model_name: str = "google/bert_uncased_L-2_H-128_A-2",
            batch_size: int = 32,
            max_length: int = 128
        ):
        """
        Initialize the DataModule.

        Args:
            model_name (str): Name of the pre-trained model to use for tokenization.
                Defaults to "google/bert_uncased_L-2_H-128_A-2".
            batch_size (int): Number of samples per batch. Defaults to 32.
            max_length (int): Maximum sequence length for tokenization. Defaults to 128.

        Note:
            The tokenizer is initialized during __init__ to ensure it's available
            for all data processing steps.
        """
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.train_data = None
        self.val_data = None

    def prepare_data(self):
        """
        Download and prepare the CoLA dataset.

        This method loads the CoLA dataset from the Hugging Face datasets library
        and splits it into training and validation sets. The data is stored
        as instance variables for later use in the setup method.

        Note:
            This method is called only once, typically on a single GPU.
            It should not be used to assign state (self.x = y).
        """
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        """
        Tokenize a single example from the CoLA dataset.

        Args:
            example (dict): A dictionary containing the sentence to tokenize.
                Expected keys:
                - "sentence": The text to tokenize (str)

        Returns:
            dict: A dictionary containing:
                - input_ids: Token IDs for the input sequence
                - attention_mask: Attention mask for the sequence
                - Other tokenizer outputs as specified by the tokenizer

        Note:
            The tokenization uses the max_length parameter specified during initialization.
            Longer sequences will be truncated, and shorter ones will be padded.
        """
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        """
        Set up the dataset for training, validation, and testing.

        This method is called after prepare_data and is used to prepare the data
        for each stage of training. It tokenizes the data and sets the format to
        PyTorch tensors.

        Args:
            stage (str, optional): The stage of training. Can be "fit", "test", or None.
                Defaults to None.

        Note:
            This method is called on every GPU in distributed training.
            It should be used to assign state (self.x = y).
        """
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
               type="torch",
               columns=["input_ids", "attention_mask", "label"],
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        """
        Create and return the training data loader.

        Returns:
            torch.utils.data.DataLoader: A DataLoader instance for the training data
                with the specified batch size and shuffling enabled.

        Note:
            The training data is shuffled to ensure good generalization.
        """
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Create and return the validation data loader.

        Returns:
            torch.utils.data.DataLoader: A DataLoader instance for the validation data
                with the specified batch size and shuffling disabled.

        Note:
            The validation data is not shuffled to maintain consistent evaluation.
        """
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
        )

if __name__=="__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
