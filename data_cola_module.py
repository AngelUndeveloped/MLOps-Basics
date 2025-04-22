"""
Data module for the CoLA (Corpus of Linguistic Acceptability) dataset.

This module handles the loading, preprocessing, and batching of the CoLA dataset
for training and evaluating language models. It uses the Hugging Face datasets
library and transformers tokenizer to prepare the data for model training.

Attributes:
    model_name (str): Name of the pre-trained model to use for tokenization.
        Defaults to "google/bert_uncased_L-2_H-128_A-2".
    batch_size (int): Number of samples per batch. Defaults to 32.
"""

import torch
# import datasets
import lightning as L

from datasets import load_dataset
from transformers import AutoTokenizer

class ColaDataModule(L.LightningDataModule):
    """
    A LightningDataModule for handling the CoLA dataset.

    This class manages the loading, preprocessing, and batching of the CoLA dataset
    for training and evaluating language models. It inherits from LightningDataModule
    and implements the necessary methods for data handling in PyTorch Lightning.

    Args:
        model_name (str): Name of the pre-trained model to use for tokenization.
            Defaults to "google/bert_uncased_L-2_H-128_A-2".
        batch_size (int): Number of samples per batch. Defaults to 32.

    Methods:
        prepare_data: Downloads and prepares the dataset.
        setup: Sets up the dataset for training, validation, and testing.
        train_dataloader: Returns the training dataloader.
        val_dataloader: Returns the validation dataloader.
        test_dataloader: Returns the test dataloader.
    """
    def __init__(self, model_name: str = "google/bert_uncased_L-2_H-128_A-2", batch_size: int = 32):
        """
        Initialize the ColaDataModule.

        Args:
            model_name (str): Name of the pre-trained model to use for tokenization.
                Defaults to "google/bert_uncased_L-2_H-128_A-2".
            batch_size (int): Number of samples per batch. Defaults to 32.
        """
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_data = None
        self.val_data = None

    def prepare_data(self):
        """
        Download and prepare the CoLA dataset.

        This method loads the CoLA dataset from the Hugging Face datasets library
        and splits it into training, validation, and test sets. The data is stored
        as instance variables for later use in the setup method.

        Note:
            This method is called only once, typically on a single GPU.
        """
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        """
        Tokenize a single example from the CoLA dataset.

        Args:
            example (dict): A dictionary containing the sentence to tokenize.

        Returns:
            dict: A dictionary containing the tokenized sentence with input_ids,
                attention_mask, and other tokenizer outputs.
        """
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=256,
            # return_tensors="pt" # this is not needed because the tokenizer returns a tensor
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
            )

    def train_data_loader(self):
        """
        Create and return the training data loader.

        Returns:
            torch.utils.data.DataLoader: A DataLoader instance for the training data
                with the specified batch size and shuffling enabled.
        """
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_data_loader(self):
        """
        Create and return the validation data loader.

        Returns:
            torch.utils.data.DataLoader: A DataLoader instance for the validation data
                with the specified batch size and shuffling disabled.
        """
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
        )
