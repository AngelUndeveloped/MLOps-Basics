"""
Inference module for the CoLA (Corpus of Linguistic Acceptability) task.

This module provides functionality for making predictions on new sentences
using a trained model. It handles the preprocessing of input text and
returns predictions with confidence scores for grammatical acceptability.

Classes:
    ColaPredictor: A class for making predictions on new sentences.
"""

import torch
from model import ColaModel
from data import DataModule


class ColaPredictor:
    """
    A class for making predictions on grammatical acceptability of sentences.

    This class loads a trained model and provides methods to predict whether
    a given sentence is grammatically acceptable or not. It handles the
    preprocessing of input text and returns predictions with confidence scores.

    Args:
        model_path (str): Path to the trained model checkpoint file.

    Attributes:
        model_path (str): Path to the trained model checkpoint file.
        model (ColaModel): The loaded and frozen model for inference.
        processor (DataModule): Data processor for tokenizing input text.
        softmax (torch.nn.Softmax): Softmax layer for converting logits to probabilities.
        labels (list): List of possible labels ["unacceptable", "acceptable"].

    Methods:
        predict: Make a prediction on a given sentence.
    """

    def __init__(self, model_path: str):
        """
        Initialize the ColaPredictor.

        Args:
            model_path (str): Path to the trained model checkpoint file.

        Note:
            The model is loaded, set to evaluation mode, and frozen to prevent
            any further training or gradient computation.
        """
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ["unacceptable", "acceptable"]

    def predict(self, text: str) -> list:
        """
        Make a prediction on a given sentence.

        Args:
            text (str): The input sentence to evaluate for grammatical acceptability.

        Returns:
            list: A list of dictionaries containing predictions and confidence scores.
                Each dictionary has the following keys:
                - "label": The predicted label ("unacceptable" or "acceptable")
                - "score": The confidence score for the prediction (float between 0 and 1)

        Example:
            >>> predictor = ColaPredictor("path/to/model.ckpt")
            >>> predictions = predictor.predict("The movie was good.")
            >>> print(predictions)
            [{"label": "acceptable", "score": 0.95}, {"label": "unacceptable", "score": 0.05}]
        """
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        # Convert to tensors and add batch dimension
        input_ids = torch.tensor([processed["input_ids"]])
        attention_mask = torch.tensor([processed["attention_mask"]])

        # Get model predictions
        logits = self.model(input_ids, attention_mask)
        scores = self.softmax(logits[0]).tolist()

        # Format predictions
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})

        return predictions


if __name__ == "__main__":
    # Example usage
    predictor = ColaPredictor(
        model_path="./models/epoch=0-step=267.ckpt"
    )
    predictions = predictor.predict("The movie was good.")
    print(predictions)
