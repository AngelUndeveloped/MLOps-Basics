import torch
from model import AGNewsModel
from data import DataModule


class AGNewsPredictor:
    """
    A class for making predictions on text using a trained AG News classification model.
    
    This class loads a trained model from a checkpoint and provides a simple interface
    for making predictions on new text samples.
    
    Attributes:
        model_path (str): Path to the model checkpoint
        model (AGNewsModel): The loaded PyTorch Lightning model
        processor (DataModule): Data processor for tokenizing input text
        softmax (nn.Softmax): Softmax layer for converting logits to probabilities
        labels (list): List of class labels for the AG News dataset
    """
    def __init__(self, model_path):
        """
        Initialize the AG News predictor.
        
        Args:
            model_path (str): Path to the model checkpoint file
        """
        self.model_path = model_path
        # Load the trained model from checkpoint
        self.model = AGNewsModel.load_from_checkpoint(model_path)
        # Set the model to evaluation mode
        self.model.eval()
        # Freeze the model parameters
        self.model.freeze()
        # Initialize the data processor
        self.processor = DataModule()  # Create an instance of DataModule
        # Initialize the softmax layer for converting logits to probabilities
        self.softmax = torch.nn.Softmax(dim=0)
        # Define the class labels for the AG News dataset
        self.labels = ["World", "Sports", "Business", "Science"]  # Fixed typo: "lables" -> "labels"

    def predict(self, text):
        """
        Make a prediction on a given text.
        
        Args:
            text (str): The input text to classify
            
        Returns:
            list: A list of dictionaries containing label and score pairs
        """
        # Create a sample dictionary with the input text
        inference_sample = {"text": text}
        
        # Tokenize the input text using the model's tokenizer
        # Note: We need to access the tokenizer from the processor
        processed = self.processor.tokenizer(
            inference_sample["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        # Make a prediction using the model
        with torch.no_grad():  # Disable gradient computation for inference
            logits = self.model(
                processed["input_ids"],
                processed["attention_mask"]
            )
        
        # Convert logits to probabilities using softmax
        scores = self.softmax(logits[0]).tolist()
        
        # Create a list of predictions with labels and scores
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})  # Fixed typo: "lable" -> "label"
        
        return predictions
    

if __name__ == "__main__":
    # Example usage of the AGNewsPredictor
    predictor = AGNewsPredictor(model_path="./models/model.pt")
    text = "The president of the United States is Joe Biden"
    predictions = predictor.predict(text)
    print(predictions)