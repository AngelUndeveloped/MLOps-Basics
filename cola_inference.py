import torch
from cola_lightning_model import cola_model
from data_cola_module import ColaDataModule

class cola_predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        # Loading the trained model
        self.model = cola_model.load_from_checkpoint(model_path)
        # keep the model in evaluation mode
        self.model.eval()
        self.model.freeze()
        self.processor = ColaDataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ["unacceptable", "acceptable"]
    
    def predict(self, text):
        # text => run time input
        inference_sample = {"sentence": text}
        # tokenize the text input
        processed = self.processor.tokenize_data(inference_sample)
        # predictions
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]])
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"lable":label, "score":score})
        return predictions
    
if __name__== "__main__":
    sentence = "The movie was good."
    predictor = cola_predictor(
        model_path = "models/cola_model.pth"
    )
    predictions = predictor.predict(sentence)
    print(predictions)
