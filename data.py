import torch
import datasets
import lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "google/bert_uncased_L-2_H-128_A-2"
batch_size = 32
test_size_split = 0.2
random_state_val = 42

class DataModule(pl.LightningDataModule):
    def __init__(self, model_name=model_name, batch_size=batch_size):
        super().__init__

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        ag_news_test_dataset = load_dataset("wangrongsheng/ag_news",split="test")
        ag_news_df = load_dataset("wangrongsheng/ag_news", split="train").to_pandas()
        ag_news_train_df, ag_news_validation_df = train_test_split(ag_news_df, test_size=test_size_split, random_state=random_state_val)
        ag_news_train_dataset = Dataset.from_pandas(ag_news_train_df)
        ag_news_validation_dataset = Dataset.from_pandas(ag_news_validation_df)

    def tokenize_data(self, example):
        return self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    