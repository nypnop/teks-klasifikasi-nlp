import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TextClassificationDataset(Dataset):
    #Static Constants Variable
    LABEL2INDEX = {"no":0, "yes":1}
    INDEX2LABEL = {0: "no", 1:"yes"}
    NUM_LABELS = 2
    
    def load_dataset(self, path):
        data = pd.read_csv(path, sep="\t", header=None)
        data.columns = ["text_a", "label"]
        data["label"] = data["label"].map(self.LABEL2INDEX)
        return data

    def __init__(self, path, tokenizer, max_len):
        self.data = self.load_dataset(path)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row["text"]
        label = row["label"]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class TextClassificationDataLoader(DataLoader):
    def __init__(self, path, tokenizer, max_len, batch_size):
        dataset = TextClassificationDataset(path, tokenizer, max_len)
        super(TextClassificationDataLoader, self).__init__(dataset, batch_size=batch_size, num_workers=4)
    
    def collate_fn(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }