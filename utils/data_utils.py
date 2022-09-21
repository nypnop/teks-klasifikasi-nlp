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
        data = pd.read_csv(path)
        data["label"] = data["label"].apply(lambda x: self.LABEL2INDEX[x])
        return data

    def __init__(self, path, tokenizer, no_special_tokens=False, *args,**kwargs):
        self.data = self.load_dataset(path)
        self.tokenizer = tokenizer
        self.no_special_tokens = no_special_tokens
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row["text_a"]
        label = row["label"]
        
        subword = self.tokenizer.encode(text, add_special_tokens=not self.no_special_tokens)
        mask = [1] * len(subword)
        token_type = [0] * len(subword)
        return subword, mask, token_type, label
    
class TextClassificationDataLoader(DataLoader):
    def __init__(self, dataset, max_len=512, *args,**kwargs):
        super(TextClassificationDataLoader, self).__init__(dataset, *args,**kwargs)
        self.collate_fn = self._collate_fn
        self.max_len = max_len
    
    def _collate_fn(self, batch):
        batch_size = len(batch)
        
        max_len = max([len(x[0]) for x in batch])
        max_len = min(self.max_len, max_len)
        

        subword_batch = np.zeros((batch_size, max_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_len), dtype=np.float32)
        token_type_batch = np.zeros((batch_size, max_len), dtype=np.int64)
        label_batch = np.zeros((batch_size), dtype=np.int64)
        
        seq_list = []
        
        for i, (subword, mask, token_type, label) in enumerate(batch):
            subword = subword[:max_len]
            subword_batch[i,:len(subword)] = subword
            mask_batch[i,:len(subword)] = 1
            # token_type_batch[i,:max_len] = token_type
            # label_batch[i] = label
            
            # seq_list.append(raw)
        
        return subword_batch, mask_batch, token_type_batch, label_batch, seq_list