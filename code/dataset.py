import pandas as pd
from transformers import AutoModel,AutoTokenizer
import torch, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from features import TEXT_COL, TARGET, LABEL_OFFSET, ID_COL, NUM_LABELS
from metrics import label_to_score

class EssayDataset(Dataset):
  def __init__(self, df, tokenizer, text_col="full_text", label_col="score", max_length=256, has_labels=True):
      self.df = df.reset_index(drop=True)
      self.tokenizer = tokenizer
      self.text_col = text_col
      self.label_col = label_col
      self.max_length = max_length
      self.has_labels = has_labels
      
      if self.has_labels:
         self.label_values = self.df[label_col].tolist()

  def __len__(self):
      return len(self.df)
  
  def __getitem__(self, idx):
     text = str(self.df.loc[idx, self.text_col]).strip()

     enc = self.tokenizer(
        text,
        truncation=True,
        max_length=self.max_length,
        padding="max_length",
     )

     item = {
        "input_ids" : torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask" : torch.tensor(enc["attention_mask"], dtype=torch.long),
     }

     if "token_type_ids" in enc:
        item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)

     if self.has_labels:
        item['labels'] = torch.tensor(label_to_score((self.label_values[idx])), dtype=torch.long)
    
     return item

