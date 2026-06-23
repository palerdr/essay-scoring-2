from pathlib import Path
import kagglehub
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel,AutoTokenizer
import torch, torch.nn.functional as F
from tqdm import tqdm

def load_competition_data(handle:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test from the kagglehub cache. The competition_download call
      is a cache hit as long as download_datasets.py has run once."""
    local_dir = Path(kagglehub.competition_download(handle))
    train = pd.read_csv(local_dir/"train.csv")
    test = pd.read_csv(local_dir/"test.csv")
    
    return train, test

def k_fold(train):
  FOLDS = 15
  train["fold"] = -1
  skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
  for fold,(train_index, val_index) in enumerate(skf.split(train,train["score"])):
      train.loc[val_index,"fold"] = fold
  print('Train samples per fold:')
  train.fold.value_counts().sort_index()

class EmbedDataset(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, max_length):
      self.df = df.reset_index(drop=True)
      self.tokenizer = tokenizer
      self.max = max_length

  def __len__(self):
      return len(self.df)
  
  def __getitem__(self, idx):
     text = self.df.loc[idx, "fulltext"]
     tokens = self.tokenizer(
        text,
        None,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length = self.max,
        return_tensors='pt',
     )
     tokens = {k:v.squeeze(0) for k,v in tokens.items()}
     return tokens
    