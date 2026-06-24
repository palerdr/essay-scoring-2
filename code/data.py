from pathlib import Path
import kagglehub
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel,AutoTokenizer
import torch, torch.nn.functional as F

def load_competition_data(handle:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test from the kagglehub cache. The competition_download call
      is a cache hit as long as download_datasets.py has run once."""
    local_dir = Path(kagglehub.competition_download(handle))
    train = pd.read_csv(local_dir/"train.csv")
    test = pd.read_csv(local_dir/"test.csv")
    
    return train, test

def k_fold(train_df, target_col, n_splits=15, seed=42):
  train_df["fold"] = -1
  skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
  for fold,(train_index, val_index) in enumerate(skf.split(train_df,train_df["score"])):
      train_df.loc[val_index,"fold"] = fold
  print('Train samples per fold:')
  train_df.fold.value_counts().sort_index()

def train_val_split(train_df, target_col, val_size, seed):
   ...

  