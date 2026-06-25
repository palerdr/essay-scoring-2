from pathlib import Path
import kagglehub
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_competition_data(handle: str, input_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test from a local Kaggle input folder or the kagglehub cache."""
    local_dir = Path(input_dir) if input_dir is not None else Path(kagglehub.competition_download(handle))
    train = pd.read_csv(local_dir/"train.csv")
    test = pd.read_csv(local_dir/"test.csv")
    
    return train, test


def make_train_val_split(
    train_df: pd.DataFrame,
    target_col: str = "score",
    val_size: float = 0.2,
    seed: int = 42,
):
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df[target_col],
    )
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)

def make_folds(train_df, target_col, n_splits=15, seed=42):
  train_df["fold"] = -1
  skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
  for fold,(train_index, val_index) in enumerate(skf.split(train_df,train_df["score"])):
      train_df.loc[val_index,"fold"] = fold
  print('Train samples per fold:')
  train_df.fold.value_counts().sort_index()



  
