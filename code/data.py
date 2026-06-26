from pathlib import Path
import kagglehub
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from metrics import score_to_label

ID_COL = "essay_id"
TEXT_COL = "full_text"
TARGET_COL = "score"

def load_competition_data(handle: str, input_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test from a local Kaggle input folder or the kagglehub cache."""
    local_dir = Path(input_dir) if input_dir is not None else Path(kagglehub.competition_download(handle))
    train = pd.read_csv(local_dir/"train.csv")
    test = pd.read_csv(local_dir/"test.csv")
    
    return train, test


def make_train_val_split(
    train_df: pd.DataFrame,
    target_col: str = TARGET_COL,
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


class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, text_col=TEXT_COL, label_col=TARGET_COL, max_length=256, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length
        self.has_labels = has_labels
        if self.has_labels:
            self.score_values = self.df[label_col].tolist()

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
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }

        if "token_type_ids" in enc:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)

        if self.has_labels:
            item["labels"] = torch.tensor(score_to_label(self.score_values[idx]), dtype=torch.long)

        return item
