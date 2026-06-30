from pathlib import Path

import kagglehub
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from metrics import score_to_label

ID_COL = "essay_id"
TEXT_COL = "full_text"
TARGET_COL = "score"
TRUNCATION_MODES = {"first", "head_tail"}


def load_competition_data(handle: str, input_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kaggle train/test CSVs from a local folder or the kagglehub cache."""
    data_dir = Path(input_dir) if input_dir is not None else Path(kagglehub.competition_download(handle))
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")


def train_val_split(train_df: pd.DataFrame, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make the normal single holdout split, stratified by score."""
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df[TARGET_COL],
    )
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)


def add_folds(train_df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Return train_df with a stratified integer fold column."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    df = train_df.reset_index(drop=True).copy()
    df["row_idx"] = range(len(df))
    min_class_count = int(df[TARGET_COL].value_counts().min())
    if min_class_count < n_folds:
        raise ValueError(f"Cannot create {n_folds} folds; smallest score class has {min_class_count} rows.")

    df["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(df, df[TARGET_COL])):
        df.loc[val_idx, "fold"] = fold
    return df


def _special_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Return prefix/suffix special tokens for one text sequence."""
    prefix = []
    suffix = []
    if getattr(tokenizer, "cls_token_id", None) is not None:
        prefix.append(tokenizer.cls_token_id)
    elif getattr(tokenizer, "bos_token_id", None) is not None:
        prefix.append(tokenizer.bos_token_id)

    if getattr(tokenizer, "sep_token_id", None) is not None:
        suffix.append(tokenizer.sep_token_id)
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        suffix.append(tokenizer.eos_token_id)
    return prefix, suffix


def tokenize_essay(tokenizer, text: str, max_length: int, truncation_mode: str = "first") -> dict:
    """Tokenize one essay with either normal first-token or first+last-token truncation."""
    if truncation_mode not in TRUNCATION_MODES:
        raise ValueError(f"Unknown truncation_mode={truncation_mode!r}; expected {sorted(TRUNCATION_MODES)}.")

    text = str(text).strip()
    if truncation_mode == "first":
        return tokenizer(text, truncation=True, max_length=max_length, padding="max_length")

    prefix_ids, suffix_ids = _special_token_ids(tokenizer)
    content_length = max_length - len(prefix_ids) - len(suffix_ids)
    if content_length <= 0:
        raise ValueError(f"max_length={max_length} is too small for tokenizer special tokens.")

    input_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(input_ids) > content_length:
        head_length = content_length // 2
        tail_length = content_length - head_length
        input_ids = input_ids[:head_length] + input_ids[-tail_length:]

    input_ids = (prefix_ids + input_ids + suffix_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)

    pad_length = max_length - len(input_ids)
    if pad_length > 0:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        pad_type_id = getattr(tokenizer, "pad_token_type_id", 0)
        if getattr(tokenizer, "padding_side", "right") == "left":
            input_ids = [pad_id] * pad_length + input_ids
            attention_mask = [0] * pad_length + attention_mask
            token_type_ids = [pad_type_id] * pad_length + token_type_ids
        else:
            input_ids = input_ids + [pad_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            token_type_ids = token_type_ids + [pad_type_id] * pad_length

    encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in getattr(tokenizer, "model_input_names", []):
        encoded["token_type_ids"] = token_type_ids
    return encoded


class EssayDataset(Dataset):
    """PyTorch dataset for train/validation/test essays."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        truncation_mode: str = "first",
        has_labels: bool = True,
        task: str = "regression",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_mode = truncation_mode
        self.has_labels = has_labels
        self.task = task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = tokenize_essay(
            self.tokenizer,
            row[TEXT_COL],
            max_length=self.max_length,
            truncation_mode=self.truncation_mode,
        )
        item = {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}

        if self.has_labels:
            score = row[TARGET_COL]
            if self.task == "regression":
                item["labels"] = torch.tensor(float(score), dtype=torch.float)
            else:
                item["labels"] = torch.tensor(score_to_label(score), dtype=torch.long)
        return item
