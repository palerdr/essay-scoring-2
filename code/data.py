from pathlib import Path

import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset


ID_COL = "essay_id"
TEXT_COL = "full_text"
TARGET_COL = "score"


def load_competition_data(handle: str, input_dir: str | None = None):
    source = Path(input_dir) if input_dir else Path(kagglehub.competition_download(handle))
    train_df = pd.read_csv(source / "train.csv")
    test_df = pd.read_csv(source / "test.csv")
    return train_df, test_df


class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, truncation_strategy="first", has_labels=True, task="regression"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.truncation_strategy = truncation_strategy
        self.has_labels = has_labels
        self.task = task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_text = str(row[TEXT_COL]).strip()

        if self.truncation_strategy == "first":
            encoded = self.tokenizer(
                raw_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
        else:
            tokenized = self.tokenizer(raw_text, add_special_tokens=False, truncation=False)
            ids = tokenized["input_ids"]

            cls_or_bos = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
            sep_or_eos = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id
            prefix = [cls_or_bos] if cls_or_bos is not None else []
            suffix = [sep_or_eos] if sep_or_eos is not None else []

            keep = self.max_length - len(prefix) - len(suffix)
            if keep < 1:
                raise ValueError("max_length is too small for tokenizer special tokens.")

            if len(ids) > keep:
                left = keep // 2
                right = keep - left
                ids = ids[:left] + ids[-right:]

            ids = (prefix + ids + suffix)[: self.max_length]
            encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
            if "token_type_ids" in self.tokenizer.model_input_names:
                encoded["token_type_ids"] = [0] * len(ids)

            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            while len(encoded["input_ids"]) < self.max_length:
                encoded["input_ids"].append(pad_id)
                encoded["attention_mask"].append(0)
                if "token_type_ids" in encoded:
                    encoded["token_type_ids"].append(0)

        item = {k: torch.tensor(v, dtype=torch.long) for k, v in encoded.items()}

        if self.has_labels:
            score = int(row[TARGET_COL])
            if self.task == "regression":
                item["labels"] = torch.tensor(float(score), dtype=torch.float)
            else:
                item["labels"] = torch.tensor(score - 1, dtype=torch.long)

        return item
