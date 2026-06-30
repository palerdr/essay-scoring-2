import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ID_COL = "essay_id"
TEXT_COL = "full_text"
SCORE_MIN = 1
SCORE_MAX = 6
VALID_TRUNCATION_STRATEGIES = {"first", "head_tail"}
KAGGLE_INPUT_DIR = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
KAGGLE_WORKING_DIR = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
AES2_ARTIFACT_DIR = os.environ.get("AES2_ARTIFACT_DIR")
SUBMISSION_PATH = KAGGLE_WORKING_DIR / "submission.csv"


class GruEssayClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        pad_token_id,
        embedding_dim,
        hidden_size,
        num_layers,
        bidirectional,
        dropout,
        task="classification",
    ):
        super().__init__()
        self.task = task
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        direction_count = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * direction_count, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        embedded = self.embedding(input_ids)
        if attention_mask is None:
            _, hidden = self.gru(embedded)
        else:
            lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)

        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.bidirectional else hidden[-1]
        return SimpleNamespace(logits=self.classifier(self.dropout(final_hidden)))


class EssayTestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, truncation_strategy="first"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = encode_text(
            tokenizer=self.tokenizer,
            text=str(self.df.loc[idx, TEXT_COL]).strip(),
            max_length=self.max_length,
            truncation_strategy=self.truncation_strategy,
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        return item


def encode_text(tokenizer, text, max_length, truncation_strategy="first"):
    if truncation_strategy not in VALID_TRUNCATION_STRATEGIES:
        raise ValueError(
            f"Unknown truncation_strategy={truncation_strategy!r}. "
            f"Expected one of {sorted(VALID_TRUNCATION_STRATEGIES)}."
        )

    text = str(text).strip()
    if truncation_strategy == "first":
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = tokenized["input_ids"]
    prefix_ids, suffix_ids = single_sequence_special_ids(tokenizer)
    content_length = max_length - len(prefix_ids) - len(suffix_ids)
    if content_length <= 0:
        raise ValueError(f"max_length={max_length} is too small for tokenizer special tokens.")

    if len(input_ids) > content_length:
        head_length = content_length // 2
        tail_length = content_length - head_length
        input_ids = input_ids[:head_length] + input_ids[-tail_length:]

    encoded_ids = prefix_ids + input_ids + suffix_ids
    enc = {"input_ids": encoded_ids[:max_length]}
    enc["attention_mask"] = [1] * len(enc["input_ids"])

    if "token_type_ids" in getattr(tokenizer, "model_input_names", []):
        enc["token_type_ids"] = [0] * len(enc["input_ids"])

    pad_length = max_length - len(enc["input_ids"])
    if pad_length > 0:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        pad_token_type_id = getattr(tokenizer, "pad_token_type_id", 0)
        if getattr(tokenizer, "padding_side", "right") == "left":
            enc["input_ids"] = [pad_id] * pad_length + enc["input_ids"]
            enc["attention_mask"] = [0] * pad_length + enc["attention_mask"]
            if "token_type_ids" in enc:
                enc["token_type_ids"] = [pad_token_type_id] * pad_length + enc["token_type_ids"]
        else:
            enc["input_ids"] = enc["input_ids"] + [pad_id] * pad_length
            enc["attention_mask"] = enc["attention_mask"] + [0] * pad_length
            if "token_type_ids" in enc:
                enc["token_type_ids"] = enc["token_type_ids"] + [pad_token_type_id] * pad_length

    return enc


def search_roots():
    roots = []
    if AES2_ARTIFACT_DIR:
        roots.append(Path(AES2_ARTIFACT_DIR))
    roots.append(KAGGLE_INPUT_DIR)

    unique_roots = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved.exists() and resolved not in seen:
            unique_roots.append(resolved)
            seen.add(resolved)
    return unique_roots


def describe_missing_artifact(roots):
    lines = [
        "No valid model artifact found.",
        "Expected a folder containing submission_config.json and model.safetensors.",
        f"Search roots: {[str(root) for root in roots]}",
    ]
    for root in roots:
        config_matches = sorted(root.glob("**/submission_config.json"))
        lines.append(f"submission_config.json under {root}: {[str(path) for path in config_matches[:10]]}")
        nearby = sorted(path for path in root.glob("*") if path.is_file() or path.is_dir())
        lines.append(f"top-level entries under {root}: {[path.name for path in nearby[:30]]}")
    return "\n".join(lines)


def find_model_artifact():
    matches_by_dir = {}
    roots = search_roots()
    for root in roots:
        for config_path in sorted(root.glob("**/submission_config.json")):
            config = json.loads(config_path.read_text())
            model_dir = config_path.parent
            has_hf_weights = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
            has_gru_weights = (model_dir / "model.pt").exists() and (model_dir / "gru_config.json").exists()
            if has_hf_weights or has_gru_weights:
                matches_by_dir[model_dir.resolve()] = (model_dir, config)
    matches = list(matches_by_dir.values())
    assert matches, describe_missing_artifact(roots)
    assert len(matches) == 1, f"Found multiple model artifacts: {[str(match[0]) for match in matches]}"
    return matches[0]


def find_test_csv():
    matches = []
    for path in sorted(KAGGLE_INPUT_DIR.glob("**/test.csv")):
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if ID_COL in columns and TEXT_COL in columns and "score" not in columns:
            matches.append(path)
    assert matches, "Attach the competition input Dataset; no matching test.csv found."
    return matches[0]


def logits_to_scores(logits, config):
    score_min = int(config.get("score_min", SCORE_MIN))
    score_max = int(config.get("score_max", SCORE_MAX))
    if config.get("task") == "regression":
        values = logits.squeeze(-1).detach().cpu().numpy()
        thresholds = config.get("thresholds")
        if thresholds is not None:
            return (np.digitize(values, np.asarray(thresholds, dtype=float)) + score_min).clip(score_min, score_max).astype(int).tolist()
        return np.rint(values).clip(score_min, score_max).astype(int).tolist()

    labels = logits.argmax(dim=-1).detach().cpu().numpy()
    return (labels + int(config.get("label_offset", 1))).clip(score_min, score_max).astype(int).tolist()


def single_sequence_special_ids(tokenizer):
    prefix = []
    suffix = []
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if cls_token_id is not None:
        prefix.append(cls_token_id)
    elif bos_token_id is not None:
        prefix.append(bos_token_id)

    if sep_token_id is not None:
        suffix.append(sep_token_id)
    elif eos_token_id is not None:
        suffix.append(eos_token_id)

    return prefix, suffix


def load_model(model_dir, config, tokenizer):
    if config["architecture"] == "transformer":
        return AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).float()

    if config["architecture"] == "gru":
        gru_config = json.loads((model_dir / "gru_config.json").read_text())
        model = GruEssayClassifier(**gru_config)
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        return model

    raise ValueError(f"Unknown architecture: {config['architecture']}")


def cuda_is_compatible():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return False

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    device_arch = f"sm_{capability[0]}{capability[1]}"
    arch_list = [arch for arch in torch.cuda.get_arch_list() if arch.startswith("sm_")]
    print(f"CUDA device: {device_name} ({device_arch}); torch CUDA arches: {arch_list or 'unknown'}")

    if arch_list and device_arch not in arch_list:
        print("This GPU is incompatible with the installed torch build.")
        print("On Kaggle, choose Accelerator -> GPU T4 x2 instead of GPU P100.")
        return False

    try:
        probe = torch.ones(1, device="cuda")
        _ = probe + 1
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"CUDA probe failed: {type(exc).__name__}: {exc}")
        print("On Kaggle, choose Accelerator -> GPU T4 x2 instead of GPU P100.")
        return False
    return True


def select_device():
    preference = os.environ.get("AES2_DEVICE")
    if preference is None:
        preference = "cuda" if os.environ.get("AES2_USE_CUDA") == "1" else "auto"
    preference = preference.lower()

    if preference == "cpu":
        return torch.device("cpu")
    if preference in {"auto", "cuda"} and cuda_is_compatible():
        return torch.device("cuda")

    require_cuda = os.environ.get("AES2_REQUIRE_CUDA", "0") == "1" or preference == "cuda"
    if require_cuda:
        raise RuntimeError(
            "A compatible CUDA GPU is required for this submission, but Kaggle did not provide one. "
            "Set the notebook accelerator to GPU T4 x2, not GPU P100."
        )

    print("Using CPU because a compatible CUDA device is not available.")
    return torch.device("cpu")


def main():
    KAGGLE_WORKING_DIR.mkdir(parents=True, exist_ok=True)
    model_dir, config = find_model_artifact()
    test_df = pd.read_csv(find_test_csv())

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = load_model(model_dir, config, tokenizer)
    device = select_device()
    print("Device:", device)
    model.to(device)
    model.eval()
    batch_size = int(os.environ.get("AES2_BATCH_SIZE", config.get("inference_batch_size", 8)))
    print("Batch size:", batch_size)

    loader = DataLoader(
        EssayTestDataset(
            test_df,
            tokenizer,
            int(config["max_length"]),
            config.get("truncation_strategy", "first"),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    scores = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="predict"):
            batch = {key: value.to(device) for key, value in batch.items()}
            scores.extend(logits_to_scores(model(**batch).logits, config))

    submission = pd.DataFrame({ID_COL: test_df[ID_COL].tolist(), "score": scores})
    submission.to_csv(SUBMISSION_PATH, index=False)
    assert list(submission.columns) == [ID_COL, "score"]
    assert submission["score"].between(int(config.get("score_min", 1)), int(config.get("score_max", 6))).all()
    print(submission.head())
    print("Wrote:", SUBMISSION_PATH)


main()
