from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def build_model(
    model_name,
    num_labels,
    task="classification",
    architecture="transformer",
    tokenizer=None,
    gru_config=None,
):
    if architecture == "gru":
        if tokenizer is None:
            raise ValueError("A tokenizer is required to build the GRU model.")
        if gru_config is None:
            raise ValueError("GRU config is required to build the GRU model.")
        return GruEssayClassifier(
            vocab_size=len(tokenizer),
            num_labels=num_labels,
            pad_token_id=tokenizer.pad_token_id,
            embedding_dim=gru_config.embedding_dim,
            hidden_size=gru_config.hidden_size,
            num_layers=gru_config.num_layers,
            bidirectional=gru_config.bidirectional,
            dropout=gru_config.dropout,
            task=task,
        )

    if architecture != "transformer":
        raise ValueError(f"Unknown model architecture: {architecture}")

    problem_type = "regression" if task == "regression" else "single_label_classification"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    return model.float()


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
        self.num_labels = num_labels
        self.task = task
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id,
        )
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
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.regression_loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
        embedded = self.embedding(input_ids)

        if attention_mask is None:
            _, hidden = self.gru(embedded)
        else:
            lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(
                embedded,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)

        if self.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]

        logits = self.classifier(self.dropout(final_hidden))
        loss = None
        if labels is not None:
            if self.task == "regression":
                loss = self.regression_loss_fn(logits.squeeze(-1).float(), labels.float())
            else:
                loss = self.classification_loss_fn(logits.float(), labels)

        return SimpleNamespace(loss=loss, logits=logits)
