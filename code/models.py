from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch, torch.nn as nn

def build_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def build_model(model_name, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class BertClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = out.last_hidden_state[:,0,:]
        logits = self.head(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {
            "loss":loss,
            "logits":logits,
        }
