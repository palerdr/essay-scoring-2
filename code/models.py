from transformers import AutoTokenizer, AutoModelForSequenceClassification

def build_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def build_model(model_name, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
