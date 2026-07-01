from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name, num_labels, task="regression", architecture="transformer", **_):
    if architecture != "transformer":
        raise ValueError("This repository currently keeps inference and training on HF transformer models only.")

    problem_type = "regression" if task == "regression" else "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    return model.float()
