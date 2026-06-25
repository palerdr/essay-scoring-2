import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

COMPETITION_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
CHECKPOINT_PATH = "/kaggle/input/essay-scoring-checkpoint/best_model.pt"
SUBMISSION_PATH = "/kaggle/working/submission.csv"

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 256
NUM_LABELS = 6
BATCH_SIZE = 8
TEXT_COL = "full_text"
ID_COL = "essay_id"
LABEL_OFFSET = 1


def label_to_score(label):
    return int(label) + LABEL_OFFSET


class EssayTestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, TEXT_COL]).strip()
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        return item


test_df = pd.read_csv(f"{COMPETITION_DIR}/test.csv")
print(f"test shape: {test_df.shape}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_ds = EssayTestDataset(test_df, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

pred_scores = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        pred_labels = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
        pred_scores.extend([label_to_score(x) for x in pred_labels])

submission = pd.DataFrame({
    "essay_id": test_df[ID_COL].tolist(),
    "score": pred_scores,
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"wrote {SUBMISSION_PATH}")
print(submission.head())
print(submission.shape)
