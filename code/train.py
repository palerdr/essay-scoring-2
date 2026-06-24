import hydra
import torch, torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import load_competition_data
from omegaconf import DictConfig
from dataset import EssayDataset
from models import build_tokenizer, build_model, BertClassifier
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from features import TEXT_COL, TARGET, LABEL_OFFSET, ID_COL, NUM_LABELS

@hydra.main(version_base=None, config_path='../conf', config_name="config")
def train(cfg: DictConfig) -> None:
    debug = cfg.debug.enabled
    m = cfg.model

    train_df, test_df = load_competition_data(cfg.dataset.competition_handle)

    train_df, val_df = train_test_split(
        train_df,
        test_size=m.val_size,
        random_state=m.seed,
        stratify=train_df["score"],
    )

    EPOCHS = cfg.debug.epochs if debug else m.epochs
    BATCH = m.batch_size


    if debug:
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"Train columns: {train_df.columns}")
    
        train_df = train_df.head(cfg.debug.max_train_rows)
        val_df = val_df.head(cfg.debug.max_val_rows)
    
    tokenizer = build_tokenizer(m.name)

    train_dataset = EssayDataset(train_df, tokenizer, has_labels=True)
    val_dataset = EssayDataset(val_df, tokenizer, has_labels=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=(cfg.debug.batch_size if debug else BATCH),
        shuffle=True,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=(cfg.debug.batch_size if debug else BATCH),
        shuffle=False,
    )

    if debug:
        first_tensor = train_dataset[0]
        for key in ("input_ids", "attention_mask", "labels"):
            print(f"{key} : {first_tensor[key].shape}")
        print(f"labels min: {first_tensor['labels'].min().item()}")
        print(f"labels max: {first_tensor['labels'].max().item()}")
    
    # model = BertClassifier(
    #     model_name=m.name,
    #     num_labels=m.num_labels,
    # )
    
    #loss = nn.CrossEntropyLoss

    model = build_model(
        model_name = m.name,
        num_labels = m.num_labels,
    )

    optimizer = optim.AdamW(model.parameters(), lr=m.lr, weight_decay=m.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for _ in range(EPOCHS):
        print(f"starting epoch {_ + 1}/{EPOCHS}")
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            print(f"train loss: {loss.item()}")
            loss.backward()
            optimizer.step()
    
        model.eval()

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
        scheduler.step()
        print(f"finished epoch {_ + 1}/{EPOCHS}")
                
if __name__ == "__main__":
    train()