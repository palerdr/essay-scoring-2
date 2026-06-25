import random
import hydra
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from data import load_competition_data, make_train_val_split
from dataset import EssayDataset
from engine import evaluate, train_one_epoch
from features import TARGET
from models import build_model, build_tokenizer

def set_seed(seed: int)-> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@hydra.main(version_base=None, config_path='../conf', config_name="config")
def train(cfg: DictConfig) -> None:
    debug = cfg.debug.enabled
    m = cfg.model
    t = cfg.train
    set_seed(t.seed)
    EPOCHS = cfg.debug.epochs if debug else t.epochs

    train_df, test_df = load_competition_data(
        handle=cfg.dataset.competition_handle,
        input_dir=cfg.dataset.input_dir,
    )
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train columns: {list(train_df.columns)}")
    train_df, val_df = make_train_val_split(
        train_df=train_df,
        target_col=TARGET,
        val_size=t.val_size,
        seed=t.seed,
    )

    if debug:
        train_df = train_df.head(cfg.debug.max_train_rows).reset_index(drop=True)
        val_df = val_df.head(cfg.debug.max_val_rows).reset_index(drop=True)

        print(f"Debug train shape: {train_df.shape}")
        print(f"Debug val shape: {val_df.shape}")
        print(f"Debug train score counts:\n{train_df[TARGET].value_counts().sort_index()}")
        print(f"Debug val score counts:\n{val_df[TARGET].value_counts().sort_index()}")
    
        
    
    tokenizer = build_tokenizer(m.name)

    train_ds = EssayDataset(
        train_df,
        tokenizer,
        text_col="full_text",
        label_col=TARGET,
        max_length=m.max_length,
        has_labels=True,
    )

    val_ds = EssayDataset(
        val_df,
        tokenizer,
        text_col="full_text",
        label_col=TARGET,
        max_length=m.max_length,
        has_labels=True,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.debug.batch_size if debug else t.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.debug.batch_size if debug else t.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    if debug:
        first_batch = next(iter(train_loader))
        print(f"batch input_ids: {first_batch['input_ids'].shape}")
        print(f"batch attention_mask: {first_batch['attention_mask'].shape}")
        print(f"batch labels: {first_batch['labels'].shape}")
    
    # model = BertClassifier(
    #     model_name=m.name,
    #     num_labels=m.num_labels,
    # )
    
    #loss = nn.CrossEntropyLoss

    model = build_model(model_name = m.name, num_labels = m.num_labels)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_qwk = -1e8
    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_qwk = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            debug_max_steps=cfg.debug.max_steps if debug else None,
        )
        val_loss, val_qwk = evaluate(
            model=model,
            val_loader=val_loader,
            device=device,
            debug_max_steps=cfg.debug.max_steps if debug else None,
        )
        print(
            f"epoch {epoch + 1}/{cfg.debug.epochs if debug else t.epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_qwk={train_qwk:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_qwk={val_qwk:.4f}"
        )
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save(model.state_dict(), "best_model.pt")
            print(f"saved best_model.pt with val_qwk={best_val_qwk:.4f}")
        scheduler.step()
        
                
if __name__ == "__main__":
    train()
